#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.qos
import numpy as np
import math
import threading
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header, Bool
import tf2_ros

# ── PointCloud2 helper ────────────────────────────────────────────────────────
def _make_cloud(header: Header, points: np.ndarray) -> PointCloud2: # simple PointCloud2 builder for XYZ float32 data
    # This function creates a PointCloud2 message from a numpy array of points represent the paint deposits. 
    msg = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = len(points)
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = 12 * len(points)
    msg.is_dense     = True
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = points.astype(np.float32).tobytes()
    return msg


# ── Voxel grid for spatial deduplication ─────────────────────────────────────
class VoxelGrid: # simple 3D voxel grid to track which regions have already received paint deposits, preventing infinite density and improving performance by skipping points in already-painted voxels. The cell size is configurable, and the grid allows one "new" point per voxel per tick to enable accumulation over time.
    def __init__(self, cell_size: float):
        self._inv  = 1.0 / cell_size
        # map: voxel_key -> last_seen_tick
        # When a tick is provided to is_new(), we treat a voxel as "new"
        # once per tick (allowing accumulation across multiple ticks).
        self._seen: dict = {}

    def is_new(self, x: float, y: float, z: float, tick: int = None) -> bool:
        key = (int(math.floor(x * self._inv)),
               int(math.floor(y * self._inv)),
               int(math.floor(z * self._inv)))
        if tick is None:
            # legacy behaviour: first time ever seen -> new, else not new
            if key in self._seen:
                return False
            self._seen[key] = -1
            return True
        else:
            last = self._seen.get(key, None)
            if last == tick:
                return False
            # mark as seen in this tick and allow future ticks to add again
            self._seen[key] = tick
            return True

    def __len__(self):
        return len(self._seen)

# ── Main node ─────────────────────────────────────────────────────────────────
class SpraySimNode(Node):

    def __init__(self):
        super().__init__('spray_sim_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('end_effector_frame',   'link_6')
        self.declare_parameter('world_frame',          'world')
        self.declare_parameter('gz_world_name',        'world_demo')
        self.declare_parameter('cone_length',          0.30)   # ST-6 ø1.0mm: 300mm rated standoff
        self.declare_parameter('cone_half_angle_deg',  18.4)   # ST-6 ø1.0mm: fan 200mm wide at 300mm → atan(100/300)
        self.declare_parameter('sigma',                0.035)  # ST-6 ø1.0mm: SMD 15µm → radial σ ≈ 35mm at rated standoff
        self.declare_parameter('num_sample_rings',     8)
        self.declare_parameter('num_angular_pts',      36)
        self.declare_parameter('paint_point_spacing',  0.010)  # ST-6 ø1.0mm: tighter than ø2.0mm; ~10mm voxel spacing
        self.declare_parameter('min_weight',           0.10)
        self.declare_parameter('max_paint_points',     30000)
        self.declare_parameter('max_gz_spheres',       8000)
        self.declare_parameter('gz_sphere_radius',     0.008)
        self.declare_parameter('publish_rate_hz',      10.0)
        self.declare_parameter('gz_spawn_every_n',     5)
        self.declare_parameter('enforce_min_eef_speed', True)

        self._ee_frame    = self.get_parameter('end_effector_frame').value
        self._world_frame = self.get_parameter('world_frame').value
        self._gz_world    = self.get_parameter('gz_world_name').value
        self._cone_len    = self.get_parameter('cone_length').value
        self._half_ang    = math.radians(self.get_parameter('cone_half_angle_deg').value)
        self._sigma       = self.get_parameter('sigma').value
        self._n_rings     = self.get_parameter('num_sample_rings').value
        self._n_ang       = self.get_parameter('num_angular_pts').value
        self._pt_spacing  = self.get_parameter('paint_point_spacing').value
        self._min_weight  = self.get_parameter('min_weight').value
        self._max_pts     = self.get_parameter('max_paint_points').value
        self._max_gz      = self.get_parameter('max_gz_spheres').value
        self._gz_radius   = self.get_parameter('gz_sphere_radius').value
        self._enforce_min = self.get_parameter('enforce_min_eef_speed').value
        rate_hz           = self.get_parameter('publish_rate_hz').value
        self._gz_every_n  = self.get_parameter('gz_spawn_every_n').value
    

        # ── Paint state ──────────────────────────────────────────────────────
        self._voxel = VoxelGrid(self._pt_spacing)
        self._paint_cloud: list = []

        # Incremental RViz marker state
        # _pending_pts accumulate until published; committed holds

        self._pending_pts:   list = []
        self._committed_pts: list = []

        # Heartbeat: re-send full cloud to RViz every ~1 s so late-joining
        # or restarted RViz instances catch up.
        self._heartbeat_ticks = max(1, int(rate_hz))

        self._spray_on = False
        self._cone_pts = self._build_cone_sample_pts()
        self._tick     = 0

        # ── EEF speed gate ───────────────────────────────────────────────────
        self._prev_eef_pos:  np.ndarray = None   # last known EEF world position
        self._eef_speed:     float      = 0.0    # m/s estimated from TF delta
        self._MIN_EEF_SPEED: float      = 0.003  # m/s — below this = "still"

        # ── TF2 ─────────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Publishers ──────────────────────────────────────────────────────
        self._marker_pub = self.create_publisher(
            MarkerArray, '/spray/cone_markers', 10) # RViz visualization of the spray cone and paint points
        self._cloud_pub  = self.create_publisher(
            PointCloud2, '/spray/paint_points', 10) # raw point cloud of paint deposits for external tools / rosbag

        # ── Spray subscriber ─────────────────────────────────────────────────
        spray_qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, '/spray/active',
                                 self._spray_active_cb, spray_qos)
        
        self.create_subscription(Float32MultiArray, '/spray/rl_action', self.rl_action_callback, 10)
        
        self._timer = self.create_timer(1.0 / rate_hz, self._timer_cb)

    def destroy_node(self):
        # self._gz_worker.stop()
        super().destroy_node()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _spray_active_cb(self, msg: Bool):
        if msg.data == self._spray_on:
            return
        self._spray_on = msg.data
        self.get_logger().info(
            f'Spray → {"ON ✓" if self._spray_on else "OFF ✗"}')
        
    def rl_action_callback(self, msg):
        # msg.data layout published by cartesian_trajectory_controller: [standoff (m), flow (0-1)]
        # standoff maps to cone_length; flow scales sigma.
        if len(msg.data) < 2:
            return

        standoff = float(msg.data[0])   # metres
        flow     = float(msg.data[1])   # 0-1

        # Safety limits matching cartesian_trajectory_controller ranges
        new_cone_length = np.clip(standoff, 0.05, 0.50)

        # Derive sigma from flow: flow=0 -> tight (0.010), flow=1 -> wide (0.060)
        new_sigma = np.clip(0.010 + 0.050 * flow, 0.005, 0.10)

        self._cone_len = new_cone_length
        self._sigma    = new_sigma

        # Rebuild cone samples using new parameters
        self._cone_pts = self._build_cone_sample_pts()

        self.get_logger().info(
            f'RL UPDATE -> standoff={standoff:.3f} m  flow={flow:.3f} '
            f'-> cone_length={self._cone_len:.3f}  sigma={self._sigma:.4f}'
        )

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _build_cone_sample_pts(self) -> np.ndarray:
        pts = []
        pts.append([0.0, 0.0, 0.0, 1.0])
        for k in range(1, self._n_rings + 1):
            z     = self._cone_len * k / self._n_rings
            r_max = z * math.tan(self._half_ang)
            for j in range(self._n_ang):
                theta = 2.0 * math.pi * j / self._n_ang
                r = np.random.normal(0.0, self._sigma)
                r = np.clip(r, -r_max, r_max)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                w = math.exp(-r**2 / (2.0 * self._sigma**2))
                pts.append([x, y, z, w])
        return np.array(pts, dtype=np.float64)

    def _transform_to_matrix(self, t) -> np.ndarray:
        tx = t.transform.translation.x
        ty = t.transform.translation.y
        tz = t.transform.translation.z
        qx = t.transform.rotation.x
        qy = t.transform.rotation.y
        qz = t.transform.rotation.z
        qw = t.transform.rotation.w
        n  = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n < 1e-9:
            return np.eye(4)
        qx /= n; qy /= n; qz /= n; qw /= n
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [tx, ty, tz]
        return T

    def _apply_transform(self, T: np.ndarray, pts: np.ndarray) -> np.ndarray:
        ones  = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        return (T @ pts_h.T).T[:, :3]


    # ── RViz markers ──────────────────────────────────────────────────────────
    def _cone_wireframe(self, stamp, T) -> Marker:
        m = Marker()
        m.header.stamp = stamp; m.header.frame_id = self._world_frame
        m.ns = 'spray_cone_wire'; m.id = 0
        m.type = Marker.LINE_LIST; m.action = Marker.ADD
        m.scale.x = 0.002
        m.color   = ColorRGBA(r=0.1, g=0.7, b=1.0, a=0.6)
        m.lifetime = Duration(seconds=0.2).to_msg()
        base_r = self._cone_len * math.tan(self._half_ang)
        tip_l  = np.array([[0., 0., 0.]])
        for i in range(8):
            theta  = 2 * math.pi * i / 8
            edge_l = np.array([[base_r * math.cos(theta),
                                 base_r * math.sin(theta),
                                 self._cone_len]])
            tw = self._apply_transform(T, tip_l)[0]
            ew = self._apply_transform(T, edge_l)[0]
            m.points += [Point(x=tw[0], y=tw[1], z=tw[2]),
                         Point(x=ew[0], y=ew[1], z=ew[2])]
        circle_l = np.array([
            [base_r * math.cos(2 * math.pi * i / 36),
             base_r * math.sin(2 * math.pi * i / 36),
             self._cone_len]
            for i in range(36)])
        cw = self._apply_transform(T, circle_l)
        for i in range(36):
            ni = (i + 1) % 36
            m.points += [Point(x=cw[i][0],  y=cw[i][1],  z=cw[i][2]),
                         Point(x=cw[ni][0], y=cw[ni][1], z=cw[ni][2])]
        return m

    def _nozzle_marker(self, stamp, T) -> Marker:
        m = Marker()
        m.header.stamp = stamp; m.header.frame_id = self._world_frame
        m.ns = 'spray_nozzle'; m.id = 9999
        m.type = Marker.ARROW; m.action = Marker.ADD
        tw = self._apply_transform(T, np.array([[0., 0., 0.]]))[0]
        aw = self._apply_transform(T, np.array([[0., 0., self._cone_len]]))[0]
        m.points = [Point(x=tw[0], y=tw[1], z=tw[2]),
                    Point(x=aw[0], y=aw[1], z=aw[2])]
        m.scale.x = 0.006; m.scale.y = 0.012; m.scale.z = 0.015
        m.color   = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)
        m.lifetime = Duration(seconds=0.3).to_msg()
        return m

    def _build_paint_sphere_list(self, stamp, pts: list,
                                  action: int) -> Marker: 
        m = Marker()
        m.header.stamp    = stamp
        m.header.frame_id = self._world_frame
        m.ns     = 'paint_cloud'
        m.id     = 0
        m.type   = Marker.SPHERE_LIST
        m.action = action   # always Marker.ADD — RViz accumulates
        m.scale.x = 0.008
        m.scale.y = 0.008
        m.scale.z = 0.008
        # No lifetime → persistent until node dies
        for pt in pts:
            m.points.append(Point(x=float(pt[0]),
                                   y=float(pt[1]),
                                   z=float(pt[2])))
            m.colors.append(ColorRGBA(r=0.0, g=0.5, b=1.0, a=1.0))
        return m

    # ── Timer ─────────────────────────────────────────────────────────────────
    def _timer_cb(self):
        self._tick += 1

        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self._world_frame, self._ee_frame, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return

        stamp = self.get_clock().now().to_msg()
        T     = self._transform_to_matrix(tf_stamped)

        # ── EEF speed estimate ────────────────────────────────────────────────
        # Compute how fast the end-effector is moving by diffing the TF
        cur_eef_pos = np.array([
            tf_stamped.transform.translation.x,
            tf_stamped.transform.translation.y,
            tf_stamped.transform.translation.z,
        ])
        if self._prev_eef_pos is not None:
            dt = 1.0 / self.get_parameter('publish_rate_hz').value
            self._eef_speed = float(
                np.linalg.norm(cur_eef_pos - self._prev_eef_pos) / dt)
        self._prev_eef_pos = cur_eef_pos

        # Debug: log EEF speed and spray state periodically to help diagnose
        # why deposition sometimes stops even when the arm appears to move.
        try:
            self.get_logger().info(
                f"spray={self._spray_on} speed={self._eef_speed:.4f}",
                throttle_duration_sec=0.8)
        except Exception:
            self.get_logger().info(f"spray={self._spray_on} speed={self._eef_speed:.4f}")

        # Debug: cone tip world position (helps verify cone length / TF)
        try:
            tip = self._apply_transform(T, np.array([[0.0, 0.0, self._cone_len]]))[0]
            self.get_logger().info(
                f"Cone tip: x={tip[0]:.3f} y={tip[1]:.3f} z={tip[2]:.3f}",
                throttle_duration_sec=1.0)
        except Exception:
            pass

        # ── Accumulate new paint points ───────────────────────────────────────
        new_pts_this_tick: list = []

        # Gate: only deposit paint when spray is ON AND robot is moving.
        if self._spray_on and (not self._enforce_min or self._eef_speed >= self._MIN_EEF_SPEED):
            pts_w   = self._apply_transform(T, self._cone_pts[:, :3])
            weights = self._cone_pts[:, 3]

            for pw, w in zip(pts_w, weights):
                if w < self._min_weight:
                    continue
                # allow one deposit per voxel per tick so thickness can
                # accumulate across repeated passes; pass current tick
                if not self._voxel.is_new(float(pw[0]),
                                           float(pw[1]),
                                           float(pw[2]),
                                           tick=self._tick):
                    continue

                new_pts_this_tick.append(pw.copy())
                self._paint_cloud.append(pw.copy())

            if len(self._paint_cloud) > self._max_pts:
                self._paint_cloud = self._paint_cloud[-self._max_pts:]

        # ── RViz MarkerArray ──────────────────────────────────────────────────
        ma = MarkerArray()

        # Live cone visuals (only while spraying)
        if self._spray_on:
            ma.markers.append(self._cone_wireframe(stamp, T))

        # Nozzle arrow — always visible
        ma.markers.append(self._nozzle_marker(stamp, T))

        # ── Incremental paint cloud publish ───────────────────────────────────

        if new_pts_this_tick:
            self._pending_pts.extend(new_pts_this_tick)

        heartbeat = (self._tick % self._heartbeat_ticks == 0)

        if heartbeat:
            all_pts = self._committed_pts + self._pending_pts
            if all_pts:
                ma.markers.append(
                    self._build_paint_sphere_list(stamp, all_pts, Marker.ADD))
                self._committed_pts = all_pts
                self._pending_pts   = []
        elif self._pending_pts:
            # Incremental: only the new slice since last publish
            ma.markers.append(
                self._build_paint_sphere_list(
                    stamp, self._pending_pts, Marker.ADD))
            self._committed_pts.extend(self._pending_pts)
            self._pending_pts = []

        self._marker_pub.publish(ma)

        # ── PointCloud2 (restored from v11 regression) ───────────────────────
        if self._paint_cloud:
            h = Header()
            h.stamp    = stamp
            h.frame_id = self._world_frame
            self._cloud_pub.publish(
                _make_cloud(h, np.array(self._paint_cloud, dtype=np.float32)))


def main(args=None):
    rclpy.init(args=args)
    node = SpraySimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()