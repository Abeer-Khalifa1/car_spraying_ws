#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import rclpy.qos

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import PointField

import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import threading
import csv as csv_module
import os
import struct

from scipy.spatial import cKDTree


class SurfaceCoverage3D:
    """
    3-D surface coverage tracker driven by trajectory normals.
    CSV column order: 0:x 1:y 2:z 3:qx 4:qy 5:qz 6:qw 7:nx 8:ny 9:nz
    """

    def __init__(self, waypoints: np.ndarray,
                 patch_half_width: float = 0.10,
                 sample_spacing: float = 0.01,
                 voxel_size: float = 0.005):
        if waypoints.shape[1] < 10:
            raise ValueError(f'Expected ≥10 columns, got {waypoints.shape[1]}')

        self.pos     = waypoints[:, 0:3]
        self.normals = waypoints[:, 7:10]

        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self.normals = self.normals / norms

        self._build_frames()
        self.surface_points = self._generate_surface_cloud(
            patch_half_width, sample_spacing, voxel_size
        )
        self.coverage = np.zeros(len(self.surface_points), dtype=np.float32)
        self.kdtree   = cKDTree(self.surface_points)

    def _build_frames(self):
        N  = len(self.pos)
        t1 = np.empty_like(self.pos)
        t1[:-1] = self.pos[1:] - self.pos[:-1]
        t1[-1]  = t1[-2]

        dot = np.einsum('ij,ij->i', t1, self.normals)
        t1  = t1 - dot[:, None] * self.normals

        bad = np.linalg.norm(t1, axis=1) < 1e-9
        if bad.any():
            fallback = np.tile([1.0, 0.0, 0.0], (N, 1))
            dot_fb   = np.einsum('ij,ij->i', fallback, self.normals)
            fallback = fallback - dot_fb[:, None] * self.normals
            t1[bad]  = fallback[bad]

        t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)
        t1_norm = np.where(t1_norm < 1e-9, 1.0, t1_norm)
        self.t1 = t1 / t1_norm

        self.t2    = np.cross(self.normals, self.t1)
        t2_norm    = np.linalg.norm(self.t2, axis=1, keepdims=True)
        t2_norm    = np.where(t2_norm < 1e-9, 1.0, t2_norm)
        self.t2    = self.t2 / t2_norm

    def _generate_surface_cloud(self, patch_half_width, sample_spacing,
                                voxel_size) -> np.ndarray:
        du = np.arange(-patch_half_width, patch_half_width + sample_spacing,
                       sample_spacing)
        dv = np.arange(-patch_half_width, patch_half_width + sample_spacing,
                       sample_spacing)
        DU, DV = np.meshgrid(du, dv, indexing='ij')
        DU = DU.ravel()[:, None]
        DV = DV.ravel()[:, None]

        patches = (
            self.pos[:, None, :]
            + DU[None, :, :] * self.t1[:, None, :]
            + DV[None, :, :] * self.t2[:, None, :]
        )
        surface_points = patches.reshape(-1, 3).astype(np.float32)

        voxels = np.floor(surface_points / voxel_size).astype(np.int64)
        _, unique_idx = np.unique(voxels, axis=0, return_index=True)
        return surface_points[unique_idx]

    def deposit_paint_batch(self, points: np.ndarray,
                            amount: float = 1.0,
                            max_dist: float = 0.03):
        """Batch deposit: points is (M, 3) float32."""
        if len(points) == 0:
            return
        distances, indices = self.kdtree.query(points.astype(np.float32),
                                               workers=-1)
        mask = distances <= max_dist
        np.add.at(self.coverage, indices[mask], amount)


def _rotation_matrix_to_quaternion(R: np.ndarray):
    m00, m01, m02 = R[0]; m10, m11, m12 = R[1]; m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s; x = (m21-m12)/s; y = (m02-m20)/s; z = (m10-m01)/s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0+m00-m11-m22) * 2.0
        w = (m21-m12)/s; x = 0.25*s; y = (m01+m10)/s; z = (m02+m20)/s
    elif m11 > m22:
        s = np.sqrt(1.0+m11-m00-m22) * 2.0
        w = (m02-m20)/s; x = (m01+m10)/s; y = 0.25*s; z = (m12+m21)/s
    else:
        s = np.sqrt(1.0+m22-m00-m11) * 2.0
        w = (m10-m01)/s; x = (m02+m20)/s; y = (m12+m21)/s; z = 0.25*s
    quat = np.array([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    return quat/norm if norm >= 1e-9 else np.array([0.,0.,0.,1.])


class CoverageMapGenerator(Node):

    def __init__(self):
        super().__init__('coverage_map_generator')

        self.declare_parameter('trajectory_csv',   '')
        self.declare_parameter('patch_half_width', 0.10)
        self.declare_parameter('sample_spacing',   0.01)
        self.declare_parameter('voxel_size',       0.005)
        self.declare_parameter('max_deposit_dist', 0.08)

        trajectory_csv        = self.get_parameter('trajectory_csv').value
        patch_half_width      = self.get_parameter('patch_half_width').value
        sample_spacing        = self.get_parameter('sample_spacing').value
        voxel_size            = self.get_parameter('voxel_size').value
        self.max_deposit_dist = self.get_parameter('max_deposit_dist').value

        self.surface = self._load_surface(
            trajectory_csv, patch_half_width, sample_spacing, voxel_size
        )

        self.get_logger().info(
            f'SurfaceCoverage3D: {len(self.surface.pos)} waypoints → '
            f'{len(self.surface.surface_points)} surface samples | '
            f'max_deposit_dist={self.max_deposit_dist:.3f} m'
        )

        self.lock = threading.Lock()

        # Track last paint cloud size — spray_sim publishes full accumulated
        # cloud every tick; we only deposit the NEW tail each callback.
        self._last_cloud_size = 0

        qos_reliable = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, '/spray/coverage_cloud', qos_reliable
        )
        self._coverage_pub = self.create_publisher(
            Float32MultiArray, '/spray/coverage_vector', 10
        )

        # Match spray_sim_node publisher QoS (depth=10, BEST_EFFORT/VOLATILE)
        self.create_subscription(
            PointCloud2, '/spray/paint_points',
            self._cloud_callback,
            rclpy.qos.QoSProfile(
                depth=10,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE
            )
        )

        self.create_timer(0.2, self._publish_coverage)
        self.get_logger().info('CoverageMapGenerator (3D) started.')
        self._log_tick = 0

    def _load_surface(self, csv_path, patch_half_width, sample_spacing,
                      voxel_size) -> SurfaceCoverage3D:
        rows = []
        if csv_path and os.path.isfile(csv_path):
            try:
                with open(csv_path, newline='') as f:
                    for row in csv_module.reader(f):
                        if len(row) >= 10:
                            try:
                                rows.append([float(v) for v in row[:10]])
                            except ValueError:
                                pass
                self.get_logger().info(f'Loaded {len(rows)} waypoints from {csv_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not read CSV ({e}); using fallback')

        if len(rows) < 3:
            self.get_logger().warn(
                'Not enough trajectory waypoints — using synthetic curved surface. '
                'Pass trajectory_csv:=<path> to fix this.'
            )
            synth = []
            for theta in np.linspace(-np.pi/4, np.pi/4, 20):
                for phi in np.linspace(-np.pi/6, np.pi/6, 10):
                    x  =  0.3 * np.cos(theta) * np.cos(phi)
                    y  =  0.3 * np.sin(phi)
                    z  =  0.3 * np.sin(theta) * np.cos(phi) + 0.5
                    nx = -np.cos(theta) * np.cos(phi)
                    ny = -np.sin(phi)
                    nz = -np.sin(theta) * np.cos(phi)
                    synth.append([x, y, z, 0., 0., 0., 1., nx, ny, nz])
            rows = synth

        return SurfaceCoverage3D(
            np.asarray(rows, dtype=float),
            patch_half_width=patch_half_width,
            sample_spacing=sample_spacing,
            voxel_size=voxel_size
        )

    def _cloud_callback(self, msg: PointCloud2):
        total_pts = msg.width

        # Reset if spray_sim was restarted (cloud shrank)
        if total_pts < self._last_cloud_size:
            self._last_cloud_size = 0

        if total_pts == self._last_cloud_size:
            return

        # Slice only new points from the raw bytes — avoid re-processing history
        new_count  = total_pts - self._last_cloud_size
        point_step = msg.point_step   # 12 bytes (XYZ float32)
        byte_start = self._last_cloud_size * point_step
        raw_slice  = bytes(msg.data)[byte_start: byte_start + new_count * point_step]
        pts_np     = np.frombuffer(raw_slice, dtype=np.float32).reshape(-1, 3)

        self._last_cloud_size = total_pts

        self._log_tick += 1
        if self._log_tick % 10 == 0:
            self.get_logger().info(
                f'Incoming paint pts: {len(pts_np)} new | '
                f'total cloud={total_pts} | '
                f'zmin={pts_np[:,2].min():.3f} zmax={pts_np[:,2].max():.3f}'
            )

        with self.lock:
            self.surface.deposit_paint_batch(
                pts_np,
                amount=1.0,
                max_dist=self.max_deposit_dist
            )

    def _publish_coverage(self):
        with self.lock:
            pts      = self.surface.surface_points.copy()
            coverage = self.surface.coverage.copy()

        stamp    = self.get_clock().now().to_msg()
        frame_id = 'world'

        fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 16
        data_np    = np.column_stack([pts, coverage]).astype(np.float32)

        cloud_msg                 = PointCloud2()
        cloud_msg.header.stamp    = stamp
        cloud_msg.header.frame_id = frame_id
        cloud_msg.height          = 1
        cloud_msg.width           = len(pts)
        cloud_msg.fields          = fields
        cloud_msg.is_bigendian    = False
        cloud_msg.point_step      = point_step
        cloud_msg.row_step        = point_step * len(pts)
        cloud_msg.data            = data_np.tobytes()
        cloud_msg.is_dense        = True
        self._cloud_pub.publish(cloud_msg)

        cov_msg = Float32MultiArray()
        cov_msg.layout.dim.append(
            MultiArrayDimension(
                label='surface_samples',
                size=len(coverage),
                stride=len(coverage)
            )
        )
        cov_msg.data = coverage.tolist()
        self._coverage_pub.publish(cov_msg)

    def save_coverage_to_csv(self):
        filename = (
            '/home/user/car_spraying_ws/src/square_trajectory/coverage_3d.csv'
        )
        with self.lock:
            pts      = self.surface.surface_points.copy()
            coverage = self.surface.coverage.copy()
        data = np.column_stack([pts, coverage])
        np.savetxt(filename, data, delimiter=',',
                   header='x,y,z,coverage', fmt='%.5f', comments='')
        self.get_logger().info(f'3D coverage saved to: {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = CoverageMapGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_coverage_to_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()