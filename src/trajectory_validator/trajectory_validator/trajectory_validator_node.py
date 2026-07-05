#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger

from .robot_workspace import (
    WORKSPACE_AABB, MAX_REACH, MIN_REACH,
    check_point, clamp_point,
)
from .csv_loader import load_csv


# ── colour helpers ────────────────────────────────────────────────────────────

def _rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


_GREEN  = _rgba(0.10, 0.90, 0.10, 0.90)
_RED    = _rgba(0.90, 0.10, 0.10, 0.95)
_ORANGE = _rgba(1.00, 0.55, 0.00, 0.12)
_LABEL  = _rgba(1.00, 0.20, 0.20, 1.00)
_LINE   = _rgba(0.20, 0.85, 0.20, 0.80)


# ── marker builders ───────────────────────────────────────────────────────────

def _base_marker(node: Node, ns: str, uid: int,
                 mtype: int, frame_id: str) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp    = node.get_clock().now().to_msg()
    m.ns              = ns
    m.id              = uid
    m.type            = mtype
    m.action          = Marker.ADD
    m.pose.orientation.w = 1.0
    return m


def build_marker_array(node: Node, records: list[dict],
                       frame_id: str) -> MarkerArray:
    ma = MarkerArray()
    now = node.get_clock().now().to_msg()

    # ── 1. Workspace AABB box ─────────────────────────────────────────────
    box = _base_marker(node, 'workspace', 0, Marker.CUBE, frame_id)
    xl, xh = WORKSPACE_AABB['x']
    yl, yh = WORKSPACE_AABB['y']
    zl, zh = WORKSPACE_AABB['z']
    box.pose.position.x = (xl + xh) / 2
    box.pose.position.y = (yl + yh) / 2
    box.pose.position.z = (zl + zh) / 2
    box.scale.x = xh - xl
    box.scale.y = yh - yl
    box.scale.z = zh - zl
    box.color   = _ORANGE
    ma.markers.append(box)

    # ── 2. Max-reach sphere ───────────────────────────────────────────────
    sph = _base_marker(node, 'workspace', 1, Marker.SPHERE, frame_id)
    d = MAX_REACH * 2
    sph.scale.x = sph.scale.y = sph.scale.z = d
    sph.color = _rgba(1.0, 0.5, 0.0, 0.04)
    ma.markers.append(sph)

    # ── 3. Waypoint spheres + text labels -────────────────────────────────
    line = _base_marker(node, 'trajectory_path', 10, Marker.LINE_STRIP, frame_id)
    line.scale.x = 0.005
    line.color   = _LINE

    for idx, rec in enumerate(records):
        ok, viols = check_point(rec['x'], rec['y'], rec['z'])

        # Sphere
        sp = _base_marker(node, 'waypoints', idx + 200, Marker.SPHERE, frame_id)
        sp.pose.position.x = rec['x']
        sp.pose.position.y = rec['y']
        sp.pose.position.z = rec['z']
        sp.scale.x = sp.scale.y = sp.scale.z = 0.015
        sp.color = _GREEN if ok else _RED
        ma.markers.append(sp)

        # Row-number text
        txt = _base_marker(node, 'labels', idx + 10_000,
                           Marker.TEXT_VIEW_FACING, frame_id)
        txt.pose.position.x = rec['x']
        txt.pose.position.y = rec['y']
        txt.pose.position.z = rec['z'] + 0.025
        txt.scale.z = 0.012
        txt.color   = _LABEL if not ok else _rgba(0.4, 0.4, 0.4, 0.6)
        txt.text    = f"#{rec['row_index']}"
        if not ok:
            txt.text += '\n' + viols[0]
        ma.markers.append(txt)

        # Path line — safe points only
        if ok:
            p = Point()
            p.x, p.y, p.z = rec['x'], rec['y'], rec['z']
            line.points.append(p)

    if line.points:
        ma.markers.append(line)

    return ma


# ── node ──────────────────────────────────────────────────────────────────────

class TrajectoryValidatorNode(Node):

    def __init__(self) -> None:
        super().__init__('trajectory_validator')

        # ── declare parameters ────────────────────────────────────────────
        self.declare_parameter(
            'csv_path', '',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Absolute path to trajectory CSV file'))
        self.declare_parameter(
            'frame_id', 'base_link',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='TF frame for markers'))
        self.declare_parameter(
            'rate_hz', 1.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Marker republish rate in Hz'))
        self.declare_parameter(
            'clamp', False,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Log clamped position alongside each violation'))

        # ── publishers ────────────────────────────────────────────────────
        self._pub_markers = self.create_publisher(
            MarkerArray, '~/trajectory_markers', 10)
        self._pub_sphere  = self.create_publisher(
            Marker, '~/workspace_sphere', 10)

        # ── service ───────────────────────────────────────────────────────
        self._srv_reload = self.create_service(
            Trigger, '~/reload', self._reload_cb)

        # ── parameter-change callback (ros2 param set …) ──────────────────
        self.add_on_set_parameters_callback(self._on_param_change)

        self._marker_array: MarkerArray | None = None

        # ── initial load ──────────────────────────────────────────────────
        csv_path = self.get_parameter('csv_path').value
        if csv_path:
            self._load(csv_path)
        else:
            self.get_logger().info(
                'trajectory_validator ready — set csv_path parameter to load a CSV.\n'
                '  ros2 param set /trajectory_validator csv_path /path/to/traj.csv\n'
                '  ros2 service call /trajectory_validator/reload std_srvs/srv/Trigger {}')

        # ── republish timer ───────────────────────────────────────────────
        rate = self.get_parameter('rate_hz').value
        self._timer = self.create_timer(1.0 / rate, self._timer_cb)

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self, path: str) -> bool:
        if not os.path.isfile(path):
            self.get_logger().error(f'File not found: {path}')
            return False
        try:
            records = load_csv(path)
        except Exception as exc:
            self.get_logger().error(f'Error loading CSV: {exc}')
            return False

        frame = self.get_parameter('frame_id').value
        do_clamp = self.get_parameter('clamp').value

        n_safe   = 0
        n_unsafe = 0
        for rec in records:
            ok, viols = check_point(rec['x'], rec['y'], rec['z'])
            if ok:
                n_safe += 1
            else:
                n_unsafe += 1
                clamp_str = ''
                if do_clamp:
                    cx, cy, cz = clamp_point(rec['x'], rec['y'], rec['z'])
                    clamp_str = f'  → clamped ({cx:+.4f}, {cy:+.4f}, {cz:+.4f})'
                self.get_logger().warn(
                    f'Row {rec["row_index"]:4d} '
                    f'({rec["x"]:+.4f}, {rec["y"]:+.4f}, {rec["z"]:+.4f})'
                    f'\n    ' + '\n    '.join(viols) + clamp_str)

        self.get_logger().info(
            f'Loaded {len(records)} waypoints from {path}  '
            f'[{n_safe} safe / {n_unsafe} UNSAFE]')

        self._marker_array = build_marker_array(self, records, frame)
        return True

    def _timer_cb(self) -> None:
        if self._marker_array is not None:
            self._pub_markers.publish(self._marker_array)

    def _reload_cb(self, _req: Trigger.Request,
                   res: Trigger.Response) -> Trigger.Response:
        path = self.get_parameter('csv_path').value
        ok   = self._load(path)
        res.success = ok
        res.message = 'Reloaded successfully' if ok else 'Reload failed — check logs'
        return res

    def _on_param_change(self, params: list[Parameter]):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == 'csv_path' and p.value:
                self._load(p.value)
        return SetParametersResult(successful=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryValidatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
