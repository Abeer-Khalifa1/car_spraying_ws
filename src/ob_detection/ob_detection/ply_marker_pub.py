#!/usr/bin/env python3
"""
ply_marker_pub.py
==================
Reads part_mesh.ply and part_mesh_coverage.ply from disk and re-publishes
them as visualization_msgs/Marker (TRIANGLE_LIST) at a configurable rate.

Topics published:
  /part_mesh_marker           — raw mesh, solid blue
  /part_mesh_coverage_marker  — coverage-coloured mesh (colour from PLY face RGB)
  /part_mesh_cloud            — raw mesh vertices as PointCloud2
  /part_mesh_coverage_cloud   — coverage mesh vertices as PointCloud2 with RGB

Parameters:
  mesh_ply         (str)   path to part_mesh.ply
  coverage_ply     (str)   path to part_mesh_coverage.ply
  mesh_frame       (str)   TF frame id  (default: camera_color_optical_frame)
  publish_rate_hz  (float) republish rate (default: 1.0)

FIX — what changed vs the original:
  1. File-existence is checked before trying to parse, giving a clear
     actionable error message (with a find-command hint) instead of the
     bare "No such file or directory" that appeared in the original log.
  2. A WARNING is emitted after __init__ when neither PLY was loaded so the
     user can see at a glance that nothing will be published.
  3. Minor: validate publish_rate_hz > 0 with a sensible fallback.
"""

import os
import struct

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal ASCII/binary PLY parser  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ply(path):
    """
    Returns (vertices, faces, face_colors).
      vertices   : list of (x, y, z) float tuples
      faces      : list of [i0, i1, i2] int lists
      face_colors: list of (r, g, b) uint8 tuples, or None if no colour in file
    """
    with open(path, 'rb') as f:
        raw = f.read()

    # ── header ──
    header_end = raw.find(b'end_header')
    if header_end == -1:
        raise ValueError(f"No end_header in {path}")
    header_bytes = raw[:header_end]
    header = header_bytes.decode('ascii', errors='replace')
    body   = raw[header_end + len('end_header'):]
    # skip the newline after end_header
    if body[:1] in (b'\r', b'\n'):
        body = body[1:]
    if body[:1] == b'\n':
        body = body[1:]

    lines_h = [l.strip() for l in header.splitlines()]

    is_binary_little = any('binary_little_endian' in l for l in lines_h)
    is_binary_big    = any('binary_big_endian'    in l for l in lines_h)
    is_ascii         = not (is_binary_little or is_binary_big)

    # count vertices / faces and detect face colour properties
    n_verts = n_faces = 0
    has_face_color = False
    current_element = None
    for l in lines_h:
        if l.startswith('element vertex'):
            n_verts = int(l.split()[-1])
            current_element = 'vertex'
        elif l.startswith('element face'):
            n_faces = int(l.split()[-1])
            current_element = 'face'
        elif l.startswith('property') and current_element == 'face':
            if 'red' in l or 'green' in l or 'blue' in l:
                has_face_color = True

    vertices    = []
    faces       = []
    face_colors = [] if has_face_color else None

    if is_ascii:
        data_lines = body.decode('ascii', errors='replace').splitlines()
        idx = 0
        for _ in range(n_verts):
            parts = data_lines[idx].split(); idx += 1
            vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
        for _ in range(n_faces):
            parts = data_lines[idx].split(); idx += 1
            n = int(parts[0])
            face = [int(parts[1 + k]) for k in range(n)]
            faces.append(face[:3])
            if has_face_color and len(parts) >= 1 + n + 3:
                r = int(parts[1 + n])
                g = int(parts[2 + n])
                b = int(parts[3 + n])
                face_colors.append((r, g, b))
    else:
        endian = '<' if is_binary_little else '>'
        offset = 0
        for _ in range(n_verts):
            x, y, z = struct.unpack_from(endian + 'fff', body, offset)
            vertices.append((x, y, z))
            offset += 12
        for _ in range(n_faces):
            n_idx = struct.unpack_from('B', body, offset)[0]; offset += 1
            idxs  = list(struct.unpack_from(endian + 'i' * n_idx, body, offset))
            offset += 4 * n_idx
            faces.append(idxs[:3])
            if has_face_color:
                r, g, b = struct.unpack_from('BBB', body, offset); offset += 3
                face_colors.append((r, g, b))

    return vertices, faces, face_colors


# ─────────────────────────────────────────────────────────────────────────────
#  ROS 2 node
# ─────────────────────────────────────────────────────────────────────────────

class PlyMarkerPublisher(Node):

    def __init__(self):
        super().__init__('ply_marker_publisher')

        self.declare_parameter('mesh_ply',        '')
        self.declare_parameter('coverage_ply',    '')
        self.declare_parameter('mesh_frame',      'camera_color_optical_frame')
        self.declare_parameter('publish_rate_hz', 1.0)

        mesh_path     = self.get_parameter('mesh_ply').value
        coverage_path = self.get_parameter('coverage_ply').value
        self.frame_id = self.get_parameter('mesh_frame').value
        rate_hz       = float(self.get_parameter('publish_rate_hz').value)

        # FIX: guard against non-positive rate
        if rate_hz <= 0.0:
            self.get_logger().warn(
                f'publish_rate_hz={rate_hz} is invalid; defaulting to 1.0 Hz')
            rate_hz = 1.0

        self.mesh_pub          = self.create_publisher(Marker, '/part_mesh_marker',          1)
        self.coverage_pub      = self.create_publisher(Marker, '/part_mesh_coverage_marker', 1)
        self.mesh_cloud_pub    = self.create_publisher(PointCloud2, '/part_mesh_cloud', 1)
        self.coverage_cloud_pub= self.create_publisher(PointCloud2, '/part_mesh_coverage_cloud', 1)

        self.mesh_marker     = None
        self.coverage_marker = None
        self.mesh_cloud      = None
        self.coverage_cloud  = None

        if mesh_path:
            self.mesh_marker, self.mesh_cloud = self._load_plain(
                mesh_path, ns='part_mesh', marker_id=0)
        if coverage_path:
            self.coverage_marker, self.coverage_cloud = self._load_coverage(
                coverage_path, ns='part_mesh_coverage', marker_id=1)

        # FIX: warn clearly when neither file was loaded so the user isn't
        #      left wondering why nothing appears in RViz.
        if self.mesh_marker is None and self.coverage_marker is None:
            self.get_logger().warn(
                '\n'
                '  ┌─────────────────────────────────────────────────────────┐\n'
                '  │  NO PLY FILES LOADED — no markers will be published.    │\n'
                '  │                                                         │\n'
                '  │  Either the files do not exist yet (run the detection   │\n'
                '  │  pipeline first) or the paths are wrong.                │\n'
                '  │                                                         │\n'
                '  │  Locate your files with:                                │\n'
                '  │    find ~ -name "part_mesh*.ply" 2>/dev/null            │\n'
                '  │                                                         │\n'
                '  │  Then re-launch with explicit paths:                    │\n'
                '  │    mesh_ply:=/abs/path/part_mesh.ply                    │\n'
                '  │    coverage_ply:=/abs/path/part_mesh_coverage.ply       │\n'
                '  └─────────────────────────────────────────────────────────┘'
            )

        period = 1.0 / rate_hz
        self.create_timer(period, self._publish)
        self.get_logger().info(
            f'PLY marker + PointCloud2 publisher ready  frame={self.frame_id}  '
            f'rate={rate_hz:.1f} Hz\n'
            f'  mesh:     {mesh_path or "(not set)"}\n'
            f'  coverage: {coverage_path or "(not set)"}\n'
            f'  pointclouds: /part_mesh_cloud, /part_mesh_coverage_cloud')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _check_path(self, path: str) -> bool:
        """
        FIX: validate the file exists before attempting to parse it.
        Logs an actionable error with a search hint when it does not.
        """
        if not os.path.isfile(path):
            self.get_logger().error(
                f'PLY file not found: {path}\n'
                f'  Search for it with:  find ~ -name "{os.path.basename(path)}" 2>/dev/null\n'
                f'  Then pass the correct path via the launch argument.'
            )
            return False
        return True

    # ── builders ──────────────────────────────────────────────────────────────

    def _base_marker(self, ns, marker_id):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.ns    = ns
        m.id    = marker_id
        m.type  = Marker.TRIANGLE_LIST
        m.action= Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.pose.orientation.w = 1.0
        return m

    @staticmethod
    def _rgb_to_float(r, g, b):
        rgb = (int(r) << 16) | (int(g) << 8) | int(b)
        return struct.unpack('f', struct.pack('I', rgb))[0]

    def _make_cloud(self, vertices, rgb_values=None):
        header = Header()
        header.frame_id = self.frame_id

        if rgb_values is None:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            points = [(float(x), float(y), float(z)) for x, y, z in vertices]
        else:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            points = []
            for (x, y, z), (r, g, b) in zip(vertices, rgb_values):
                points.append((
                    float(x),
                    float(y),
                    float(z),
                    self._rgb_to_float(r, g, b),
                ))

        return point_cloud2.create_cloud(header, fields, points)

    def _load_plain(self, path, ns, marker_id):
        """Solid semi-transparent blue marker."""
        # FIX: existence check first
        if not self._check_path(path):
            return None, None

        try:
            vertices, faces, _ = _parse_ply(path)
        except Exception as e:
            self.get_logger().error(f'Failed to parse {path}: {e}')
            return None, None

        m = self._base_marker(ns, marker_id)
        c = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.7)
        for face in faces:
            for vi in face:
                p = Point()
                p.x, p.y, p.z = (float(vertices[vi][0]),
                                  float(vertices[vi][1]),
                                  float(vertices[vi][2]))
                m.points.append(p)
                m.colors.append(c)

        cloud = self._make_cloud(vertices)

        self.get_logger().info(
            f'[mesh] loaded {path}  ({len(vertices)} verts, {len(faces)} faces)')
        return m, cloud

    def _load_coverage(self, path, ns, marker_id):
        """Per-face colour marker from PLY face RGB (BGR stored by detection node)."""
        # FIX: existence check first
        if not self._check_path(path):
            return None, None

        try:
            vertices, faces, face_colors = _parse_ply(path)
        except Exception as e:
            self.get_logger().error(f'Failed to parse {path}: {e}')
            return None, None

        m = self._base_marker(ns, marker_id)

        # fallback colour if PLY has no face colours
        default_c = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.85)

        vertex_color_map = None
        if face_colors:
            cloud_points = []
            for i, face in enumerate(faces):
                bgr = face_colors[i] if i < len(face_colors) else (128, 128, 128)
                rgb = (bgr[2], bgr[1], bgr[0])
                for vi in face:
                    vertex = vertices[vi]
                    cloud_points.append((vertex[0], vertex[1], vertex[2], rgb[0], rgb[1], rgb[2]))
            cloud = self._make_cloud(
                [(x, y, z) for x, y, z, *_ in cloud_points],
                [(r, g, b) for *_, r, g, b in cloud_points])
        else:
            cloud = self._make_cloud(vertices)

        for i, face in enumerate(faces):
            if face_colors and i < len(face_colors):
                bgr = face_colors[i]          # detection node writes BGR
                c = ColorRGBA(
                    r=float(bgr[2]) / 255.0,  # R ← B channel
                    g=float(bgr[1]) / 255.0,
                    b=float(bgr[0]) / 255.0,  # B ← R channel
                    a=0.85)
            else:
                c = default_c
            for vi in face:
                p = Point()
                p.x, p.y, p.z = (float(vertices[vi][0]),
                                  float(vertices[vi][1]),
                                  float(vertices[vi][2]))
                m.points.append(p)
                m.colors.append(c)

        self.get_logger().info(
            f'[coverage] loaded {path}  ({len(vertices)} verts, {len(faces)} faces, '
            f'colours={"yes" if face_colors else "no"})')
        return m, cloud

    # ── timer callback ────────────────────────────────────────────────────────

    def _publish(self):
        now = self.get_clock().now().to_msg()
        if self.mesh_marker:
            self.mesh_marker.header.stamp = now
            self.mesh_pub.publish(self.mesh_marker)
        if self.coverage_marker:
            self.coverage_marker.header.stamp = now
            self.coverage_pub.publish(self.coverage_marker)
        if self.mesh_cloud:
            self.mesh_cloud.header.stamp = now
            self.mesh_cloud_pub.publish(self.mesh_cloud)
        if self.coverage_cloud:
            self.coverage_cloud.header.stamp = now
            self.coverage_cloud_pub.publish(self.coverage_cloud)


def main(args=None):
    rclpy.init(args=args)
    node = PlyMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()