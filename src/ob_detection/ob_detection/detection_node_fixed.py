#!/usr/bin/env python3
"""
car_parts_segmentation_spray.py - SQUARE SPIRAL SPRAY PATH
Uses polygon inward offsetting (contour shrinking) to generate
proper closed rectangular rings that step inward — no collapsing lines.
"""

import os
import threading
import csv
from pathlib import Path

if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from sensor_msgs_py import point_cloud2
except ImportError:
    point_cloud2 = None


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def get_mask_center(mask_binary):
    m = cv2.moments(mask_binary)
    if m["m00"] == 0:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))


def get_mask_contour(mask_binary):
    contours, _ = cv2.findContours(mask_binary.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).squeeze()


def shrink_contour(contour_pts, amount):
    """
    Inward-offset a polygon by `amount` pixels using the erosion trick:
    draw the polygon filled on a canvas, erode by `amount`, re-extract contour.
    Returns the shrunken contour as (N,2) int32, or None if it vanished.
    """
    # Bounding box for a tight canvas
    x, y, w, h = cv2.boundingRect(contour_pts.reshape(-1, 1, 2))
    pad = amount + 4
    canvas = np.zeros((h + pad * 2, w + pad * 2), dtype=np.uint8)

    # Shift contour onto canvas
    shifted = contour_pts.copy()
    shifted[:, 0] -= x - pad
    shifted[:, 1] -= y - pad
    cv2.fillPoly(canvas, [shifted.reshape(-1, 1, 2)], 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (amount * 2 + 1, amount * 2 + 1))
    eroded = cv2.erode(canvas, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea).squeeze()
    if best.ndim == 1:
        best = best.reshape(1, 2)
    if len(best) < 4:
        return None

    # Shift back to original image coordinates
    best[:, 0] += x - pad
    best[:, 1] += y - pad
    return best.astype(np.int32)


def resample_contour(pts, spacing):
    """Emit one point every `spacing` pixels of arc length along pts (closed loop)."""
    if len(pts) < 2:
        return pts
    out   = [pts[0].copy()]
    accum = 0.0
    for i in range(1, len(pts)):
        seg    = float(np.linalg.norm(pts[i].astype(float) - pts[i-1].astype(float)))
        accum += seg
        while accum >= spacing:
            accum -= spacing
            out.append(pts[i].copy())
    return np.array(out, dtype=np.int32)


def rotate_to_nearest(ring_pts, target):
    """Roll ring_pts so the point closest to target comes first."""
    dists = np.linalg.norm(ring_pts.astype(float) - np.array(target, float), axis=1)
    return np.roll(ring_pts, -int(np.argmin(dists)), axis=0)


def straight_connector(p0, p1, step=3):
    """Return interpolated points from p0 to p1, spaced ~step pixels."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    dist = np.linalg.norm(p1 - p0)
    n    = max(2, int(dist / step))
    return np.array(
        [np.round(p0 + (p1 - p0) * t / n).astype(int) for t in range(1, n + 1)],
        dtype=np.int32)


# ═══════════════════════════════════════════════════════════════
#  SQUARE SPIRAL  —  inward polygon offsetting
# ═══════════════════════════════════════════════════════════════

def generate_square_spiral(mask_binary, spacing_px=20):
    """
    Build a square/rectangular inward spiral by repeatedly shrinking
    the outermost contour of the mask.

    Each ring is a CLOSED loop traced along the shrunken polygon.
    Rings are connected with a short straight segment so the full
    path is one continuous line — exactly like wrapping tape inward.
    """
    # Get outer contour of the full mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([])

    current_contour = max(contours, key=cv2.contourArea).squeeze()
    if current_contour.ndim == 1:
        current_contour = current_contour.reshape(1, 2)

    all_points = []
    prev_end   = None
    offset     = 0   # total erosion applied so far

    while True:
        if len(current_contour) < 4:
            break
        if cv2.contourArea(current_contour.reshape(-1, 1, 2)) < spacing_px ** 2:
            break

        # Resample this ring to uniform waypoints
        ring_pts = resample_contour(current_contour,
                                    spacing=max(2, spacing_px))

        # Close the loop: ensure last pt connects back to first
        # (contour from cv2 is already closed, but resample may drop the wrap)
        ring_pts = np.vstack([ring_pts, ring_pts[0]])

        # Rotate start to nearest previous endpoint
        if prev_end is not None:
            ring_pts = rotate_to_nearest(ring_pts, prev_end)

        # Connector from previous end → this ring's start
        if prev_end is not None:
            conn = straight_connector(prev_end, ring_pts[0], step=3)
            all_points.extend(conn)

        all_points.extend(ring_pts)
        prev_end = ring_pts[-1]

        # Shrink contour inward by spacing_px for the next ring
        offset += spacing_px
        next_contour = shrink_contour(
            max(cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)[0],
                key=cv2.contourArea).squeeze(),
            amount=offset)

        if next_contour is None:
            break
        current_contour = next_contour

    # Final center waypoint
    center = get_mask_center(mask_binary)
    if center:
        if prev_end is not None:
            conn = straight_connector(prev_end, center, step=3)
            all_points.extend(conn)
        all_points.append(np.array(center, dtype=np.int32))

    return np.array(all_points, dtype=np.int32) if all_points else np.array([])


def resample_path_to_count(points, target_points):
    """Resample a path to a fixed number of points while preserving endpoints."""
    if len(points) == 0 or target_points <= 0:
        return np.array([], dtype=np.int32)
    if len(points) == 1 or target_points == 1:
        return points[:1]

    src_idx = np.linspace(0, len(points) - 1, num=len(points), dtype=np.float32)
    dst_idx = np.linspace(0, len(points) - 1, num=target_points, dtype=np.float32)
    x = np.interp(dst_idx, src_idx, points[:, 0])
    y = np.interp(dst_idx, src_idx, points[:, 1])
    return np.round(np.column_stack((x, y))).astype(np.int32)


def adaptive_path_target(points_count, min_points, max_points):
    """Keep natural path density, clamped to the requested waypoint range."""
    min_points = max(4, int(min_points))
    max_points = max(min_points, int(max_points))
    return int(np.clip(points_count, min_points, max_points))


def compute_path_orientation_and_normal(path_metric, index):
    """
    Compute orientation (as quaternion) and surface normal for a spray nozzle on the path.
    
    Returns a tuple of (qx, qy, qz, qw, nx, ny, nz) where:
    - qx, qy, qz, qw: quaternion representing spray nozzle orientation (yaw along path + pitch downward)
    - nx, ny, nz: perpendicular direction to the path (90° rotated in XY plane, pointing outward)
    """
    # Default: nozzle pointing straight down (pitch = -90°) with no yaw
    qx, qy, qz, qw = 0.0, 0.7071068, 0.0, 0.7071068  # -90° pitch around Y axis
    nx, ny, nz = 0.0, 0.0, 1.0  # Default normal pointing up
    
    if len(path_metric) < 2:
        return qx, qy, qz, qw, nx, ny, nz
    
    # Compute tangent direction along the path
    tangent = np.array([0.0, 0.0, 0.0])
    
    if index > 0 and index < len(path_metric) - 1:
        p_prev = np.array(path_metric[index - 1][:3])
        p_next = np.array(path_metric[index + 1][:3])
        tangent = p_next - p_prev
    elif index == 0 and len(path_metric) > 1:
        p_curr = np.array(path_metric[0][:3])
        p_next = np.array(path_metric[1][:3])
        tangent = p_next - p_curr
    elif index == len(path_metric) - 1 and len(path_metric) > 1:
        p_prev = np.array(path_metric[-2][:3])
        p_curr = np.array(path_metric[-1][:3])
        tangent = p_curr - p_prev
    
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-6:
        tangent = tangent / tangent_norm
        
        # Compute yaw from XY components (heading along path)
        yaw = np.arctan2(tangent[1], tangent[0])
        
        # Compute pitch from elevation change (how much the path goes up/down)
        xy_dist = np.sqrt(tangent[0]**2 + tangent[1]**2)
        if xy_dist > 1e-6:
            path_pitch = np.arctan2(tangent[2], xy_dist)
        else:
            path_pitch = 0.0
        
        # Nozzle orientation: 
        # - Pitch of -90° (pointing down toward surface) + path_pitch (elevation changes)
        # - Yaw along the path heading
        nozzle_pitch = -np.pi / 2 + path_pitch  # -90° + elevation change
        
        # Create quaternion from yaw and nozzle_pitch (roll = 0)
        # Quaternion: rotate by yaw around Z, then by nozzle_pitch around Y
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        cp = np.cos(nozzle_pitch / 2)
        sp = np.sin(nozzle_pitch / 2)
        cr = 1.0  # cos(roll/2) where roll = 0
        sr = 0.0  # sin(roll/2) where roll = 0
        
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        
        # Compute perpendicular direction (90° rotated from tangent in XY plane)
        # This points to the side of the spray path
        nx = -tangent[1]  # Rotate 90° CCW in XY plane
        ny = tangent[0]
        nz = 0.0  # Perpendicular is in XY plane
    
    return qx, qy, qz, qw, nx, ny, nz


def remove_bright_regions(mask_bin, frame, brightness_threshold=200):
    """
    Aggressively remove bright regions (windows/glass) from a mask.
    Uses erosion on bright pixels to create exclusion zones.
    
    Args:
        mask_bin: Binary mask of the region
        frame: Color frame to analyze brightness
        brightness_threshold: Pixel brightness threshold (0-255) above which to exclude
    
    Returns:
        Cleaned mask with bright regions removed
    """
    if frame is None:
        return mask_bin
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create bright regions mask (very aggressive)
    bright_mask = (gray > brightness_threshold).astype(np.uint8)
    
    # Dilate bright regions to expand exclusion zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bright_expanded = cv2.dilate(bright_mask, kernel, iterations=2)
    
    # Subtract bright regions from the original mask
    cleaned_mask = mask_bin.copy()
    cleaned_mask[bright_expanded > 0] = 0
    
    # Apply morphological closing to fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
    
    return cleaned_mask


# ═══════════════════════════════════════════════════════════════
#  DISPLAY / CSV
# ═══════════════════════════════════════════════════════════════

def create_mask_overlay(frame, mask, color, alpha=0.3):
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    return cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)


def get_source_package_dir():
    env_dir = os.environ.get('OB_DETECTION_SOURCE_DIR')
    if env_dir:
        env_path = Path(env_dir).expanduser().resolve()
        if env_path.is_dir():
            return env_path

    module_path = Path(__file__).resolve()
    for parent in [Path.cwd().resolve(), *Path.cwd().resolve().parents,
                   module_path.parent, *module_path.parents]:
        candidates = (
            parent / 'car_spraying_ws_FULLY_ADAPTIVE' / 'src' / 'ob_detection' / 'ob_detection',
            parent / 'car_spraying_ws_FULLY_ADAPTIVE' / 'car_spraying_ws' / 'src' / 'ob_detection' / 'ob_detection',
            parent / 'car_spraying_ws' / 'src' / 'ob_detection' / 'ob_detection',
            parent / 'src' / 'ob_detection' / 'ob_detection',
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate

    return module_path.parent


SAVE_DIR = str(get_source_package_dir() / 'spray_paths')


def save_path_csv(points_metric, previous_file=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    if previous_file and os.path.isfile(previous_file):
        try:
            os.remove(previous_file)
        except OSError as e:
            print(f'[WARN] {e}')
    filename = os.path.join(SAVE_DIR, 'path_dim.csv')
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        # Write data rows only
        for p in points_metric:
            # Extract position
            x, y, z = p[0], p[1], p[2]
            # Extract orientation (quaternion) - default to identity if not provided
            qx, qy, qz, qw = p[3] if len(p) > 3 else 0.0, p[4] if len(p) > 4 else 0.0, p[5] if len(p) > 5 else 0.0, p[6] if len(p) > 6 else 1.0
            # Extract normal - default to zero if not provided
            nx, ny, nz = p[7] if len(p) > 7 else 0.0, p[8] if len(p) > 8 else 0.0, p[9] if len(p) > 9 else 0.0
            w.writerow([f'{x:.6f}', f'{y:.6f}', f'{z:.6f}', f'{qx:.6f}', f'{qy:.6f}', f'{qz:.6f}', f'{qw:.6f}', f'{nx:.6f}', f'{ny:.6f}', f'{nz:.6f}'])
    return filename


def save_mesh_ply(filename, vertices, faces):
    """Write a simple ASCII PLY mesh from vertices and triangular faces."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        for v in vertices:
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')


def coverage_status_and_color(coverage, unpainted_thresh, overpainted_thresh):
    if coverage < unpainted_thresh:
        return 'unpainted', (0, 80, 255)
    if coverage < overpainted_thresh:
        return 'good', (0, 220, 80)
    return 'overpainted', (255, 40, 40)


def save_colored_mesh_ply(filename, vertices, faces, face_colors):
    """Write an ASCII PLY mesh with per-face RGB coverage colors."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for v in vertices:
            f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face, color in zip(faces, face_colors):
            r, g, b = [int(c) for c in color]
            f.write(f'3 {face[0]} {face[1]} {face[2]} {r} {g} {b}\n')


def save_triangle_coverage_csv(filename, coverage, statuses,
                               centers=None, normals=None):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'triangle_id', 'coverage', 'status',
            'center_x', 'center_y', 'center_z',
            'normal_x', 'normal_y', 'normal_z',
        ])
        for idx, (value, status) in enumerate(zip(coverage, statuses)):
            center = centers[idx] if centers is not None else (0.0, 0.0, 0.0)
            normal = normals[idx] if normals is not None else (0.0, 0.0, 0.0)
            w.writerow([
                idx, f'{float(value):.6f}', status,
                f'{float(center[0]):.6f}',
                f'{float(center[1]):.6f}',
                f'{float(center[2]):.6f}',
                f'{float(normal[0]):.6f}',
                f'{float(normal[1]):.6f}',
                f'{float(normal[2]):.6f}',
            ])


def triangle_centers_and_normals(vertices, faces):
    if len(vertices) == 0 or len(faces) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32))

    tri_vertices = vertices[faces]
    centers = tri_vertices.mean(axis=1).astype(np.float32)
    normals = np.cross(
        tri_vertices[:, 1] - tri_vertices[:, 0],
        tri_vertices[:, 2] - tri_vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(
        normals, lengths,
        out=np.zeros_like(normals, dtype=np.float32),
        where=lengths > 1e-9)
    return centers, normals.astype(np.float32)


def accumulate_triangle_coverage(vertices, faces, spray_points,
                                 spray_radius_m, paint_amount):
    """
    Add paint to nearby mesh triangles using nearest-center lookup.

    This is a lightweight substitute for Open3D RaycastingScene: each spray
    waypoint is treated as a paint sample and deposits more paint on closer
    triangle centers inside the spray radius.
    """
    if len(vertices) == 0 or len(faces) == 0 or len(spray_points) == 0:
        return np.zeros(len(faces), dtype=np.float32)

    centers, _ = triangle_centers_and_normals(vertices, faces)
    coverage = np.zeros(len(faces), dtype=np.float32)
    radius = max(float(spray_radius_m), 1e-6)
    amount = float(paint_amount)

    for point in np.asarray(spray_points, dtype=np.float32)[:, :3]:
        diff = centers - point
        dist = np.linalg.norm(diff, axis=1)
        hit_ids = np.flatnonzero(dist <= radius)

        if hit_ids.size == 0:
            hit_ids = np.array([int(np.argmin(dist))], dtype=np.int32)
            weights = np.array([1.0], dtype=np.float32)
        else:
            weights = 1.0 - (dist[hit_ids] / radius)
            weights = np.clip(weights, 0.05, 1.0).astype(np.float32)

        coverage[hit_ids] += amount * weights

    return coverage


def coverage_face_colors(coverage, unpainted_thresh, overpainted_thresh):
    statuses = []
    colors = []
    for value in coverage:
        status, color = coverage_status_and_color(
            float(value), unpainted_thresh, overpainted_thresh)
        statuses.append(status)
        colors.append(color)
    return statuses, np.asarray(colors, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════
#  ROS 2 NODE
# ═══════════════════════════════════════════════════════════════

class CarPartsSegmentationSprayNode(Node):

    def __init__(self):
        super().__init__('car_parts_segmentation_spray')

        self.declare_parameter('confidence',        0.35)
        self.declare_parameter('iou_threshold',     0.45)
        self.declare_parameter('spiral_spacing_px', 20)
        self.declare_parameter('min_path_points', 300)
        self.declare_parameter('max_path_points', 300)
        self.declare_parameter('brightness_threshold', 200)
        self.declare_parameter('show_masks',        True)
        self.declare_parameter('mask_alpha',        0.15)
        self.declare_parameter('camera_topic',      '/color_image/compressed')
        self.declare_parameter('depth_topic',       '/depth_image')
        self.declare_parameter('use_point_cloud',   True)
        self.declare_parameter('point_cloud_topic', '/pointcloud')
        self.declare_parameter('point_cloud_timeout_sec', 0.35)
        self.declare_parameter('publish_segmented_cloud', True)
        self.declare_parameter('segmented_cloud_topic', '/segmented_part_cloud')
        self.declare_parameter('cloud_from_depth_fallback', True)
        self.declare_parameter('enable_replica_mesh', True)
        self.declare_parameter('mesh_sampling_px', 2)
        self.declare_parameter('enable_mesh_coverage', True)
        self.declare_parameter('spray_radius_m', 0.025)
        self.declare_parameter('spray_paint_amount', 1.0)
        self.declare_parameter('coverage_unpainted_thresh', 10.0)
        self.declare_parameter('coverage_overpainted_thresh', 50.0)
        self.declare_parameter('cloud_roi_px',      2)
        self.declare_parameter('cloud_scale_x',     0.0)
        self.declare_parameter('cloud_scale_y',     0.0)
        self.declare_parameter('cloud_offset_x_px', 0.0)
        self.declare_parameter('cloud_offset_y_px', 0.0)
        self.declare_parameter('max_segmented_cloud_points', 12000)
        self.declare_parameter('depth_timeout_sec', 0.35)
        self.declare_parameter('depth_roi_px',      3)
        self.declare_parameter('min_depth_m',       0.05)
        self.declare_parameter('max_depth_m',       5.0)
        self.declare_parameter('camera_fx_px',      700.0)
        self.declare_parameter('camera_fy_px',      700.0)
        self.declare_parameter('camera_cx_px',      -1.0)
        self.declare_parameter('camera_cy_px',      -1.0)
        self.declare_parameter('depth_scale_x',     0.0)
        self.declare_parameter('depth_scale_y',     0.0)
        self.declare_parameter('depth_offset_x_px', 0.0)
        self.declare_parameter('depth_offset_y_px', 0.0)
        self.declare_parameter('real_ref_width_cm', 14.0)

        self.conf_thresh       = self.get_parameter('confidence').value
        self.iou_thresh        = self.get_parameter('iou_threshold').value
        self.spiral_spacing_px = self.get_parameter('spiral_spacing_px').value
        self.min_path_points   = self.get_parameter('min_path_points').value
        self.max_path_points   = self.get_parameter('max_path_points').value
        self.brightness_threshold = self.get_parameter('brightness_threshold').value
        self.show_masks        = self.get_parameter('show_masks').value
        self.mask_alpha        = self.get_parameter('mask_alpha').value
        self.camera_topic      = self.get_parameter('camera_topic').value
        self.depth_topic       = self.get_parameter('depth_topic').value
        self.use_point_cloud   = self.get_parameter('use_point_cloud').value
        self.point_cloud_topic = self.get_parameter('point_cloud_topic').value
        self.point_cloud_timeout_sec = self.get_parameter('point_cloud_timeout_sec').value
        self.publish_segmented_cloud = self.get_parameter('publish_segmented_cloud').value
        self.segmented_cloud_topic = self.get_parameter('segmented_cloud_topic').value
        self.cloud_from_depth_fallback = self.get_parameter('cloud_from_depth_fallback').value
        self.enable_replica_mesh = self.get_parameter('enable_replica_mesh').value
        self.mesh_sampling_px = self.get_parameter('mesh_sampling_px').value
        self.enable_mesh_coverage = self.get_parameter('enable_mesh_coverage').value
        self.spray_radius_m = self.get_parameter('spray_radius_m').value
        self.spray_paint_amount = self.get_parameter('spray_paint_amount').value
        self.coverage_unpainted_thresh = self.get_parameter(
            'coverage_unpainted_thresh').value
        self.coverage_overpainted_thresh = self.get_parameter(
            'coverage_overpainted_thresh').value
        self.cloud_roi_px      = self.get_parameter('cloud_roi_px').value
        self.cloud_scale_x     = self.get_parameter('cloud_scale_x').value
        self.cloud_scale_y     = self.get_parameter('cloud_scale_y').value
        self.cloud_offset_x_px = self.get_parameter('cloud_offset_x_px').value
        self.cloud_offset_y_px = self.get_parameter('cloud_offset_y_px').value
        self.max_segmented_cloud_points = self.get_parameter('max_segmented_cloud_points').value
        self.depth_timeout_sec = self.get_parameter('depth_timeout_sec').value
        self.depth_roi_px      = self.get_parameter('depth_roi_px').value
        self.min_depth_m       = self.get_parameter('min_depth_m').value
        self.max_depth_m       = self.get_parameter('max_depth_m').value
        self.camera_fx_px      = self.get_parameter('camera_fx_px').value
        self.camera_fy_px      = self.get_parameter('camera_fy_px').value
        self.camera_cx_px      = self.get_parameter('camera_cx_px').value
        self.camera_cy_px      = self.get_parameter('camera_cy_px').value
        self.depth_scale_x     = self.get_parameter('depth_scale_x').value
        self.depth_scale_y     = self.get_parameter('depth_scale_y').value
        self.depth_offset_x_px = self.get_parameter('depth_offset_x_px').value
        self.depth_offset_y_px = self.get_parameter('depth_offset_y_px').value
        self.real_ref_width_cm = self.get_parameter('real_ref_width_cm').value

        model_path = '/home/user/car_spraying_ws/src/ob_detection/ob_detection/car_parts_best_seg.pt'
        if not os.path.exists(model_path):
            self.get_logger().warn(f'Model not found: {model_path}')
            model_path = '/home/user/car_spraying_ws/src/ob_detection/ob_detection/car_parts_best.pt'

        self.get_logger().info(f'Loading: {model_path}')
        self.model  = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Device: {self.device.upper()}')
        self.model.to(self.device)

        camera_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        depth_qos = QoSProfile(depth=3)
        cloud_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(CompressedImage, self.camera_topic,
                                 self.camera_callback, camera_qos)
        self.create_subscription(Image, self.depth_topic,
                                 self.depth_callback, depth_qos)
        if self.use_point_cloud and point_cloud2 is not None:
            self.create_subscription(PointCloud2, self.point_cloud_topic,
                                     self.point_cloud_callback, cloud_qos)
        elif self.use_point_cloud:
            self.get_logger().warn(
                'sensor_msgs_py.point_cloud2 is not available; point cloud disabled')

        self.segmented_cloud_pub = None
        if (self.use_point_cloud and self.publish_segmented_cloud and
                point_cloud2 is not None):
            self.segmented_cloud_pub = self.create_publisher(
                PointCloud2, self.segmented_cloud_topic, 1)

        self._display_frame  = None
        self._frame_lock     = threading.Lock()
        self._depth_lock     = threading.Lock()
        self._cloud_lock     = threading.Lock()
        self._latest_depth   = None
        self._latest_depth_stamp = None
        self._latest_depth_header = None
        self._latest_cloud   = None
        self._latest_cloud_stamp = None
        self._cloud_info_logged = False
        self._last_tick      = cv2.getTickCount()
        self._fps            = 0.0
        self.part_colors     = {}
        self._last_saved     = None
        self.save_counter    = 0

        self.get_logger().info(
            f'LiDAR depth enabled: camera={self.camera_topic}, depth={self.depth_topic}')
        if self.use_point_cloud and point_cloud2 is not None:
            self.get_logger().info(
                f'Point cloud enabled: cloud={self.point_cloud_topic}')
        self.get_logger().info('READY — SQUARE SPIRAL (polygon offset rings)')

    def get_part_color(self, cid, cname):
        if cid not in self.part_colors:
            np.random.seed(hash(cname) % 2**32)
            self.part_colors[cid] = tuple(np.random.randint(50, 255, 3).tolist())
        return self.part_colors[cid]

    def estimate_depth_m(self, px_w):
        real_width_m = self.real_ref_width_cm / 100.0
        return real_width_m * self.camera_fx_px / max(px_w, 1)

    def depth_callback(self, msg):
        if msg.encoding not in ('16UC1', 'mono16'):
            self.get_logger().warn(
                f'Unsupported depth encoding {msg.encoding}; expected 16UC1',
                throttle_duration_sec=2.0)
            return
        if msg.height == 0 or msg.width == 0 or not msg.data:
            return

        depth_mm = np.frombuffer(msg.data, dtype=np.uint16)
        expected = int(msg.height * msg.width)
        if depth_mm.size < expected:
            self.get_logger().warn(
                f'Depth image too small: got {depth_mm.size}, expected {expected}',
                throttle_duration_sec=2.0)
            return

        depth_mm = depth_mm[:expected].reshape((msg.height, msg.width)).copy()
        with self._depth_lock:
            self._latest_depth = depth_mm
            self._latest_depth_stamp = msg.header.stamp
            self._latest_depth_header = msg.header

    def get_latest_depth(self):
        with self._depth_lock:
            if self._latest_depth is None:
                return None, None, None
            return (self._latest_depth.copy(), self._latest_depth_stamp,
                    self._latest_depth_header)

    def point_cloud_callback(self, msg):
        if not self._cloud_info_logged:
            self.get_logger().info(
                f'PointCloud2 received: width={msg.width}, height={msg.height}, '
                f'point_step={msg.point_step}, row_step={msg.row_step}, '
                f'is_dense={msg.is_dense}')
            self._cloud_info_logged = True
        if msg.height <= 1:
            self.get_logger().warn(
                'Point cloud is unorganized; need an organized cloud for mask mapping',
                throttle_duration_sec=2.0)
            return
        with self._cloud_lock:
            self._latest_cloud = msg
            self._latest_cloud_stamp = msg.header.stamp

    def get_latest_cloud(self):
        with self._cloud_lock:
            return self._latest_cloud, self._latest_cloud_stamp

    def stamp_age_sec(self, stamp):
        if stamp is None:
            return None
        now = self.get_clock().now().nanoseconds
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        return abs(now - stamp_ns) / 1e9

    def color_to_depth_pixel(self, x, y, img_w, img_h, depth_w, depth_h):
        sx = self.depth_scale_x if self.depth_scale_x > 0 else depth_w / max(img_w, 1)
        sy = self.depth_scale_y if self.depth_scale_y > 0 else depth_h / max(img_h, 1)
        u = int(round(x * sx + self.depth_offset_x_px))
        v = int(round(y * sy + self.depth_offset_y_px))
        return u, v

    def color_to_cloud_pixel(self, x, y, img_w, img_h, cloud_w, cloud_h):
        sx = self.cloud_scale_x if self.cloud_scale_x > 0 else cloud_w / max(img_w, 1)
        sy = self.cloud_scale_y if self.cloud_scale_y > 0 else cloud_h / max(img_h, 1)
        u = int(round(x * sx + self.cloud_offset_x_px))
        v = int(round(y * sy + self.cloud_offset_y_px))
        return u, v

    def depth_at_color_pixel(self, x, y, img_w, img_h, depth_img):
        depth_h, depth_w = depth_img.shape[:2]
        u, v = self.color_to_depth_pixel(x, y, img_w, img_h, depth_w, depth_h)
        if u < 0 or v < 0 or u >= depth_w or v >= depth_h:
            return None

        roi = max(0, int(self.depth_roi_px))
        x1 = max(0, u - roi)
        x2 = min(depth_w, u + roi + 1)
        y1 = max(0, v - roi)
        y2 = min(depth_h, v + roi + 1)
        samples_m = depth_img[y1:y2, x1:x2].astype(np.float32) / 1000.0
        valid = samples_m[
            (samples_m >= self.min_depth_m) &
            (samples_m <= self.max_depth_m) &
            np.isfinite(samples_m)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def mask_depth_median(self, mask_bin, img_w, img_h, depth_img):
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return None
        step = max(1, int(np.ceil(xs.size / 5000)))

        depths = []
        for x, y in zip(xs[::step], ys[::step]):
            z_m = self.depth_at_color_pixel(int(x), int(y), img_w, img_h, depth_img)
            if z_m is not None:
                depths.append(z_m)

        if not depths:
            return None
        valid = np.array(depths, dtype=np.float32)
        return float(np.median(valid))

    def pixel_to_camera_point(self, x, y, z_m, img_w, img_h):
        cx = self.camera_cx_px if self.camera_cx_px >= 0 else img_w / 2.0
        cy = self.camera_cy_px if self.camera_cy_px >= 0 else img_h / 2.0
        fx = max(float(self.camera_fx_px), 1e-6)
        fy = max(float(self.camera_fy_px), 1e-6)
        return [(x - cx) * z_m / fx,
                (y - cy) * z_m / fy,
                z_m]

    def mask_depth_cloud_points(self, mask_bin, img_w, img_h, depth_img):
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        max_points = max(1, int(self.max_segmented_cloud_points))
        step = max(1, int(np.ceil(xs.size / max_points)))
        points = []
        for x, y in zip(xs[::step], ys[::step]):
            z_m = self.depth_at_color_pixel(int(x), int(y), img_w, img_h, depth_img)
            if z_m is not None:
                points.append(self.pixel_to_camera_point(
                    int(x), int(y), z_m, img_w, img_h))

        return np.asarray(points, dtype=np.float32)

    def valid_cloud_point(self, point):
        if point is None or len(point) < 3:
            return False
        x, y, z = point[:3]
        return (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and
                self.min_depth_m <= float(z) <= self.max_depth_m)

    def point_cloud_at_color_pixel(self, x, y, img_w, img_h, cloud_msg):
        if point_cloud2 is None or cloud_msg is None or cloud_msg.height <= 1:
            return None

        cloud_w, cloud_h = cloud_msg.width, cloud_msg.height
        u, v = self.color_to_cloud_pixel(x, y, img_w, img_h, cloud_w, cloud_h)
        if u < 0 or v < 0 or u >= cloud_w or v >= cloud_h:
            return None

        roi = max(0, int(self.cloud_roi_px))
        u1 = max(0, u - roi)
        u2 = min(cloud_w, u + roi + 1)
        v1 = max(0, v - roi)
        v2 = min(cloud_h, v + roi + 1)
        uvs = [(uu, vv) for vv in range(v1, v2) for uu in range(u1, u2)]
        points = []
        for point in point_cloud2.read_points(
                cloud_msg, field_names=('x', 'y', 'z'),
                skip_nans=True, uvs=uvs):
            if self.valid_cloud_point(point):
                points.append([float(point[0]), float(point[1]), float(point[2])])

        if not points:
            return None
        return np.median(np.asarray(points, dtype=np.float32), axis=0).tolist()

    def segmented_cloud_points(self, mask_bin, img_w, img_h, cloud_msg):
        if point_cloud2 is None or cloud_msg is None or cloud_msg.height <= 1:
            return np.empty((0, 3), dtype=np.float32)

        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        max_points = max(1, int(self.max_segmented_cloud_points))
        step = max(1, int(np.ceil(xs.size / max_points)))
        cloud_w, cloud_h = cloud_msg.width, cloud_msg.height
        uvs = []
        seen = set()
        for x, y in zip(xs[::step], ys[::step]):
            u, v = self.color_to_cloud_pixel(
                int(x), int(y), img_w, img_h, cloud_w, cloud_h)
            if 0 <= u < cloud_w and 0 <= v < cloud_h and (u, v) not in seen:
                uvs.append((u, v))
                seen.add((u, v))

        points = []
        for point in point_cloud2.read_points(
                cloud_msg, field_names=('x', 'y', 'z'),
                skip_nans=True, uvs=uvs):
            if self.valid_cloud_point(point):
                points.append([float(point[0]), float(point[1]), float(point[2])])

        return np.asarray(points, dtype=np.float32)

    def publish_segmented_cloud_msg(self, points, header_or_cloud):
        if (self.segmented_cloud_pub is None or point_cloud2 is None or
                points.size == 0 or header_or_cloud is None):
            return
        header = getattr(header_or_cloud, 'header', header_or_cloud)
        msg = point_cloud2.create_cloud_xyz32(header, points.tolist())
        self.segmented_cloud_pub.publish(msg)

    def organized_cloud_xyz(self, cloud_msg):
        if point_cloud2 is None or cloud_msg is None or cloud_msg.height <= 1:
            return None

        xyz = np.full((cloud_msg.height, cloud_msg.width, 3), np.nan, dtype=np.float32)
        point_iter = point_cloud2.read_points(
            cloud_msg, field_names=('x', 'y', 'z'), skip_nans=False)
        for idx, point in enumerate(point_iter):
            if idx >= cloud_msg.width * cloud_msg.height:
                break
            y = idx // cloud_msg.width
            x = idx % cloud_msg.width
            xyz[y, x] = [float(point[0]), float(point[1]), float(point[2])]
        return xyz

    def make_replica_mesh(self, mask_bin, img_w, img_h, cloud_msg=None, depth_img=None):
        if mask_bin is None or mask_bin.sum() == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        cloud_xyz = None
        if cloud_msg is not None and cloud_msg.height > 1:
            cloud_xyz = self.organized_cloud_xyz(cloud_msg)

        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        y1 += 1
        x1 += 1

        step = max(1, int(self.mesh_sampling_px))
        sample_ys = list(range(y0, y1, step))
        sample_xs = list(range(x0, x1, step))
        valid = np.zeros((len(sample_ys), len(sample_xs)), dtype=bool)
        points = np.full((len(sample_ys), len(sample_xs), 3), np.nan, dtype=np.float32)

        for gy, y in enumerate(sample_ys):
            for gx, x in enumerate(sample_xs):
                if y >= img_h or x >= img_w or mask_bin[y, x] == 0:
                    continue

                pt = None
                if cloud_xyz is not None:
                    u, v = self.color_to_cloud_pixel(x, y, img_w, img_h,
                                                    cloud_msg.width, cloud_msg.height)
                    if 0 <= u < cloud_msg.width and 0 <= v < cloud_msg.height:
                        p = cloud_xyz[v, u]
                        if self.valid_cloud_point(p):
                            pt = p

                if pt is None and depth_img is not None:
                    z_m = self.depth_at_color_pixel(x, y, img_w, img_h, depth_img)
                    if z_m is not None:
                        pt = np.asarray(self.pixel_to_camera_point(x, y, z_m, img_w, img_h), dtype=np.float32)

                if pt is not None and np.isfinite(pt).all():
                    valid[gy, gx] = True
                    points[gy, gx] = pt

        if not valid.any():
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

        index_map = -np.ones_like(valid, dtype=np.int32)
        vertices = []
        for y in range(valid.shape[0]):
            for x in range(valid.shape[1]):
                if valid[y, x]:
                    index_map[y, x] = len(vertices)
                    vertices.append(points[y, x].tolist())

        faces = []
        for y in range(valid.shape[0] - 1):
            for x in range(valid.shape[1] - 1):
                corners = [valid[y, x], valid[y, x + 1], valid[y + 1, x], valid[y + 1, x + 1]]
                if all(corners):
                    v0 = index_map[y, x]
                    v1 = index_map[y, x + 1]
                    v2 = index_map[y + 1, x]
                    v3 = index_map[y + 1, x + 1]
                    faces.append([v0, v2, v1])
                    faces.append([v1, v2, v3])
                else:
                    # Add triangles for any 3 valid corners to preserve mesh connectivity
                    if corners[0] and corners[1] and corners[2]:
                        faces.append([index_map[y, x], index_map[y, x + 1], index_map[y + 1, x]])
                    if corners[1] and corners[2] and corners[3]:
                        faces.append([index_map[y, x + 1], index_map[y + 1, x + 1], index_map[y + 1, x]])
                    if corners[0] and corners[1] and corners[3]:
                        faces.append([index_map[y, x], index_map[y, x + 1], index_map[y + 1, x + 1]])
                    if corners[0] and corners[2] and corners[3]:
                        faces.append([index_map[y, x], index_map[y + 1, x], index_map[y + 1, x + 1]])

        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)
        return vertices, faces

    def cloud_depth_median(self, points):
        if points.size == 0:
            return None
        return float(np.median(points[:, 2]))

    def camera_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        img_h, img_w = frame.shape[:2]
        depth_img, depth_stamp, depth_header = self.get_latest_depth()
        depth_age = self.stamp_age_sec(depth_stamp)
        depth_ok = depth_img is not None and (
            depth_age is None or depth_age <= self.depth_timeout_sec)
        cloud_msg, cloud_stamp = self.get_latest_cloud()
        cloud_age = self.stamp_age_sec(cloud_stamp)
        cloud_ok = (self.use_point_cloud and point_cloud2 is not None and
                    cloud_msg is not None and cloud_msg.height > 1 and
                    (cloud_age is None or
                     cloud_age <= self.point_cloud_timeout_sec))

        results = self.model.predict(
            source=frame, conf=self.conf_thresh, iou=self.iou_thresh,
            imgsz=640, verbose=False, device=self.device, retina_masks=True)

        now = cv2.getTickCount()
        dt  = (now - self._last_tick) / cv2.getTickFrequency()
        self._fps       = 1.0 / max(dt, 1e-6)
        self._last_tick = now

        display_frame = frame.copy()

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

            # First pass: collect all detections organized by label
            detections = []
            windows_masks = []
            for box, mask in zip(results[0].boxes, masks):
                if mask.ndim == 3:
                    mask = mask.squeeze(0)
                mask_bin = (mask > 0.5).astype(np.uint8)
                if mask_bin.shape != (img_h, img_w):
                    mask_bin = cv2.resize(mask_bin, (img_w, img_h),
                                          interpolation=cv2.INTER_NEAREST)
                conf    = float(box.conf[0])
                cls     = int(box.cls[0])
                label   = self.model.names[cls]
                detections.append({
                    'box': box,
                    'mask_bin': mask_bin,
                    'conf': conf,
                    'cls': cls,
                    'label': label
                })
                if label.lower() == 'window':
                    windows_masks.append(mask_bin)

            # Second pass: process detections, excluding windows from doors
            for det in detections:
                box, mask_bin = det['box'], det['mask_bin']
                conf, cls, label = det['conf'], det['cls'], det['label']

                # Remove bright regions (windows/glass) from all detections
                mask_bin = remove_bright_regions(mask_bin, frame, self.brightness_threshold)

                # If this is a door, also subtract all window masks
                if label.lower() == 'door':
                    if windows_masks:
                        for win_mask in windows_masks:
                            mask_bin = np.bitwise_and(mask_bin, np.bitwise_not(win_mask))

                # Skip if mask is now empty
                if mask_bin.sum() == 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                segmented_cloud = np.empty((0, 3), dtype=np.float32)
                segmented_cloud_source = None
                if cloud_ok:
                    segmented_cloud = self.segmented_cloud_points(
                        mask_bin, img_w, img_h, cloud_msg)
                    if segmented_cloud.size > 0:
                        segmented_cloud_source = 'point cloud'
                    self.publish_segmented_cloud_msg(segmented_cloud, cloud_msg)
                if (segmented_cloud.size == 0 and depth_ok and
                        self.cloud_from_depth_fallback):
                    segmented_cloud = self.mask_depth_cloud_points(
                        mask_bin, img_w, img_h, depth_img)
                    if segmented_cloud.size > 0:
                        segmented_cloud_source = 'depth cloud'
                        self.publish_segmented_cloud_msg(
                            segmented_cloud, depth_header)

                mask_depth_m = None
                if segmented_cloud.size > 0:
                    mask_depth_m = self.cloud_depth_median(segmented_cloud)
                elif depth_ok:
                    mask_depth_m = self.mask_depth_median(mask_bin, img_w, img_h, depth_img)
                if mask_depth_m is None:
                    mask_depth_m = self.estimate_depth_m(x2 - x1)
                    self.get_logger().warn(
                        'No valid point cloud/depth for detection; using width fallback',
                        throttle_duration_sec=2.0)

                path_px = generate_square_spiral(
                    mask_bin, spacing_px=self.spiral_spacing_px)
                if len(path_px) < 4:
                    continue
                target_points = adaptive_path_target(
                    len(path_px), self.min_path_points, self.max_path_points)
                if len(path_px) != target_points:
                    path_px = resample_path_to_count(path_px, target_points)

                path_metric = []
                cloud_path_points = 0
                depth_cloud_path_points = 0
                for pt in path_px:
                    cloud_point = None
                    if cloud_ok:
                        cloud_point = self.point_cloud_at_color_pixel(
                            int(pt[0]), int(pt[1]), img_w, img_h, cloud_msg)
                    if cloud_point is not None:
                        path_metric.append(cloud_point)
                        cloud_path_points += 1
                        continue

                    z_m = None
                    if depth_ok:
                        z_m = self.depth_at_color_pixel(
                            int(pt[0]), int(pt[1]), img_w, img_h, depth_img)
                    if z_m is None:
                        z_m = mask_depth_m
                    path_metric.append(self.pixel_to_camera_point(
                        int(pt[0]), int(pt[1]), z_m, img_w, img_h))
                    if segmented_cloud_source == 'depth cloud':
                        depth_cloud_path_points += 1
                
                # Add orientation and normal data to each point
                path_metric_with_orientation = []
                for i, pt in enumerate(path_metric):
                    qx, qy, qz, qw, nx, ny, nz = compute_path_orientation_and_normal(path_metric, i)
                    # Extend point with orientation and normal data
                    pt_extended = list(pt) + [qx, qy, qz, qw, nx, ny, nz]
                    path_metric_with_orientation.append(pt_extended)

                fname = save_path_csv(path_metric_with_orientation, self._last_saved)
                self._last_saved   = fname
                mesh_fname = None
                if self.enable_replica_mesh:
                    mesh_vertices, mesh_faces = self.make_replica_mesh(
                        mask_bin, img_w, img_h,
                        cloud_msg if cloud_ok else None,
                        depth_img if depth_ok else None)
                    if mesh_vertices.shape[0] >= 3:
                        mesh_fname = os.path.join(SAVE_DIR, 'part_mesh.ply')
                        save_mesh_ply(mesh_fname, mesh_vertices, mesh_faces)
                        if mesh_faces.shape[0] >= 1:
                            self.get_logger().info(
                                f'[MESH] saved {mesh_fname} '
                                f'({len(mesh_vertices)} verts, {len(mesh_faces)} faces)')
                            if self.enable_mesh_coverage:
                                triangle_coverage = accumulate_triangle_coverage(
                                    mesh_vertices, mesh_faces, path_metric,
                                    self.spray_radius_m, self.spray_paint_amount)
                                triangle_centers, triangle_normals = (
                                    triangle_centers_and_normals(
                                        mesh_vertices, mesh_faces))
                                coverage_statuses, face_colors = coverage_face_colors(
                                    triangle_coverage,
                                    self.coverage_unpainted_thresh,
                                    self.coverage_overpainted_thresh)
                                coverage_mesh_fname = os.path.join(
                                    SAVE_DIR, 'part_mesh_coverage.ply')
                                coverage_csv_fname = os.path.join(
                                    SAVE_DIR, 'triangle_coverage.csv')
                                save_colored_mesh_ply(
                                    coverage_mesh_fname, mesh_vertices,
                                    mesh_faces, face_colors)
                                save_triangle_coverage_csv(
                                    coverage_csv_fname, triangle_coverage,
                                    coverage_statuses,
                                    triangle_centers, triangle_normals)
                                counts = {
                                    name: coverage_statuses.count(name)
                                    for name in ('unpainted', 'good', 'overpainted')
                                }
                                self.get_logger().info(
                                    f'[COVERAGE] saved {coverage_mesh_fname} and '
                                    f'{coverage_csv_fname} '
                                    f"(unpainted={counts['unpainted']}, "
                                    f"good={counts['good']}, "
                                    f"overpainted={counts['overpainted']})")
                        else:
                            self.get_logger().warn(
                                f'[MESH] saved {mesh_fname} ({len(mesh_vertices)} verts, no faces)')
                    else:
                        self.get_logger().warn(
                            '[MESH] not enough vertices to build replica mesh')

                self.save_counter += 1
                if cloud_path_points > 0:
                    path_source = 'point cloud'
                elif depth_cloud_path_points > 0:
                    path_source = 'depth cloud'
                else:
                    path_source = 'depth image'
                self.get_logger().info(
                    f'[SAVED #{self.save_counter}] {fname} '
                    f'({len(path_metric_with_orientation)} pts, {path_source}, '
                    f'{len(segmented_cloud)} segmented pts)')

                color = self.get_part_color(cls, label)

                if self.show_masks:
                    display_frame = create_mask_overlay(
                        display_frame, mask_bin, color, self.mask_alpha)
                    c = get_mask_contour(mask_bin)
                    if c is not None and len(c):
                        if c.ndim == 3:
                            c = c.squeeze()
                        cv2.drawContours(display_frame, [c.astype(np.int32)],
                                         -1, (255, 255, 255), 2)

                # draw path
                for i in range(1, len(path_px)):
                    p0, p1 = tuple(path_px[i-1]), tuple(path_px[i])
                    if (0 <= p0[0] < img_w and 0 <= p0[1] < img_h and
                            0 <= p1[0] < img_w and 0 <= p1[1] < img_h):
                        cv2.line(display_frame, p0, p1, (0, 200, 255), 1, cv2.LINE_AA)

                step = max(1, len(path_px) // 300)
                for i in range(0, len(path_px), step):
                    pt = path_px[i]
                    if 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
                        cv2.circle(display_frame, tuple(pt), 2, (255, 80, 0), -1)

                center = get_mask_center(mask_bin)
                if center:
                    cv2.circle(display_frame, center, 6, (0, 0, 255), -1)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (100, 255, 100), 1)
                source_text = path_source
                cv2.putText(display_frame,
                            (f'{label} {conf:.2f} | {source_text} | {mask_depth_m:.2f}m | '
                             f'{len(path_px)} pts | #{self.save_counter}'),
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        depth_text = 'LiDAR:waiting'
        if depth_img is not None:
            if depth_ok:
                depth_text = 'LiDAR:ok'
            else:
                depth_text = f'LiDAR:stale {depth_age:.1f}s'
        cloud_text = 'Cloud:off'
        if self.use_point_cloud and point_cloud2 is not None:
            cloud_text = 'Cloud:waiting'
            if cloud_msg is not None:
                if cloud_ok:
                    cloud_text = 'Cloud:ok'
                else:
                    cloud_text = f'Cloud:stale {cloud_age:.1f}s'
        cv2.putText(display_frame,
                    (f'FPS:{self._fps:.1f} | SQUARE SPIRAL | '
                     f'Spacing:{self.spiral_spacing_px}px | '
                     f'{depth_text} | {cloud_text}'),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, '+/- spacing | m=mask | q=quit',
                    (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)

        disp = cv2.resize(display_frame, (960, 540))
        with self._frame_lock:
            self._display_frame = disp

    def get_display_frame(self):
        with self._frame_lock:
            return self._display_frame


def main(args=None):
    rclpy.init(args=args)
    node = CarPartsSegmentationSprayNode()

    cv2.namedWindow('Car Parts | Square Spiral', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Car Parts | Square Spiral', 960, 540)

    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()

    print("\n" + "=" * 60)
    print("SQUARE SPIRAL — polygon inward offsetting")
    print("Each ring is a closed loop following the part shape")
    print(f"Save dir: {SAVE_DIR}")
    print("=" * 60 + "\n")

    try:
        while rclpy.ok():
            frame = node.get_display_frame()
            if frame is not None:
                cv2.imshow('Car Parts | Square Spiral', frame)
            key = cv2.waitKey(30) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord('m'):
                node.show_masks = not node.show_masks
            elif key in [ord('+'), ord('=')]:
                node.spiral_spacing_px = min(node.spiral_spacing_px + 2, 50)
                print(f'Spacing: {node.spiral_spacing_px}px')
            elif key in [ord('-'), ord('_')]:
                node.spiral_spacing_px = max(node.spiral_spacing_px - 2, 2)
                print(f'Spacing: {node.spiral_spacing_px}px')
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()
        t.join(timeout=2.0)
        print(f"\nTotal saves: {node.save_counter}  |  Dir: {SAVE_DIR}")


if __name__ == '__main__':
    main()
