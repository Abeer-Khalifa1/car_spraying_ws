import argparse
import os
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except Exception:
    ROS_AVAILABLE = False


# ============================================================
# Paint color HSV range for Method 1 binary coverage detection.
# Tune lower_paint / upper_paint to match your actual paint color.
# Default targets a mid-blue paint. Press 'c' at runtime to print
# the HSV value of the centre pixel so you can calibrate quickly.
# ============================================================
PAINT_HSV_LOWER = np.array([100, 50, 50],  dtype=np.uint8)   # hue 100-130 = blue
PAINT_HSV_UPPER = np.array([130, 255, 255], dtype=np.uint8)

# Grid dimensions for Method 1 coverage map
GRID_ROWS = 20
GRID_COLS = 20

# Minimum paint-coverage fraction per cell to count as "covered" (0-1)
CELL_COVERED_THRESH = 0.50


class SprayQualityInspectorV2:
    def __init__(self):
        # adaptive thresholds (you can later auto-learn these)
        # thresholds (tunable). coverage/drip scaled for easier interpretation.
        self.coverage_thresh = 5.0
        self.drip_thresh = 0.5
        # roughness threshold is on a 0-100 normalized scale after changes below
        self.rough_thresh = 25

        # Method 1: minimum fraction of grid cells that must be covered (0-1)
        self.grid_coverage_thresh = 0.80

        # HSV paint bounds (mutable so runtime calibration can update them)
        self.paint_lower = PAINT_HSV_LOWER.copy()
        self.paint_upper = PAINT_HSV_UPPER.copy()

        # --- stability state (FIX) -----------------------------------
        # The old code re-decided "use configured color vs. adaptive
        # dominant-hue color" independently every single frame, with no
        # memory between frames. Any per-frame noise (lighting flicker,
        # compression artifacts, etc.) could flip that decision, which is
        # why the mask used to look completely different frame-to-frame
        # ("sometimes masks everything, sometimes mixes").
        #
        # Now we require several consecutive low-detection frames before
        # switching to adaptive mode, and once adaptive mode locks onto a
        # hue range we keep reusing that same range instead of re-deriving
        # a (possibly different) dominant color every frame.
        self._adaptive_locked = False
        self._adaptive_lower = None
        self._adaptive_upper = None
        self._adaptive_fail_count = 0
        self._adaptive_lock_frames = 5

        # EMA-smoothed drip score, so a single noisy frame full of random
        # spurious "vertical lines" doesn't flip the GOOD/BAD status.
        self._drip_score_ema = 0.0
        self._drip_ema_alpha = 0.3

    # -----------------------------
    # 1. Preprocessing (FIXED LIGHTING)
    # -----------------------------
    def preprocess(self, img):
        img = cv2.resize(img, (640, 640))

        # illumination normalization
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # FIX: cv2.equalizeHist() stretches contrast using the FULL frame's
        # histogram, which changes based on whatever else is in view. That
        # means the same physical paint color gets remapped to different
        # pixel values from one frame to the next, which is exactly what
        # was making the fixed HSV paint range randomly stop matching (and
        # made the drip-edge detector noisy). CLAHE applies a clipped,
        # localized equalization that is far more repeatable frame-to-frame.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        norm = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)

        # FIX: a dedicated, lightly-blurred grayscale specifically for edge
        # detection (drips). Any contrast normalization amplifies small
        # sensor noise into fake "edges"; a small blur removes that without
        # meaningfully blurring an actual drip streak.
        edge_gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return norm, hsv, gray, edge_gray

    # -----------------------------
    # 2. ROI EXTRACTION (IMPORTANT)
    # -----------------------------
    def get_roi_mask(self, gray):
        # assume car panel is largest smooth region
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # clean noise
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        return mask

    # -----------------------------
    # 3. METHOD 1 — Binary Paint Coverage + Grid Map
    # -----------------------------
    def detect_binary_coverage(self, hsv, roi_mask):
        h, w = hsv.shape[:2]

        # 1) Threshold paint colour using the configured (or previously
        #    locked-in adaptive) range
        paint_mask = cv2.inRange(hsv, self.paint_lower, self.paint_upper)
        paint_mask = cv2.bitwise_and(paint_mask, paint_mask, mask=roi_mask)

        roi_area = int(np.count_nonzero(roi_mask))
        detected_px = int(np.count_nonzero(paint_mask))
        low_detection = detected_px < max(50, int(roi_area * 0.02))

        # FIX: hysteresis instead of an instant per-frame switch. Only
        # count consecutive failures, and only act once we've seen enough
        # of them in a row to trust it's a real color mismatch and not one
        # noisy frame.
        if low_detection:
            self._adaptive_fail_count += 1
        else:
            self._adaptive_fail_count = 0
            self._adaptive_locked = False  # configured color is working again

        if low_detection and self._adaptive_fail_count >= self._adaptive_lock_frames:
            if not self._adaptive_locked:
                learned = self.adaptive_paint_mask(hsv, roi_mask, return_bounds=True)
                if learned is not None:
                    mask_candidate, low_h, high_h = learned
                    self._adaptive_lower, self._adaptive_upper = low_h, high_h
                    self._adaptive_locked = True
                    paint_mask = mask_candidate
            else:
                # FIX: reuse the previously locked hue range instead of
                # re-deriving a new (possibly different) dominant hue every
                # single frame — this is what used to cause the mask to
                # "mix" between different regions frame to frame.
                paint_mask = self._hue_mask(hsv, self._adaptive_lower, self._adaptive_upper)
                paint_mask = cv2.bitwise_and(paint_mask, paint_mask, mask=roi_mask)

        # 3) Morphological cleanup (fill small gaps, remove specks)
        kernel = np.ones((5, 5), np.uint8)
        paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_CLOSE, kernel)
        paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN,  kernel)

        # 4) Build grid coverage map
        grid_pct = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

        cell_h = h // GRID_ROWS
        cell_w = w // GRID_COLS

        covered_cells = 0
        total_cells   = 0

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w

                cell_roi   = roi_mask[y0:y1, x0:x1]
                cell_paint = paint_mask[y0:y1, x0:x1]

                roi_pixels = int(np.count_nonzero(cell_roi))
                if roi_pixels == 0:
                    grid_pct[r, c] = -1.0   # cell outside ROI — mark invalid
                    continue

                pct = float(np.count_nonzero(cell_paint)) / float(roi_pixels)
                grid_pct[r, c] = pct * 100.0

                total_cells += 1
                if pct >= CELL_COVERED_THRESH:
                    covered_cells += 1

        overall = (float(covered_cells) / float(max(total_cells, 1))) * 100.0

        # 5) Build colour-coded grid visualisation
        grid_vis = np.zeros((h, w, 3), dtype=np.uint8)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                pct = grid_pct[r, c]
                if pct < 0:
                    colour = (40, 40, 40)        # outside ROI — dark grey
                elif pct >= CELL_COVERED_THRESH * 100.0:
                    colour = (0, 200, 0)         # covered — green
                elif pct > 0:
                    t = pct / (CELL_COVERED_THRESH * 100.0)
                    colour = (0, int(200 * t), int(200 * (1 - t) + 55))
                else:
                    colour = (0, 0, 200)         # missed — red
                cv2.rectangle(grid_vis, (x0, y0), (x1 - 1, y1 - 1), colour, -1)
                if cell_h > 20 and cell_w > 20 and pct >= 0:
                    cv2.putText(grid_vis, f"{pct:.0f}", (x0 + 2, y0 + cell_h // 2),
                                cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)

        for r in range(GRID_ROWS + 1):
            cv2.line(grid_vis, (0, r * cell_h), (w, r * cell_h), (80, 80, 80), 1)
        for c in range(GRID_COLS + 1):
            cv2.line(grid_vis, (c * cell_w, 0), (c * cell_w, h), (80, 80, 80), 1)

        return overall, grid_pct, paint_mask, grid_vis

    def build_defect_matrix(self, grid_pct):
        defect_matrix = np.zeros_like(grid_pct, dtype=np.float32)

        for r in range(grid_pct.shape[0]):
            for c in range(grid_pct.shape[1]):
                pct = grid_pct[r, c]

                if pct < 0:
                    defect_matrix[r, c] = -1.0
                    continue

                if pct >= CELL_COVERED_THRESH * 100.0:
                    defect_matrix[r, c] = 0.0
                else:
                    defect_matrix[r, c] = (
                        1.0 - pct / (CELL_COVERED_THRESH * 100.0)
                    )

        return defect_matrix

    def _hue_mask(self, hsv, low_h, high_h):
        if low_h <= high_h:
            lower = np.array([low_h, 50, 50], dtype=np.uint8)
            upper = np.array([high_h, 255, 255], dtype=np.uint8)
            return cv2.inRange(hsv, lower, upper)

        lower1 = np.array([0, 50, 50], dtype=np.uint8)
        upper1 = np.array([high_h, 255, 255], dtype=np.uint8)
        lower2 = np.array([low_h, 50, 50], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

    def adaptive_paint_mask(self, hsv, roi_mask, return_bounds=False):
        roi_pixels = hsv[roi_mask > 0]
        if roi_pixels.size == 0:
            return None if return_bounds else np.zeros_like(roi_mask)

        # FIX: only consider reasonably saturated/bright pixels when picking
        # the "dominant" hue. Without this, the dominant color in a mostly
        # unpainted ROI is often shadow, glare, or bare metal — not paint —
        # so the fallback would lock onto the wrong thing entirely.
        saturated = roi_pixels[(roi_pixels[:, 1] > 60) & (roi_pixels[:, 2] > 60)]
        hue_pixels = (saturated[:, 0]
                      if saturated.size > int(0.1 * roi_pixels.shape[0])
                      else roi_pixels[:, 0])

        if hue_pixels.size == 0:
            return None if return_bounds else np.zeros_like(roi_mask)

        hist = np.bincount(hue_pixels, minlength=180)
        dominant = int(np.argmax(hist))

        low_h = dominant - 15
        high_h = dominant + 15
        if low_h < 0:
            low_h += 180
        if high_h > 179:
            high_h -= 180

        paint_mask = self._hue_mask(hsv, low_h, high_h)
        paint_mask = cv2.bitwise_and(paint_mask, paint_mask, mask=roi_mask)

        if return_bounds:
            return paint_mask, low_h, high_h
        return paint_mask

    # -----------------------------
    # 4. METHOD 2 — UNEVEN COVERAGE (brightness variance)
    # -----------------------------
    def detect_coverage(self, hsv, mask):
        v = hsv[:, :, 2]

        roi = v[mask > 0]

        if len(roi) == 0:
            return 0.0, np.zeros_like(v, dtype=np.uint8)

        mean = np.mean(roi)
        std = np.std(roi)

        score = std / (mean + 1e-6)
        score = float(score) * 100.0

        local_var = cv2.Laplacian(v, cv2.CV_64F)
        local_var = np.abs(local_var)

        heatmap = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

        return score, heatmap

    # -----------------------------
    # 5. DRIP DETECTION (Hough-based)
    # -----------------------------
    def detect_drips(self, gray, mask):
        # FIX: `gray` passed in here is now the pre-blurred edge_gray from
        # preprocess(), which removes the fake "edges" that contrast
        # normalization used to introduce as noise.
        edges = cv2.Canny(gray, 80, 180)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        roi_area = max(int(np.count_nonzero(mask)), 1)

        # FIX: scale the minimum line length to the ROI's actual size
        # instead of a fixed 40px. A fixed length either lets tiny noise
        # segments count as "drips" on a small part, or never fires at all
        # on a small part where a real drip is naturally shorter.
        min_len = max(25, int(0.08 * np.sqrt(roi_area)))

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=100,
            minLineLength=min_len,
            maxLineGap=6
        )

        drip_score = 0
        line_img = np.zeros_like(gray)

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # FIX: stricter verticality ratio (3x instead of 2x). Real
                # paint drips run essentially straight down; the looser
                # ratio was letting ordinary panel seams / reflections
                # count as "drips", which is what produced random-looking
                # drip lines every frame.
                if dy > dx * 3 and dy >= min_len:
                    drip_score += 1
                    cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

        drip_score = float(drip_score) / float(roi_area)
        drip_score = drip_score * 1000.0

        # FIX: smooth the reported score across frames (EMA) so one noisy
        # frame doesn't flip GOOD/BAD status. The visualization still
        # draws whatever lines were found in the current frame.
        self._drip_score_ema = (self._drip_ema_alpha * drip_score
                                 + (1 - self._drip_ema_alpha) * self._drip_score_ema)

        line_img = cv2.bitwise_and(line_img, line_img, mask=mask)

        return self._drip_score_ema, line_img

    # -----------------------------
    # 6. ORANGE PEEL (MULTI-SCALE TEXTURE)
    # -----------------------------
    def detect_orange_peel(self, gray, mask):
        roi = cv2.bitwise_and(gray, gray, mask=mask)
        roi_pixels = roi[mask > 0]

        if roi_pixels.size == 0:
            return 0.0, np.zeros_like(gray, dtype=np.float64)

        lap1 = cv2.Laplacian(roi, cv2.CV_64F)
        lap2 = cv2.Laplacian(cv2.GaussianBlur(roi, (5, 5), 0), cv2.CV_64F)

        lap1_abs = np.abs(lap1)
        lap2_abs = np.abs(lap2)
        tex_val = float(np.mean(lap1_abs[mask > 0]))

        blur = cv2.GaussianBlur(roi, (15, 15), 0)
        high_freq = cv2.subtract(roi.astype(np.float32), blur.astype(np.float32))
        hf_mag = np.abs(high_freq)
        hf_val = float(np.mean(hf_mag[mask > 0]))

        roi_f = roi.astype(np.float32)
        mean_f = cv2.blur(roi_f, (9, 9))
        mean_sq = cv2.blur(roi_f * roi_f, (9, 9))
        std_map = np.sqrt(np.clip(mean_sq - mean_f * mean_f, 0, None))
        std_val = float(np.mean(std_map[mask > 0]))

        def safe_pct95(arr):
            vals = arr[mask > 0]
            if vals.size == 0:
                return 1.0
            p = np.percentile(vals, 95)
            return max(float(p), 1e-6)

        tex_den = safe_pct95(lap1_abs)
        hf_den  = safe_pct95(hf_mag)
        std_den = safe_pct95(std_map)

        tex_norm = tex_val / tex_den
        hf_norm  = hf_val  / hf_den
        std_norm = std_val / std_den

        w_tex, w_hf, w_std = 1.0, 0.8, 1.0

        combined_norm = w_tex * tex_norm + w_hf * hf_norm + w_std * std_norm

        score = float(combined_norm) / 3.0 * 100.0
        score = float(np.clip(score, 0.0, 1000.0))

        rough_map = std_map.astype(np.float64)
        rough_map[mask == 0] = 0.0

        return score, rough_map

    # -----------------------------
    # 7. FULL PIPELINE
    # -----------------------------
    def inspect(self, img, roi_mask=None):
        img, hsv, gray, edge_gray = self.preprocess(img)

        if roi_mask is None or np.count_nonzero(roi_mask) == 0:
            mask = self.get_roi_mask(gray)
        else:
            mask = roi_mask.astype(np.uint8)
            if mask.shape != gray.shape:
                mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255

        # Method 1: binary paint coverage grid
        grid_coverage, grid_pct, paint_mask, grid_vis = self.detect_binary_coverage(hsv, mask)
        defect_matrix = self.build_defect_matrix(grid_pct)

        save_dir = "/home/user/car_spraying_ws_6_7/car_spraying_ws/src/ob_detection/ob_detection/spray_paths"
        os.makedirs(save_dir, exist_ok=True)

        np.save(
            os.path.join(save_dir, "latest_defect_matrix.npy"),
            defect_matrix
        )

        # Method 2 + extras: uneven coverage, drips, roughness
        coverage_score, cov_map = self.detect_coverage(hsv, mask)
        drip_score,     drip_map = self.detect_drips(edge_gray, mask)
        rough_score,    rough_map = self.detect_orange_peel(gray, mask)

        result = {
            "grid_coverage":        float(grid_coverage),
            "grid_coverage_status": "GOOD" if grid_coverage >= self.grid_coverage_thresh * 100.0 else "BAD",
            "coverage_score":   float(coverage_score),
            "coverage_status":  "BAD" if coverage_score > self.coverage_thresh else "GOOD",
            "drip_score":   float(drip_score),
            "drip_status":  "BAD" if drip_score > self.drip_thresh else "GOOD",
            "roughness_score": float(rough_score),
            "rough_status":    "BAD" if rough_score > self.rough_thresh else "GOOD",
            "defect_matrix": defect_matrix.tolist(),
        }

        return result, grid_vis, paint_mask, cov_map, drip_map, rough_map, mask, defect_matrix


# ============================================================
# Helper: build a single display panel (BGR, target_size px square)
# ============================================================
def _make_panel(img, target_w, target_h):
    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (target_w, target_h))


class ROSCameraReader(Node):
    def __init__(self, camera_topic='/color_image/compressed'):
        super().__init__('defect_detection_ros_camera')
        self.camera_topic = camera_topic
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        camera_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(CompressedImage, camera_topic,
                                 self._camera_callback, camera_qos)

        self.defect_pub = self.create_publisher(
            Float32MultiArray,
            '/spray/defect_matrix',
            10
        )

    def _camera_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        with self._frame_lock:
            self._latest_frame = frame

    def get_latest_frame(self):
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()


def load_part_model(model_path):
    if not os.path.exists(model_path):
        return None, None
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, device


def build_part_mask_from_detections(detections, frame_shape):
    h, w = frame_shape[:2]
    part_mask = np.zeros((h, w), dtype=np.uint8)

    if not detections:
        return part_mask

    for det in detections:
        mask = det.get('mask')
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.squeeze(0)
            if mask_arr.shape != (h, w):
                try:
                    mask_arr = cv2.resize(mask_arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_arr = np.zeros((h, w), dtype=np.uint8)
            mask_arr = (mask_arr > 0).astype(np.uint8) * 255
            part_mask = cv2.bitwise_or(part_mask, mask_arr)
            continue

        box = det.get('box')
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(part_mask, (x1, y1), (x2, y2), 255, -1)

    return part_mask


def detect_part_labels(model, frame, conf_thresh, iou_thresh, device):
    detections = []
    if model is None:
        return detections

    results = model.predict(
        source=frame, conf=conf_thresh, iou=iou_thresh,
        imgsz=640, verbose=False, device=device, retina_masks=True)
    if not results or results[0].boxes is None:
        return detections

    img_h, img_w = frame.shape[:2]
    masks = None
    if getattr(results[0], 'masks', None) is not None:
        masks = results[0].masks.data.cpu().numpy()

    for idx, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        label = model.names.get(cls, str(cls)) if hasattr(model, 'names') else str(cls)
        conf = float(box.conf[0])
        xyxy = [int(v) for v in box.xyxy[0].tolist()]

        mask_bin = None
        mask_area = 0
        if masks is not None and idx < len(masks):
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            mask_bin = (mask > 0.5).astype(np.uint8)
            if mask_bin.shape != (img_h, img_w):
                mask_bin = cv2.resize(mask_bin, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_area = int(np.count_nonzero(mask_bin))
            mask_bin = (mask_bin * 255).astype(np.uint8)

        if mask_area == 0:
            x1, y1, x2, y2 = xyxy
            mask_area = max(0, x2 - x1) * max(0, y2 - y1)

        detections.append({'label': label, 'conf': conf, 'box': xyxy, 'mask': mask_bin, 'mask_area': mask_area})

    return detections


# ============================================================
# Demo / main loop
# ============================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Spray quality inspector with optional ROS iPhone camera and part detection')
    parser.add_argument('--ros', dest='ros', action='store_true', default=ROS_AVAILABLE,
                        help='Read camera frames from ROS topic (default when ROS is available)')
    parser.add_argument('--no-ros', dest='ros', action='store_false',
                        help='Use local laptop camera instead of ROS')
    parser.add_argument('--camera_topic', default='/color_image/compressed',
                        help='ROS CompressedImage camera topic')
    parser.add_argument('--camera_index', type=int, default=0,
                        help='OpenCV camera index when not using ROS')
    parser.add_argument('--model_path', default=None,
                        help='YOLO model path for part identification')
    parser.add_argument('--confidence', type=float, default=0.35,
                        help='YOLO detection confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                        help='YOLO NMS IoU threshold')
    parser.add_argument('--no_yolo', action='store_true',
                        help='Disable YOLO part detection overlay')
    args = parser.parse_args(argv)

    inspector = SprayQualityInspectorV2()

    model_path = args.model_path
    if model_path is None:
        model_path = '/home/user/car_spraying_ws/src/ob_detection/ob_detection/car_parts_best_seg.pt'
        if not os.path.exists(model_path):
            model_path = '/home/user/car_spraying_ws/src/ob_detection/ob_detection/car_parts_best.pt'

    yolo_model = None
    yolo_device = None
    if not args.no_yolo:
        yolo_model, yolo_device = load_part_model(model_path)
        if yolo_model is None:
            print(f'WARNING: YOLO model not found at {model_path}. Part labels disabled.')

    ros_reader = None
    ros_thread = None
    cap = None

    if args.ros:
        if not ROS_AVAILABLE:
            raise RuntimeError('ROS 2 is not available in this environment. Install rclpy and sensor_msgs.')
        rclpy.init()
        ros_reader = ROSCameraReader(args.camera_topic)
        ros_thread = threading.Thread(target=rclpy.spin, args=(ros_reader,), daemon=True)
        ros_thread.start()
        print(f'Listening to ROS camera topic: {args.camera_topic}')
    else:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera (index {args.camera_index}).')

    print('Controls:')
    print('  z/x  – coverage uniformity threshold  down/up')
    print('  a/s  – drip threshold  down/up')
    print('  k/l  – roughness threshold  down/up')
    print('  [/]  – grid coverage threshold  down/up')
    print('  c    – print HSV of centre pixel (use to calibrate paint colour)')
    print('  q / ESC – quit')

    try:
        while True:
            if args.ros:
                frame = ros_reader.get_latest_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print('Failed to grab frame from camera, retrying...')
                    time.sleep(0.1)
                    continue

            try:
                cam_display = cv2.resize(frame, (640, 640))

                part_label = 'Part: disabled'
                detections = []
                part_mask = np.zeros(cam_display.shape[:2], dtype=np.uint8)

                if yolo_model is not None:
                    detections = detect_part_labels(
                        yolo_model,
                        cam_display,
                        args.confidence,
                        args.iou_threshold,
                        yolo_device)
                    if detections:
                        top = max(detections, key=lambda d: (d['conf'], d.get('mask_area', 0)))
                        part_mask = build_part_mask_from_detections([top], cam_display.shape[:2])
                        part_label = f"Part: {top['label']} ({top['conf']:.2f})"
                    else:
                        part_label = 'Part: unknown'

                focus_mask = part_mask if np.count_nonzero(part_mask) > 0 else None
                result, grid_vis, paint_mask, cov, drip, rough, mask, defect_matrix = inspector.inspect(cam_display, roi_mask=focus_mask)

                if ros_reader is not None:
                    msg = Float32MultiArray()
                    msg.data = defect_matrix.astype(np.float32).flatten().tolist()
                    ros_reader.defect_pub.publish(msg)

                part_mask = part_mask.astype(np.uint8)
                mask = mask.astype(np.uint8)
                paint_mask = paint_mask.astype(np.uint8)
                part_roi_mask = cv2.bitwise_and(mask, part_mask) if np.count_nonzero(part_mask) > 0 else np.zeros_like(mask)

                paint_in_roi = cv2.bitwise_and(paint_mask, paint_mask, mask=part_roi_mask)
                miss_mask = cv2.bitwise_and(cv2.bitwise_not(paint_in_roi), part_roi_mask)
                miss_overlay = np.zeros_like(cam_display)
                miss_overlay[:, :, 2] = miss_mask
                cam_display = cv2.addWeighted(cam_display, 0.7, miss_overlay, 0.3, 0)

                part_roi_mask_bool = part_roi_mask > 0
                uneven_part = np.zeros_like(cov)
                uneven_part[part_roi_mask_bool] = cov[part_roi_mask_bool]
                uneven_norm = cv2.normalize(uneven_part, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                uneven_mask = cv2.threshold(uneven_norm, 110, 255, cv2.THRESH_BINARY)[1]

                disp_drip = np.zeros_like(drip, dtype=np.uint8)
                disp_drip[part_roi_mask_bool] = drip[part_roi_mask_bool]
                drip_mask = cv2.threshold(disp_drip, 20, 255, cv2.THRESH_BINARY)[1]

                if rough is None:
                    disp_rough = np.zeros((640, 640), dtype=np.uint8)
                else:
                    rough_part = np.zeros_like(rough)
                    rough_part[part_roi_mask_bool] = np.abs(rough[part_roi_mask_bool])
                    disp_rough = cv2.normalize(rough_part, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                rough_mask = cv2.threshold(disp_rough, 120, 255, cv2.THRESH_BINARY)[1]

                if detections:
                    top = max(detections, key=lambda d: d['conf'])
                    x1, y1, x2, y2 = top['box']
                    cv2.rectangle(cam_display, (x1, y1), (x2, y2), (20, 200, 20), 2)
                    cv2.putText(cam_display, f"{top['label']} {top['conf']:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (20, 220, 20), 2, cv2.LINE_AA)

                uneven_color = cv2.applyColorMap(uneven_norm, cv2.COLORMAP_JET)
                drip_color = cv2.cvtColor(disp_drip, cv2.COLOR_GRAY2BGR)
                drip_color[drip_mask > 0] = (0, 0, 255)
                rough_color = cv2.applyColorMap(disp_rough, cv2.COLORMAP_HOT)

                top_row = np.hstack([cam_display, uneven_color])
                bottom_row = np.hstack([drip_color, rough_color])
                composite_disp = np.vstack([top_row, bottom_row])

                max_w, max_h = 1280, 720
                ch, cw = composite_disp.shape[:2]
                scale = min(1.0, float(max_w) / float(cw), float(max_h) / float(ch))
                if scale < 1.0:
                    composite_disp = cv2.resize(composite_disp, (int(cw * scale), int(ch * scale)))

                dh, dw = composite_disp.shape[:2]
                pw = dw // 2
                ph = dh // 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = max(0.5, 0.8 * scale)
                tk = max(1, int(round(2 * scale)))
                line_dy = int(24 * fs * 1.2)

                def put(img, text, x, y, color=(255, 255, 255)):
                    cv2.putText(img, text, (x, y), font, fs, color, tk)

                put(composite_disp, 'Camera (defects circled)', 8, int(22 * fs))
                put(composite_disp, 'Uneven Coverage', pw + 8, int(22 * fs))
                put(composite_disp, 'Drip Lines', 8, ph + int(22 * fs))
                put(composite_disp, 'Orange Peel', pw + 8, ph + int(22 * fs))

                sx = 8
                sy = ph + int(50 * fs)
                def status_color(s):
                    return (0, 220, 0) if s == 'GOOD' else (0, 0, 220)

                put(composite_disp,
                    f'Grid Cov: {result["grid_coverage_status"]} {result["grid_coverage"]:.1f}%',
                    sx, sy, status_color(result['grid_coverage_status']))
                put(composite_disp,
                    f'Uneven:   {result["coverage_status"]} {result["coverage_score"]:.2f}',
                    sx, sy + line_dy, status_color(result['coverage_status']))
                put(composite_disp,
                    f'Drip:     {result["drip_status"]} {result["drip_score"]:.3f}',
                    sx, sy + 2 * line_dy, status_color(result['drip_status']))
                put(composite_disp,
                    f'Rough:    {result["rough_status"]} {result["roughness_score"]:.2f}',
                    sx, sy + 3 * line_dy, status_color(result['rough_status']))
                put(composite_disp,
                    part_label,
                    sx, sy + 4 * line_dy, (255, 255, 0))

                hx, hy = sx, sy + 5 * line_dy
                gray_c = (180, 180, 180)
                put(composite_disp, f'T(Uneven): {inspector.coverage_thresh:.2f}  (z/x)', hx, hy, gray_c)
                put(composite_disp, f'T(Drip):   {inspector.drip_thresh:.3f}  (a/s)', hx, hy + line_dy, gray_c)
                put(composite_disp, f'T(Rough):  {inspector.rough_thresh:.1f}  (k/l)', hx, hy + 2 * line_dy, gray_c)

                cv2.imshow('Spray Quality Inspection', composite_disp)

            except Exception as e:
                print(f'Error during processing/display: {e}')
                time.sleep(0.1)
                continue

            key = cv2.waitKey(1) & 0xFF

            if   key == ord('z'):  inspector.coverage_thresh      = max(0.0, inspector.coverage_thresh - 0.5)
            elif key == ord('x'):  inspector.coverage_thresh      += 0.5
            elif key == ord('a'):  inspector.drip_thresh          = max(0.0, inspector.drip_thresh - 0.05)
            elif key == ord('s'):  inspector.drip_thresh          += 0.05
            elif key == ord('k'):  inspector.rough_thresh         = max(0.0, inspector.rough_thresh - 1.0)
            elif key == ord('l'):  inspector.rough_thresh         += 1.0
            elif key == ord('['):  inspector.grid_coverage_thresh = max(0.0, inspector.grid_coverage_thresh - 0.05)
            elif key == ord(']'):  inspector.grid_coverage_thresh = min(1.0, inspector.grid_coverage_thresh + 0.05)

            elif key == ord('c'):
                frame_resized = cv2.resize(frame, (640, 640))
                hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
                cy, cx = 320, 320
                h_val, s_val, v_val = hsv_frame[cy, cx]
                print(f'Centre pixel HSV: H={h_val}  S={s_val}  V={v_val}')
                print(f'  Suggested lower: [{max(0,h_val-10)}, 50, 50]')
                print(f'  Suggested upper: [{min(179,h_val+10)}, 255, 255]')
                print('  Update PAINT_HSV_LOWER / PAINT_HSV_UPPER at the top of the file.')

            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if args.ros and ROS_AVAILABLE:
            try:
                ros_reader.destroy_node()
                rclpy.shutdown()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()