import importlib.util
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "ob_detection" / "Defect_detection_with_Coverage_Map.py"
SPEC = importlib.util.spec_from_file_location("defect_detector", MODULE_PATH)
defect_detector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(defect_detector)


def test_build_part_mask_prefers_segmentation_masks():
    frame_shape = (64, 64)
    mask = np.zeros(frame_shape, dtype=np.uint8)
    mask[10:30, 10:30] = 255

    detections = [{
        "label": "door",
        "conf": 0.99,
        "box": [0, 0, 63, 63],
        "mask": mask,
    }]

    part_mask = defect_detector.build_part_mask_from_detections(detections, frame_shape)

    assert part_mask.shape == frame_shape
    assert np.count_nonzero(part_mask) == 400
    assert part_mask[15, 15] == 255
    assert part_mask[0, 0] == 0


def test_detect_binary_coverage_adapts_to_non_blue_paint():
    inspector = defect_detector.SprayQualityInspectorV2()

    # pink/orange paint region in BGR
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:] = (180, 105, 255)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_mask = np.zeros((64, 64), dtype=np.uint8)
    roi_mask[8:56, 8:56] = 255

    overall, grid_pct, paint_mask, grid_vis = inspector.detect_binary_coverage(hsv, roi_mask)

    assert paint_mask.shape == roi_mask.shape
    assert np.count_nonzero(paint_mask) > 0
    assert np.count_nonzero(paint_mask) <= np.count_nonzero(roi_mask)
    assert overall >= 0.0


def test_inspect_uses_provided_roi_mask():
    inspector = defect_detector.SprayQualityInspectorV2()

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[8:56, 8:56] = (180, 100, 255)

    override_mask = np.zeros((64, 64), dtype=np.uint8)
    override_mask[16:48, 16:48] = 255

    result, grid_vis, paint_mask, cov, drip, rough, mask = inspector.inspect(frame, roi_mask=override_mask)

    expected_mask = cv2.resize(override_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(mask, expected_mask)
    assert paint_mask.shape == mask.shape
    assert grid_vis.shape[:2] == mask.shape
    assert result['grid_coverage'] >= 0.0
