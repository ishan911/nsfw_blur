"""Tests for _apply_yolo_detection helper (bugs 1, 2, 4)."""

import os
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_yolo_result(boxes_xyxy, confidences, class_ids, names):
    """Build a minimal ultralytics-shaped mock result object."""
    mock_result = MagicMock()
    mock_result.names = names

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=len(boxes_xyxy))

    mock_box_list = []
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
        b = MagicMock()
        b.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]
        b.conf = [torch.tensor(conf, dtype=torch.float32)]
        b.cls = [torch.tensor(float(cls_id), dtype=torch.float32)]
        mock_box_list.append(b)

    mock_boxes.__iter__ = MagicMock(return_value=iter(mock_box_list))
    mock_result.boxes = mock_boxes
    return mock_result


# ---------------------------------------------------------------------------
# Bug 1 — yolo_model is None must not raise
# ---------------------------------------------------------------------------

def test_none_model_returns_empty_list(solid_image_path, output_path):
    """Bug 1: passing yolo_model=None must return [] without raising."""
    import shutil
    shutil.copy2(solid_image_path, output_path)

    from main import _apply_yolo_detection
    result = _apply_yolo_detection(solid_image_path, output_path, yolo_model=None)

    assert result == []


# ---------------------------------------------------------------------------
# Bug 2 — YOLO must detect on input_path, not output_path
# ---------------------------------------------------------------------------

def test_yolo_detects_on_input_path_not_output(solid_image_path, output_path):
    """Bug 2: YOLO model must be called with input_path (original), not output_path."""
    import shutil
    shutil.copy2(solid_image_path, output_path)

    mock_boxes = MagicMock()
    mock_boxes.__len__ = MagicMock(return_value=0)
    mock_boxes.__iter__ = MagicMock(return_value=iter([]))
    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model = MagicMock(return_value=[mock_result])

    from main import _apply_yolo_detection
    _apply_yolo_detection(solid_image_path, output_path, yolo_model=mock_model)

    call_args = mock_model.call_args
    assert call_args is not None
    called_path = call_args[0][0]
    assert called_path == solid_image_path, (
        f"YOLO was called with '{called_path}' but expected input_path '{solid_image_path}'"
    )


# ---------------------------------------------------------------------------
# Bug 4 — Actual class name must be stored, not 'yolo_class_N'
# ---------------------------------------------------------------------------

def test_class_name_is_preserved_not_generic(solid_image_path, output_path):
    """Bug 4: detection dict 'class' must be the real class name, not 'yolo_class_N'."""
    import shutil
    shutil.copy2(solid_image_path, output_path)

    names = {0: "FEMALE_BREAST_EXPOSED", 1: "MALE_GENITALIA_EXPOSED"}
    mock_result = _make_mock_yolo_result(
        boxes_xyxy=[(10, 10, 50, 50)],
        confidences=[0.85],
        class_ids=[0],
        names=names,
    )
    mock_model = MagicMock(return_value=[mock_result])

    from main import _apply_yolo_detection
    detections = _apply_yolo_detection(solid_image_path, output_path, yolo_model=mock_model)

    assert len(detections) == 1
    assert detections[0]["class"] == "FEMALE_BREAST_EXPOSED", (
        f"Expected real class name but got '{detections[0]['class']}'"
    )
    assert not detections[0]["class"].startswith("yolo_class_")


# ---------------------------------------------------------------------------
# Guard — output_path missing must not crash
# ---------------------------------------------------------------------------

def test_missing_output_path_does_not_crash(solid_image_path, tmp_path):
    """If output_path does not exist, the helper must not raise — returns detections."""
    names = {0: "FEMALE_BREAST_EXPOSED"}
    mock_result = _make_mock_yolo_result(
        boxes_xyxy=[(10, 10, 50, 50)],
        confidences=[0.85],
        class_ids=[0],
        names=names,
    )
    mock_model = MagicMock(return_value=[mock_result])

    missing_output = str(tmp_path / "does_not_exist.jpg")

    from main import _apply_yolo_detection
    # Must not raise even when output_path is missing
    result = _apply_yolo_detection(solid_image_path, missing_output, yolo_model=mock_model)
    # Returns the detections (blur was skipped, but detection list is intact)
    assert isinstance(result, list)
