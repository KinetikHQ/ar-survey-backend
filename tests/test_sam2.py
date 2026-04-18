"""Tests for SAM2 segmenter and pipeline integration."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# mask_coverage_ratio (no cv2 dependency)
# ---------------------------------------------------------------------------

def test_mask_coverage_ratio_full():
    from ai.detectors.sam2_segmenter import mask_coverage_ratio
    mask = np.ones((100, 100), dtype=bool)
    assert mask_coverage_ratio(mask) == pytest.approx(1.0)


def test_mask_coverage_ratio_empty():
    from ai.detectors.sam2_segmenter import mask_coverage_ratio
    mask = np.zeros((100, 100), dtype=bool)
    assert mask_coverage_ratio(mask) == pytest.approx(0.0, abs=1e-6)


def test_mask_coverage_ratio_quarter():
    from ai.detectors.sam2_segmenter import mask_coverage_ratio
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :50] = True  # top-left quarter
    assert mask_coverage_ratio(mask) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# extract_mask_region (needs cv2)
# ---------------------------------------------------------------------------

cv2 = pytest.importorskip("cv2")


def test_extract_mask_region_crops_to_mask():
    from ai.detectors.sam2_segmenter import extract_mask_region
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:60, 30:70] = 255

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True

    result = extract_mask_region(frame, mask)
    assert result.shape[0] <= 50
    assert result.shape[1] <= 50


def test_extract_mask_region_empty_mask_returns_frame():
    from ai.detectors.sam2_segmenter import extract_mask_region
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    mask = np.zeros((50, 50), dtype=bool)

    result = extract_mask_region(frame, mask)
    assert result.shape == frame.shape


def test_extract_mask_region_zeros_out_background():
    from ai.detectors.sam2_segmenter import extract_mask_region
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 200

    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True

    result = extract_mask_region(frame, mask)
    assert result[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# Pipeline integration (mocked SAM2)
# ---------------------------------------------------------------------------

def test_pipeline_falls_back_when_sam2_unavailable():
    """Pipeline should gracefully fall back when SAM2 import fails."""
    import sys
    with patch.dict(sys.modules, {"sam2": None, "sam2.build_sam": None}):
        import importlib
        import ai.pipeline
        importlib.reload(ai.pipeline)
        assert not ai.pipeline._SAM2_AVAILABLE


def test_pipeline_per_frame_mode():
    """Per-frame pipeline should work without SAM2."""
    from ai.pipeline import _pipeline_per_frame

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [(0.0, frame), (1.0, frame)]

    with patch("ai.detectors.ppe_detector.detect_ppe", return_value=[]):
        result = _pipeline_per_frame(frames, device="cpu")
        assert result == []
