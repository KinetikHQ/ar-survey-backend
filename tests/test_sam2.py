"""Tests for SAM2 segmenter and pipeline integration."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai.detectors.sam2_segmenter import extract_mask_region, mask_coverage_ratio


# ---------------------------------------------------------------------------
# extract_mask_region
# ---------------------------------------------------------------------------

def test_extract_mask_region_crops_to_mask():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:60, 30:70] = 255  # white square

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True

    result = extract_mask_region(frame, mask)
    # Should crop to roughly the mask bounds (with 5px padding)
    assert result.shape[0] <= 50  # 60-20 + padding
    assert result.shape[1] <= 50  # 70-30 + padding


def test_extract_mask_region_empty_mask_returns_frame():
    frame = np.ones((50, 50, 3), dtype=np.uint8) * 128
    mask = np.zeros((50, 50), dtype=bool)

    result = extract_mask_region(frame, mask)
    assert result.shape == frame.shape


def test_extract_mask_region_zeros_out_background():
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 200

    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True

    result = extract_mask_region(frame, mask)
    # Outside the mask should be zeroed (within the crop region)
    # The crop is roughly [35:65, 35:65] with padding
    # So corners of the crop should be 0
    assert result[0, 0, 0] == 0  # top-left corner of crop (outside mask)


# ---------------------------------------------------------------------------
# mask_coverage_ratio
# ---------------------------------------------------------------------------

def test_mask_coverage_ratio_full():
    mask = np.ones((100, 100), dtype=bool)
    assert mask_coverage_ratio(mask) == pytest.approx(1.0)


def test_mask_coverage_ratio_empty():
    mask = np.zeros((100, 100), dtype=bool)
    assert mask_coverage_ratio(mask) == pytest.approx(0.0, abs=1e-6)


def test_mask_coverage_ratio_quarter():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:50, :50] = True  # top-left quarter
    assert mask_coverage_ratio(mask) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Pipeline integration (mocked SAM2)
# ---------------------------------------------------------------------------

def test_pipeline_falls_back_when_sam2_unavailable():
    """Pipeline should gracefully fall back when SAM2 import fails."""
    with patch.dict("sys.modules", {"sam2": None, "sam2.build_sam": None}):
        # Re-import to trigger the import check
        import importlib
        import ai.pipeline
        importlib.reload(ai.pipeline)

        # SAM2 should not be available
        assert not ai.pipeline._SAM2_AVAILABLE


def test_pipeline_per_frame_mode():
    """Per-frame pipeline should work without SAM2."""
    from ai.pipeline import _pipeline_per_frame

    # Create a dummy frame (black image, no persons)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [(0.0, frame), (1.0, frame)]

    with patch("ai.detectors.ppe_detector.detect_ppe", return_value=[]):
        result = _pipeline_per_frame(frames, device="cpu")
        assert result == []
