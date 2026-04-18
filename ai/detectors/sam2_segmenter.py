"""SAM2 video segmentation — tracks YOLO detections across frames with precise masks.

Takes bounding boxes from YOLO on the first frame, then uses SAM2 to propagate
segmentation masks across all subsequent frames. This gives pixel-accurate masks
for each detected person, improving PPE classification accuracy.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SAM2 loading — heavy dependency, only load when needed
# ---------------------------------------------------------------------------

_sam2_predictor: Any | None = None


def _get_sam2_predictor(device: str = "cpu") -> Any:
    """Return a cached SAM2 video predictor, loading on first call."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    model_id = os.getenv("SAM2_MODEL", "facebook/sam2.1-hiera-tiny")
    logger.info("Loading SAM2 model: %s (device=%s)", model_id, device)

    try:
        from sam2.build_sam import build_sam2_video_predictor
        _sam2_predictor = build_sam2_video_predictor(model_id, device=device)
        logger.info("SAM2 model loaded successfully")
    except ImportError as e:
        logger.error("SAM2 not installed: %s", e)
        raise RuntimeError(
            "SAM2 is not installed. Run: pip install -e path/to/sam2"
        ) from e

    return _sam2_predictor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_video(
    video_path: str,
    detections_per_frame: list[tuple[float, list[dict[str, Any]]]],
    device: str = "cpu",
) -> list[tuple[float, int, dict[int, np.ndarray]]]:
    """Track YOLO detections across frames using SAM2.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    detections_per_frame : list of (timestamp, detections)
        YOLO detections for each frame. Each detection dict must have
        'bbox' as [x1, y1, x2, y2].
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    list of (timestamp, frame_idx, {obj_id: mask})
        For each frame, a dict mapping object IDs to boolean masks (H, W).
    """
    if not detections_per_frame:
        logger.warning("No detections provided for SAM2 tracking")
        return []

    predictor = _get_sam2_predictor(device)

    try:
        inference_state = predictor.init_state(video_path)
    except Exception as e:
        logger.error("SAM2 init_state failed: %s", e)
        return []

    num_frames = inference_state["num_frames"]
    logger.info("SAM2 loaded video: %d frames", num_frames)

    # Use detections from the first frame as prompts
    first_timestamp, first_detections = detections_per_frame[0]
    obj_id_map: dict[int, dict[str, Any]] = {}

    for i, det in enumerate(first_detections):
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue

        try:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=i,
                box=[x1, y1, x2, y2],
            )
            obj_id_map[i] = det
            logger.debug("SAM2: added object %d bbox=[%.0f,%.0f,%.0f,%.0f]", i, x1, y1, x2, y2)
        except Exception as e:
            logger.warning("SAM2: failed to add object %d: %s", i, e)

    if not obj_id_map:
        logger.warning("No valid SAM2 prompts created")
        return []

    # Propagate masks across all frames
    results: list[tuple[float, int, dict[int, np.ndarray]]] = []

    try:
        for frame_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
            mask_dict: dict[int, np.ndarray] = {}
            for obj_id, mask in zip(obj_ids, masks):
                mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else np.array(mask)
                if mask_np.ndim > 2:
                    mask_np = mask_np.squeeze()
                mask_dict[obj_id] = mask_np.astype(bool)

            # Map frame index to timestamp (linear interpolation)
            if len(detections_per_frame) > 1:
                first_ts = detections_per_frame[0][0]
                last_ts = detections_per_frame[-1][0]
                if num_frames > 1:
                    timestamp = first_ts + (last_ts - first_ts) * frame_idx / (num_frames - 1)
                else:
                    timestamp = first_ts
            else:
                timestamp = first_timestamp

            results.append((timestamp, frame_idx, mask_dict))
    except Exception as e:
        logger.error("SAM2 propagation failed: %s", e)
        return []

    logger.info("SAM2: tracked %d objects across %d frames", len(obj_id_map), len(results))
    return results


def extract_mask_region(
    frame: np.ndarray,
    mask: np.ndarray,
    padding: int = 5,
) -> np.ndarray:
    """Extract the masked region from a frame as a cropped image.

    Useful for running PPE classification only on the segmented person,
    ignoring background noise.
    """
    if mask.sum() == 0:
        return frame  # fallback to full frame

    # Find bounding box of the mask
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # Add padding
    h, w = frame.shape[:2]
    y1 = max(0, y1 - padding)
    x1 = max(0, x1 - padding)
    y2 = min(h, y2 + padding)
    x2 = min(w, x2 + padding)

    # Apply mask and crop
    masked_frame = frame.copy()
    masked_frame[~mask] = 0  # zero out non-mask pixels
    cropped = masked_frame[y1:y2, x1:x2]

    return cropped


def mask_coverage_ratio(mask: np.ndarray) -> float:
    """Return the fraction of the frame covered by the mask (0-1)."""
    return float(mask.sum()) / (mask.size + 1e-6)
