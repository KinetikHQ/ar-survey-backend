"""AI detection pipeline — orchestrates frame extraction and PPE detection."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from ai.detectors.frame_extractor import extract_frames
from ai.detectors.ppe_detector import detect_ppe

logger = logging.getLogger(__name__)


def process_clip(video_path: str) -> list[dict[str, Any]]:
    """Process a video file and return deduplicated PPE detections.

    Parameters
    ----------
    video_path : str
        Absolute path to the downloaded MP4 file.

    Returns
    -------
    list[dict]
        Each dict matches the Result schema:
        - category: str ("ppe")
        - label: str
        - confidence: float
        - bbox: list[int]  [x1, y1, x2, y2]
        - frame_timestamp: float
        - metadata: dict | None
    """
    device = os.getenv("MODEL_DEVICE", "cpu")
    sample_rate = int(os.getenv("FRAME_SAMPLE_RATE", "1"))

    # 1. Extract frames
    frames = extract_frames(video_path, sample_rate=sample_rate)
    logger.info("Extracted %d frames from %s", len(frames), video_path)

    if not frames:
        logger.warning("No frames extracted from %s", video_path)
        return []

    # 2. Run PPE detection on each frame
    all_detections: list[dict[str, Any]] = []
    for timestamp, frame in frames:
        detections = detect_ppe(frame, timestamp, device=device)
        all_detections.extend(detections)

    # 3. Deduplicate — same person across consecutive frames
    deduplicated = _deduplicate_detections(all_detections)

    # 4. Assign unique IDs
    for det in deduplicated:
        det["id"] = str(uuid.uuid4())

    logger.info(
        "Pipeline complete: %d raw detections -> %d after dedup",
        len(all_detections),
        len(deduplicated),
    )
    return deduplicated


def _deduplicate_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate detections across frames using IoU + time window.

    If the same label appears in consecutive frames with overlapping bboxes,
    keep only the earliest detection.
    """
    if not detections:
        return []

    # Sort by timestamp
    sorted_dets = sorted(detections, key=lambda d: d["frame_timestamp"])

    kept: list[dict[str, Any]] = []
    for det in sorted_dets:
        is_duplicate = False
        for existing in kept:
            # Same label?
            if det["label"] != existing["label"]:
                continue
            # Within 2 second window?
            if abs(det["frame_timestamp"] - existing["frame_timestamp"]) > 2.0:
                continue
            # Overlapping bbox?
            if _iou(det["bbox"], existing["bbox"]) > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)

    return kept


def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / (union + 1e-6)
