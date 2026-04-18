"""AI detection pipeline — orchestrates frame extraction, YOLO detection, SAM2 tracking, and PPE classification."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import numpy as np

from ai.detectors.frame_extractor import extract_frames
from ai.detectors.ppe_detector import detect_ppe

logger = logging.getLogger(__name__)

# SAM2 is optional — fall back to per-frame detection if unavailable
_SAM2_AVAILABLE = False
try:
    from ai.detectors.sam2_segmenter import segment_video, extract_mask_region
    _SAM2_AVAILABLE = True
except ImportError:
    pass


def process_clip(video_path: str) -> list[dict[str, Any]]:
    """Process a video file and return deduplicated PPE detections.

    Pipeline modes:
    - With SAM2: YOLO on first frame → SAM2 tracking → PPE on masked regions
    - Without SAM2: YOLO + PPE on each frame independently (legacy)

    Parameters
    ----------
    video_path : str
        Absolute path to the downloaded MP4 file.

    Returns
    -------
    list[dict]
        Each dict matches the Result schema:
        - category: str (\"ppe\", \"segmentation\")
        - label: str
        - confidence: float
        - bbox: list[int]  [x1, y1, x2, y2]
        - frame_timestamp: float
        - metadata: dict | None  (includes mask_polygon when SAM2 is used)
    """
    device = os.getenv("MODEL_DEVICE", "cpu")
    sample_rate = int(os.getenv("FRAME_SAMPLE_RATE", "1"))
    use_sam2 = os.getenv("USE_SAM2", "true").lower() == "true"

    # 1. Extract frames
    frames = extract_frames(video_path, sample_rate=sample_rate)
    logger.info("Extracted %d frames from %s", len(frames), video_path)

    if not frames:
        logger.warning("No frames extracted from %s", video_path)
        return []

    # 2. Choose pipeline mode
    if use_sam2 and _SAM2_AVAILABLE:
        return _pipeline_with_sam2(frames, video_path, device)
    else:
        if use_sam2 and not _SAM2_AVAILABLE:
            logger.warning("USE_SAM2=true but SAM2 not installed — falling back to per-frame detection")
        return _pipeline_per_frame(frames, device)


def _pipeline_with_sam2(
    frames: list[tuple[float, Any]],
    video_path: str,
    device: str,
) -> list[dict[str, Any]]:
    """SAM2 pipeline: detect once, track across frames with precise masks."""
    from ai.detectors.ppe_detector import _get_model, Detection

    # Step 1: Run YOLO on first frame to get initial detections
    yolo_model = _get_model(device)
    first_timestamp, first_frame = frames[0]

    yolo_results = yolo_model.predict(
        source=first_frame,
        classes=[0],  # person class only
        conf=0.35,
        verbose=False,
    )

    # Collect YOLO detections from first frame
    first_detections = []
    if yolo_results and len(yolo_results) > 0:
        result = yolo_results[0]
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                fh, fw = first_frame.shape[:2]
                first_detections.append({
                    "bbox": [max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)],
                    "confidence": conf,
                })

    logger.info("YOLO: %d persons detected on first frame", len(first_detections))

    if not first_detections:
        logger.info("No persons detected — returning empty results")
        return []

    # Step 2: Track across frames with SAM2
    # Build detections_per_frame — use first frame detections for all
    detections_per_frame = [(ts, first_detections) for ts, _ in frames]

    sam2_results = segment_video(
        video_path=video_path,
        detections_per_frame=detections_per_frame,
        device=device,
    )

    if not sam2_results:
        logger.warning("SAM2 tracking failed — falling back to per-frame detection")
        return _pipeline_per_frame(frames, device)

    # Step 3: For each tracked object on each frame, classify PPE on masked region
    all_detections: list[dict[str, Any]] = []

    for timestamp, frame_idx, mask_dict in sam2_results:
        if frame_idx >= len(frames):
            continue

        _, frame = frames[frame_idx]

        for obj_id, mask in mask_dict.items():
            if mask.sum() == 0:
                continue

            # Get bbox from mask
            ys, xs = np.where(mask)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

            # Extract masked region for PPE classification
            masked_region = extract_mask_region(frame, mask)
            person_conf = first_detections[obj_id]["confidence"] if obj_id < len(first_detections) else 0.8

            # Classify hard hat
            from ai.detectors.ppe_detector import _classify_hard_hat, _classify_hi_vis
            hat_label, hat_conf = _classify_hard_hat(masked_region)
            all_detections.append({
                "category": "ppe",
                "label": hat_label,
                "confidence": round(hat_conf * person_conf, 4),
                "bbox": [x1, y1, x2, y2],
                "frame_timestamp": timestamp,
                "metadata": {
                    "description": (
                        "Person wearing hard hat correctly"
                        if hat_label == "hard_hat_present"
                        else "Person detected without hard hat"
                    ),
                    "tracked_object_id": obj_id,
                    "segmented": True,
                    "mask_area_pixels": int(mask.sum()),
                },
            })

            # Classify hi-vis
            vis_label, vis_conf = _classify_hi_vis(masked_region)
            all_detections.append({
                "category": "ppe",
                "label": vis_label,
                "confidence": round(vis_conf * person_conf, 4),
                "bbox": [x1, y1, x2, y2],
                "frame_timestamp": timestamp,
                "metadata": {
                    "description": (
                        "Person wearing hi-vis vest"
                        if vis_label == "hi_vis_present"
                        else "Person detected without hi-vis vest"
                    ),
                    "tracked_object_id": obj_id,
                    "segmented": True,
                    "mask_area_pixels": int(mask.sum()),
                },
            })

    # 4. Deduplicate
    deduplicated = _deduplicate_detections(all_detections)

    # 5. Assign unique IDs
    for det in deduplicated:
        det["id"] = str(uuid.uuid4())

    logger.info(
        "SAM2 pipeline complete: %d raw detections -> %d after dedup",
        len(all_detections),
        len(deduplicated),
    )
    return deduplicated


def _pipeline_per_frame(
    frames: list[tuple[float, Any]],
    device: str,
) -> list[dict[str, Any]]:
    """Legacy per-frame pipeline (no SAM2 tracking)."""
    all_detections: list[dict[str, Any]] = []
    for timestamp, frame in frames:
        detections = detect_ppe(frame, timestamp, device=device)
        all_detections.extend(detections)

    deduplicated = _deduplicate_detections(all_detections)

    for det in deduplicated:
        det["id"] = str(uuid.uuid4())

    logger.info(
        "Per-frame pipeline complete: %d raw detections -> %d after dedup",
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
