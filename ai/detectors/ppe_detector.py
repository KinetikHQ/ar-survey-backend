"""PPE compliance detection module using YOLOv8.

For MVP: detects persons via COCO-pretrained YOLOv8, then classifies each
person as compliant or violating based on heuristic colour/region checks
for hard hats (bright dome in upper-body region) and hi-vis vests
(bright fluorescent colour band in torso region).

This is a fast heuristic MVP — accuracy will improve with fine-tuned models.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model loading — import ultralytics only when first needed to keep
# startup fast and avoid crashing if the package isn't installed yet.
# ---------------------------------------------------------------------------

_yolo_model: Any | None = None


def _get_model(device: str = "cpu") -> Any:
    """Return a cached YOLOv8-nano model, loading it on first call."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        logger.info("Loading YOLOv8 model: %s (device=%s)", model_name, device)
        _yolo_model = YOLO(model_name)
        _yolo_model.to(device)
    return _yolo_model


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single PPE detection result matching the API contract schema."""

    category: str = "ppe"
    label: str = ""
    confidence: float = 0.0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    frame_timestamp: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "frame_timestamp": self.frame_timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Heuristic PPE classifiers (MVP)
# ---------------------------------------------------------------------------

def _classify_hard_hat(person_roi: np.ndarray) -> tuple[str, float]:
    """Heuristic: check upper 25 % of the person ROI for bright helmet colours.

    Hard hats are typically white, yellow, orange, red — high-saturation or
    high-value pixels in the top band suggest a hard hat is present.

    Returns (label, confidence).
    """
    h, w = person_roi.shape[:2]
    if h == 0 or w == 0:
        return "missing_hard_hat", 0.5

    upper = person_roi[: max(1, h // 4), :]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)

    # Bright hard-hat colours: H in [0-30] or [90-130], S>50, V>150
    # (red/orange/yellow and white/bright surfaces)
    bright_mask = hsv[:, :, 2] > 150
    saturation_mask = hsv[:, :, 1] > 50
    combined = bright_mask & saturation_mask

    bright_ratio = combined.sum() / (combined.size + 1e-6)

    if bright_ratio > 0.15:
        return "hard_hat_present", min(0.95, 0.6 + bright_ratio)
    else:
        return "missing_hard_hat", min(0.90, 0.55 + (1 - bright_ratio) * 0.3)


def _classify_hi_vis(person_roi: np.ndarray) -> tuple[str, float]:
    """Heuristic: check middle 30-70 % (torso band) for fluorescent colours.

    Hi-vis vests are neon yellow/green/orange — high saturation and value
    in a wide band across the torso.

    Returns (label, confidence).
    """
    h, w = person_roi.shape[:2]
    if h == 0 or w == 0:
        return "missing_hi_vis", 0.5

    top = int(h * 0.3)
    bottom = int(h * 0.7)
    torso = person_roi[top:bottom, :]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Neon yellow-green: H 20-80, S > 100, V > 180
    # Neon orange: H 10-25, S > 150, V > 200
    mask_yellow = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 80) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 180)
    mask_orange = (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 200)
    combined = mask_yellow | mask_orange

    fluorescent_ratio = combined.sum() / (combined.size + 1e-6)

    if fluorescent_ratio > 0.10:
        return "hi_vis_present", min(0.95, 0.6 + fluorescent_ratio)
    else:
        return "missing_hi_vis", min(0.90, 0.55 + (1 - fluorescent_ratio) * 0.3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_ppe(
    frame: np.ndarray,
    timestamp: float,
    device: str = "cpu",
    confidence_threshold: float = 0.35,
) -> List[Dict[str, Any]]:
    """Run PPE detection on a single video frame.

    Args:
        frame: BGR numpy array (OpenCV format).
        timestamp: Timestamp of the frame in seconds from clip start.
        device: 'cpu' or 'cuda'.
        confidence_threshold: Minimum confidence for person detections.

    Returns:
        List of detection dicts matching the API contract.
    """
    model = _get_model(device)

    # Run YOLO inference — COCO classes, we only care about class 0 (person)
    results = model.predict(
        source=frame,
        classes=[0],  # person class only
        conf=confidence_threshold,
        verbose=False,
    )

    detections: List[Dict[str, Any]] = []

    if not results or len(results) == 0:
        return detections

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0].cpu().numpy())

        # Clamp bbox to frame bounds
        fh, fw = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw, x2)
        y2 = min(fh, y2)

        person_roi = frame[y1:y2, x1:x2]

        # Classify hard hat
        hat_label, hat_conf = _classify_hard_hat(person_roi)
        detections.append(
            Detection(
                label=hat_label,
                confidence=round(hat_conf * conf, 4),  # combine heuristic confidence with person det confidence
                bbox=[x1, y1, x2, y2],
                frame_timestamp=timestamp,
                metadata={
                    "description": (
                        "Person wearing hard hat correctly"
                        if hat_label == "hard_hat_present"
                        else "Person detected without hard hat"
                    ),
                },
            ).to_dict()
        )

        # Classify hi-vis
        vis_label, vis_conf = _classify_hi_vis(person_roi)
        detections.append(
            Detection(
                label=vis_label,
                confidence=round(vis_conf * conf, 4),
                bbox=[x1, y1, x2, y2],
                frame_timestamp=timestamp,
                metadata={
                    "description": (
                        "Person wearing hi-vis vest"
                        if vis_label == "hi_vis_present"
                        else "Person detected without hi-vis vest"
                    ),
                },
            ).to_dict()
        )

    return detections
