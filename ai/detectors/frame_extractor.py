"""Video frame extraction using OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    sample_rate: int = 1,
) -> List[Tuple[float, np.ndarray]]:
    """Extract frames from a video at the given sample rate (frames per second).

    Args:
        video_path: Path to the video file.
        sample_rate: Number of frames to sample per second of video (default 1 fps).

    Returns:
        List of (timestamp_seconds, frame_numpy_array) tuples.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video cannot be opened or has zero frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has invalid metadata (fps={fps}, frames={total_frames}): {video_path}")

    frame_interval = max(1, int(round(fps / sample_rate)))

    frames: List[Tuple[float, np.ndarray]] = []
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp = frame_index / fps
                frames.append((round(timestamp, 3), frame))

            frame_index += 1
    except cv2.error as e:
        logger.warning("OpenCV error while reading %s at frame %d: %s", video_path, frame_index, e)
    finally:
        cap.release()

    logger.info(
        "Extracted %d frames from %s (%.1f fps, %d total frames)",
        len(frames),
        video_path.name,
        fps,
        total_frames,
    )

    return frames
