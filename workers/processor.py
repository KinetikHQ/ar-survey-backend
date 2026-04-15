"""RQ worker task — processes a single video clip through the AI pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

from models.database import Job, JobStatus, Result
from models.session import SessionLocal
from storage import download_to_file, get_video_key

logger = logging.getLogger(__name__)


def process_clip(job_id_str: str) -> None:
    """Download video from S3, run AI pipeline, persist results.

    This is invoked by an RQ worker via:
        queue.enqueue("workers.processor.process_clip", str(job.id))
    """
    job_id = uuid.UUID(job_id_str)
    db = SessionLocal()

    try:
        # --- fetch job row ---
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            raise ValueError(f"Job {job_id} not found in database")

        # --- mark processing ---
        job.status = JobStatus.processing
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # --- download video to a temp file ---
        key = job.video_key or get_video_key(job_id)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            download_to_file(key, tmp_path)
        except FileNotFoundError:
            # Dev mode: no actual file uploaded, create a dummy for pipeline testing
            logger.warning("No video file for job %s — creating dummy for dev testing", job_id)
            _create_dummy_video(tmp_path)

        try:
            # --- run AI pipeline (stub for now) ---
            from ai.pipeline import process_clip

            detections: list[dict] = process_clip(tmp_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # --- persist results ---
        for det in detections:
            result = Result(
                job_id=job_id,
                category=det.get("category", "ppe"),
                label=det["label"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                frame_timestamp=det["frame_timestamp"],
                metadata_=det.get("metadata"),
            )
            db.add(result)

        # --- mark completed ---
        job.status = JobStatus.completed
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as exc:
        # --- mark failed ---
        db.rollback()
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = JobStatus.failed
            job.completed_at = datetime.now(timezone.utc)
            job.error_message = str(exc)[:2000]
            db.commit()
        raise  # re-raise so RQ records the failure
    finally:
        db.close()


def _create_dummy_video(path: str, duration_seconds: int = 5, fps: int = 30) -> None:
    """Create a minimal valid MP4 for dev/testing when no real video exists."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    try:
        for _ in range(duration_seconds * fps):
            # Simple colored frames so the pipeline has something to process
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :] = (128, 128, 128)  # grey
            writer.write(frame)
    finally:
        writer.release()
