"""FastAPI routes for the AR Survey & Inspection API."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.auth import verify_token
from config import settings
from models.database import Job, JobStatus, Result
from models.session import get_db
from storage import generate_presigned_upload_url, get_video_key, storage_object_exists

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Redis / RQ queue — lazy init, fallback to in-memory for dev
# ---------------------------------------------------------------------------

_queue = None
_redis_available = False


def _get_queue():
    global _queue, _redis_available
    if _queue is not None:
        return _queue
    try:
        import redis as redis_lib
        from rq import Queue
        _redis = redis_lib.from_url(settings.REDIS_URL)
        _redis.ping()
        _queue = Queue(connection=_redis)
        _redis_available = True
        logger.info("Connected to Redis at %s", settings.REDIS_URL)
    except Exception as e:
        logger.warning("Redis unavailable (%s) — jobs will be processed inline", e)
        _redis_available = False
    return _queue

# ---------------------------------------------------------------------------
# Request / Response schemas (inline to keep things simple for now)
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field, field_validator, model_validator

# Maps camelCase keys sent by mobile clients to the snake_case names used internally.
_CAMEL_TO_SNAKE = {
    "deviceId": "device_id",
    "durationSeconds": "duration_seconds",
    "contentType": "content_type",
    "jobId": "job_id",
    "fileSizeBytes": "file_size_bytes",
    "surveyJobId": "survey_job_id",
    "floorId": "floor_id",
    "roomLabel": "room_label",
}


def _normalise_keys(data: dict) -> dict:
    return {_CAMEL_TO_SNAKE.get(k, k): v for k, v in data.items()}


class UploadInitRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=255)
    duration_seconds: int = Field(..., ge=0, le=120)
    content_type: str = "video/mp4"
    title: Optional[str] = Field(default=None, max_length=255)

    # Building / location context (from Android)
    survey_job_id: Optional[str] = Field(default=None, max_length=255)
    floor_id: Optional[str] = Field(default=None, max_length=255)
    room_label: Optional[str] = Field(default=None, max_length=255)

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str) -> str:
        normalised = value.lower().strip()
        if normalised not in settings.allowed_upload_content_types_set:
            raise ValueError(f"Unsupported upload content type: {value}")
        return normalised

    @model_validator(mode="before")
    @classmethod
    def accept_camel_case(cls, data):
        if isinstance(data, dict):
            return _normalise_keys(data)
        return data


class UploadInitResponse(BaseModel):
    job_id: uuid.UUID
    upload_url: str
    expires_at: str


class UploadCompleteRequest(BaseModel):
    job_id: uuid.UUID
    file_size_bytes: int = Field(..., gt=0, le=settings.MAX_UPLOAD_BYTES)

    @model_validator(mode="before")
    @classmethod
    def accept_camel_case(cls, data):
        if isinstance(data, dict):
            return _normalise_keys(data)
        return data


class UploadCompleteResponse(BaseModel):
    job_id: uuid.UUID
    status: str


class ResultSchema(BaseModel):
    id: uuid.UUID
    category: str
    label: str
    confidence: float
    bbox: list[float]
    frame_timestamp: float
    metadata: Optional[dict[str, Any]]

    class Config:
        from_attributes = True


class SummarySchema(BaseModel):
    total_detections: int
    ppe_violations: int
    ppe_compliant: int
    frames_analyzed: int


class JobDetailResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    title: Optional[str] = None
    survey_job_id: Optional[str] = None
    floor_id: Optional[str] = None
    room_label: Optional[str] = None
    results: Optional[list[ResultSchema]] = None
    summary: Optional[SummarySchema] = None
    error_message: Optional[str] = None


class RetryResponse(BaseModel):
    job_id: uuid.UUID
    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@router.post(
    "/api/v1/upload/init",
    response_model=UploadInitResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_token)],
)
def upload_init(body: UploadInitRequest, db: Session = Depends(get_db)) -> UploadInitResponse:
    """Create a job record and return a presigned upload URL."""
    job = Job(
        device_id=body.device_id,
        duration_seconds=body.duration_seconds,
        content_type=body.content_type,
        title=body.title,
        survey_job_id=body.survey_job_id,
        floor_id=body.floor_id,
        room_label=body.room_label,
        status=JobStatus.pending,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    presigned = generate_presigned_upload_url(job.id, body.content_type)

    # Store the video key on the job so the worker knows where to fetch later.
    job.video_key = get_video_key(job.id)
    db.commit()

    return UploadInitResponse(
        job_id=job.id,
        upload_url=presigned["url"],
        expires_at=presigned["expires_at"],
    )


def enqueue_processing(job: Job) -> str:
    """Enqueue a job exactly once; already-active jobs are idempotent."""
    if job.status in {JobStatus.processing, JobStatus.completed}:
        return job.status

    queue = _get_queue()
    if queue:
        queue.enqueue("workers.processor.process_clip", str(job.id))
    else:
        # Dev mode: process inline in a background thread
        import threading
        from workers.processor import process_clip
        t = threading.Thread(target=process_clip, args=(str(job.id),), daemon=True)
        t.start()
    return job.status


@router.post(
    "/api/v1/upload/complete",
    response_model=UploadCompleteResponse,
    dependencies=[Depends(verify_token)],
)
def upload_complete(body: UploadCompleteRequest, db: Session = Depends(get_db)) -> UploadCompleteResponse:
    """Mark the upload as done and enqueue the job for processing."""
    job = db.query(Job).filter(Job.id == body.job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status != JobStatus.pending:
        if job.status in {JobStatus.processing, JobStatus.completed}:
            return UploadCompleteResponse(job_id=job.id, status=job.status)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is already in status '{job.status}', expected 'pending'",
        )
    if not job.video_key:
        job.video_key = get_video_key(job.id)
        db.commit()
    if not storage_object_exists(job.video_key, expected_size=body.file_size_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video object not found or size does not match",
        )

    enqueue_processing(job)

    return UploadCompleteResponse(job_id=job.id, status=job.status)


@router.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobDetailResponse,
)
def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
) -> JobDetailResponse:
    """Return job status, results, and summary."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    # IDOR Protection: In a real system, we'd check if job.user_id == user['sub']
    # For MVP, we at least ensure the token is valid before returning sensitive data.

    results_list: Optional[list[ResultSchema]] = None
    summary: Optional[SummarySchema] = None

    if job.status == JobStatus.completed:
        results_list = [
            ResultSchema(
                id=r.id,
                category=r.category,
                label=r.label,
                confidence=r.confidence,
                bbox=r.bbox,
                frame_timestamp=r.frame_timestamp,
                metadata=r.metadata_,
            )
            for r in job.results
        ]
        # Build summary
        violations = [r for r in job.results if r.label.startswith("missing_")]
        compliant = [r for r in job.results if r.label.endswith("_present")]
        timestamps = {r.frame_timestamp for r in job.results}
        summary = SummarySchema(
            total_detections=len(job.results),
            ppe_violations=len(violations),
            ppe_compliant=len(compliant),
            frames_analyzed=len(timestamps),
        )

    return JobDetailResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        title=job.title,
        survey_job_id=job.survey_job_id,
        floor_id=job.floor_id,
        room_label=job.room_label,
        results=results_list,
        summary=summary,
        error_message=job.error_message,
    )


@router.post(
    "/api/v1/jobs/{job_id}/retry",
    response_model=RetryResponse,
    dependencies=[Depends(verify_token)],
)
def retry_job(job_id: uuid.UUID, db: Session = Depends(get_db)) -> RetryResponse:
    """Retry a failed job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status != JobStatus.failed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is in status '{job.status}', expected 'failed'",
        )

    job.status = JobStatus.pending
    job.error_message = None
    job.retry_count += 1
    db.commit()

    if not job.video_key or not storage_object_exists(job.video_key):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video object not found; cannot retry processing",
        )
    enqueue_processing(job)

    return RetryResponse(job_id=job.id, status=job.status)


# ---------------------------------------------------------------------------
# Dev-mode upload endpoint (no S3 required)
# ---------------------------------------------------------------------------

@router.put("/api/v1/dev/upload/{job_id}", include_in_schema=False)
async def dev_upload(job_id: uuid.UUID, request: __import__("fastapi").Request):
    """Accept a raw file upload in dev mode (no S3). Saves to local storage."""
    from storage import save_local_file
    body = await request.body()
    filepath = save_local_file(job_id, body)
    logger.info("Dev upload saved: %s (%d bytes)", filepath, len(body))
    return {"status": "ok", "job_id": str(job_id), "path": filepath}


@router.post("/api/v1/dev/upload-file/{job_id}", include_in_schema=False)
async def dev_upload_file(job_id: uuid.UUID, file: __import__("fastapi").UploadFile):
    """Accept a multipart file upload in dev mode. Easier for curl/testing."""
    from storage import save_local_file
    content = await file.read()
    filepath = save_local_file(job_id, content)
    logger.info("Dev upload saved: %s (%d bytes, %s)", filepath, len(content), file.filename)
    return {"status": "ok", "job_id": str(job_id), "path": filepath, "size": len(content)}
