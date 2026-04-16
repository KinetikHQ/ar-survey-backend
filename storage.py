"""S3 / MinIO storage abstraction using boto3.

Falls back to local file storage when S3 is unavailable (dev mode).
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy S3 client — falls back to local storage
# ---------------------------------------------------------------------------

_s3 = None
_local_storage_dir = Path("./storage/clips")


def _get_s3():
    global _s3
    if _s3 is not None:
        return _s3
    try:
        import boto3
        from botocore.client import Config as BotoConfig
        _s3 = boto3.client(
            "s3",
            endpoint_url=settings.S3_ENDPOINT,
            aws_access_key_id=settings.S3_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SECRET_KEY,
            config=BotoConfig(signature_version="s3v4"),
        )
        # Quick connectivity check
        _s3.list_buckets()
        logger.info("Connected to S3 at %s", settings.S3_ENDPOINT)
    except Exception as e:
        logger.warning("S3 unavailable (%s) — using local file storage", e)
        _s3 = None
        _local_storage_dir.mkdir(parents=True, exist_ok=True)
    return _s3

UPLOAD_EXPIRY_SECONDS = 3600  # 1 hour
DOWNLOAD_EXPIRY_SECONDS = 900  # 15 minutes


class PresignedUpload(TypedDict):
    url: str
    expires_at: str


class PresignedDownload(TypedDict):
    url: str


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _video_key(job_id: uuid.UUID) -> str:
    return f"clips/{job_id}.mp4"


def generate_presigned_upload_url(job_id: uuid.UUID, content_type: str) -> PresignedUpload:
    """Return a presigned PUT URL the client uses to upload video directly.
    
    In dev mode (no S3), returns a local file upload endpoint.
    """
    key = _video_key(job_id)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=UPLOAD_EXPIRY_SECONDS)
    
    s3 = _get_s3()
    if s3:
        url = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": settings.S3_BUCKET,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=UPLOAD_EXPIRY_SECONDS,
        )
    else:
        # Dev mode: return a local upload endpoint
        # Use request Host header or configured base URL — never localhost (phone can't reach it)
        url = f"/api/v1/dev/upload/{job_id}"
        url = settings.DEV_UPLOAD_BASE_URL.rstrip("/") + url
    return {"url": url, "expires_at": expires_at.isoformat()}


def generate_presigned_download_url(key: str) -> str:
    """Return a presigned GET URL for reading an object from S3."""
    s3 = _get_s3()
    if s3:
        return s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": settings.S3_BUCKET,
                "Key": key,
            },
            ExpiresIn=DOWNLOAD_EXPIRY_SECONDS,
        )
    else:
        return f"http://localhost:8000/api/v1/dev/download/{key}"


def download_to_file(key: str, dest: str) -> None:
    """Download an S3 object to a local file path."""
    s3 = _get_s3()
    if s3:
        s3.download_file(settings.S3_BUCKET, key, dest)
    else:
        # Dev mode: copy from local storage
        src = _local_storage_dir / Path(key).name
        if src.exists():
            import shutil
            shutil.copy2(str(src), dest)
        else:
            raise FileNotFoundError(f"File not found: {src}")


def save_local_file(job_id: uuid.UUID, data: bytes) -> str:
    """Save uploaded data to local storage (dev mode)."""
    _local_storage_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{job_id}.mp4"
    filepath = _local_storage_dir / filename
    filepath.write_bytes(data)
    return str(filepath)


def get_video_key(job_id: uuid.UUID) -> str:
    """Public accessor for the S3 key pattern."""
    return _video_key(job_id)
