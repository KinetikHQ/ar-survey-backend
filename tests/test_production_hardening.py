"""Production hardening regression tests."""

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.routes import UploadInitRequest, enqueue_processing, upload_complete, UploadCompleteRequest
from config import Settings
from models.database import Base, Job, JobStatus


def test_production_rejects_default_api_key_and_jwt_secret(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql://arsurvey:password@db:5432/ar_survey")
    monkeypatch.setenv("S3_ACCESS_KEY", "real-access-key")
    monkeypatch.setenv("S3_SECRET_KEY", "real-secret-key")
    monkeypatch.setenv("API_KEY", "dev-api-key-change-in-prod")
    monkeypatch.setenv("JWT_SECRET", "super-secret-jwt-key-change-in-prod")

    with pytest.raises(ValueError):
        Settings()


def test_upload_init_rejects_non_video_content_type():
    with pytest.raises(ValueError):
        UploadInitRequest(device_id="dev-001", duration_seconds=10, content_type="text/plain")


def test_upload_complete_rejects_missing_storage_object():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    db = Session()
    try:
        job = Job(id=uuid.uuid4(), device_id="dev-001", duration_seconds=10, status=JobStatus.pending, video_key="clips/test.mp4")
        db.add(job)
        db.commit()

        with patch("api.routes.storage_object_exists", return_value=False):
            with pytest.raises(HTTPException) as exc:
                upload_complete(UploadCompleteRequest(job_id=job.id, file_size_bytes=1024), db=db)
        assert exc.value.status_code == 400
        assert "not found" in exc.value.detail.lower()
    finally:
        db.close()


def test_enqueue_processing_is_idempotent_for_processing_or_completed_jobs():
    job = Job(id=uuid.uuid4(), device_id="dev-001", duration_seconds=10, status=JobStatus.processing)
    assert enqueue_processing(job) == "processing"

    job.status = JobStatus.completed
    assert enqueue_processing(job) == "completed"
