"""SQLAlchemy models for jobs and detection results."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.types import JSON, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Uuid(), primary_key=True, default=uuid.uuid4)
    device_id = Column(String(255), nullable=False)
    status = Column(
        String(20),
        nullable=False,
        default=JobStatus.pending.value,
    )
    video_key = Column(String(512), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    content_type = Column(String(50), nullable=True, default="video/mp4")
    title = Column(String(255), nullable=True)

    # Building / location context (from Android)
    survey_job_id = Column(String(255), nullable=True)
    floor_id = Column(String(255), nullable=True)
    room_label = Column(String(255), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # relationship
    results: Mapped[list["Result"]] = relationship("Result", back_populates="job", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_jobs_device_id", "device_id"),
        Index("ix_jobs_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Job {self.id} status={self.status}>"


class Result(Base):
    __tablename__ = "results"

    id = Column(Uuid(), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        Uuid(),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    category = Column(String(50), nullable=False, default="ppe")
    label = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2]
    frame_timestamp = Column(Float, nullable=False)  # seconds from clip start
    metadata_ = Column("metadata", JSON, nullable=True)

    job: Mapped["Job"] = relationship("Job", back_populates="results")

    __table_args__ = (
        Index("ix_results_job_id", "job_id"),
    )

    def __repr__(self) -> str:
        return f"<Result {self.id} label={self.label} confidence={self.confidence}>"
