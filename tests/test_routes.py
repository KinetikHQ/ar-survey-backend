"""Unit tests for API routes, request schemas, and storage URL generation."""

import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth import create_access_token
from api.main import app
from api.routes import UploadInitRequest, UploadCompleteRequest, _normalise_keys
from config import settings
from models.database import Base, Job, JobStatus
from models.session import get_db

# ---------------------------------------------------------------------------
# Test database — in-memory SQLite with StaticPool so all connections share
# the same database instance (required for SQLite :memory: with SQLAlchemy).
# ---------------------------------------------------------------------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def _override_get_db():
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db

# Read the real API key so auth tests don't need to patch settings.
API_KEY = settings.API_KEY


@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


@pytest.fixture
def client():
    with patch("storage._get_s3", return_value=None):
        with TestClient(app) as c:
            yield c


def auth(key: str = API_KEY) -> dict:
    return {"Authorization": f"Bearer {key}"}


def jwt_auth(sub: str = "test-user") -> dict:
    token = create_access_token({"sub": sub})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# _normalise_keys — camelCase mapping
# ---------------------------------------------------------------------------

def test_normalise_keys_converts_camel_to_snake():
    result = _normalise_keys({"deviceId": "abc", "durationSeconds": 10, "contentType": "video/mp4"})
    assert result == {"device_id": "abc", "duration_seconds": 10, "content_type": "video/mp4"}


def test_normalise_keys_passes_snake_through_unchanged():
    result = _normalise_keys({"device_id": "abc", "duration_seconds": 10})
    assert result == {"device_id": "abc", "duration_seconds": 10}


def test_normalise_keys_mixed_input():
    result = _normalise_keys({"deviceId": "abc", "duration_seconds": 10})
    assert result == {"device_id": "abc", "duration_seconds": 10}


def test_normalise_keys_survey_fields():
    result = _normalise_keys({"surveyJobId": "sj-1", "floorId": "f-2", "roomLabel": "Kitchen"})
    assert result == {"survey_job_id": "sj-1", "floor_id": "f-2", "room_label": "Kitchen"}


# ---------------------------------------------------------------------------
# UploadInitRequest — schema validation
# ---------------------------------------------------------------------------

def test_upload_init_request_accepts_snake_case():
    req = UploadInitRequest(device_id="dev-001", duration_seconds=30)
    assert req.device_id == "dev-001"
    assert req.duration_seconds == 30
    assert req.content_type == "video/mp4"


def test_upload_init_request_accepts_camel_case():
    req = UploadInitRequest.model_validate({"deviceId": "dev-001", "durationSeconds": 30})
    assert req.device_id == "dev-001"
    assert req.duration_seconds == 30


def test_upload_init_request_duration_zero_is_valid():
    req = UploadInitRequest(device_id="dev-001", duration_seconds=0)
    assert req.duration_seconds == 0


def test_upload_init_request_duration_max_boundary():
    req = UploadInitRequest(device_id="dev-001", duration_seconds=120)
    assert req.duration_seconds == 120


def test_upload_init_request_duration_over_max_raises():
    with pytest.raises(Exception):
        UploadInitRequest(device_id="dev-001", duration_seconds=121)


def test_upload_init_request_title_is_optional():
    req = UploadInitRequest(device_id="dev-001", duration_seconds=10)
    assert req.title is None


def test_upload_init_request_title_accepted_when_provided():
    req = UploadInitRequest(device_id="dev-001", duration_seconds=10, title="Site A inspection")
    assert req.title == "Site A inspection"


def test_upload_init_request_empty_device_id_raises():
    with pytest.raises(Exception):
        UploadInitRequest(device_id="", duration_seconds=10)


def test_upload_init_request_survey_fields_optional():
    req = UploadInitRequest(device_id="dev-001", duration_seconds=10)
    assert req.survey_job_id is None
    assert req.floor_id is None
    assert req.room_label is None


def test_upload_init_request_survey_fields_accepted():
    req = UploadInitRequest(
        device_id="dev-001",
        duration_seconds=10,
        survey_job_id="sj-123",
        floor_id="floor-2",
        room_label="Kitchen",
    )
    assert req.survey_job_id == "sj-123"
    assert req.floor_id == "floor-2"
    assert req.room_label == "Kitchen"


def test_upload_init_request_survey_fields_camel_case():
    req = UploadInitRequest.model_validate({
        "deviceId": "dev-001",
        "durationSeconds": 10,
        "surveyJobId": "sj-456",
        "floorId": "floor-3",
        "roomLabel": "Bathroom",
    })
    assert req.survey_job_id == "sj-456"
    assert req.floor_id == "floor-3"
    assert req.room_label == "Bathroom"


def test_upload_init_request_full_android_payload():
    """Simulate the exact JSON the Android app sends."""
    req = UploadInitRequest.model_validate({
        "deviceId": "abc123",
        "durationSeconds": 45,
        "title": "Ground floor walkthrough",
        "contentType": "video/mp4",
        "surveyJobId": "sj-789",
        "floorId": "floor-1",
        "roomLabel": "A - Living Room",
    })
    assert req.device_id == "abc123"
    assert req.duration_seconds == 45
    assert req.title == "Ground floor walkthrough"
    assert req.content_type == "video/mp4"
    assert req.survey_job_id == "sj-789"
    assert req.floor_id == "floor-1"
    assert req.room_label == "A - Living Room"


# ---------------------------------------------------------------------------
# UploadCompleteRequest — camelCase
# ---------------------------------------------------------------------------

def test_upload_complete_request_accepts_camel_case():
    job_id = uuid.uuid4()
    req = UploadCompleteRequest.model_validate({"jobId": str(job_id), "fileSizeBytes": 1024})
    assert req.job_id == job_id
    assert req.file_size_bytes == 1024


# ---------------------------------------------------------------------------
# Auth — static API key
# ---------------------------------------------------------------------------

def test_upload_init_returns_201(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=auth(),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "job_id" in body
    assert "upload_url" in body
    assert "expires_at" in body


def test_upload_init_no_auth_returns_403(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
    )
    assert resp.status_code == 403


def test_upload_init_wrong_token_returns_401(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=auth("definitely-wrong-key"),
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auth — JWT
# ---------------------------------------------------------------------------

def test_upload_init_jwt_returns_201(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=jwt_auth(),
    )
    assert resp.status_code == 201


def test_get_job_jwt_returns_200(client):
    # Create a job first
    init_resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=jwt_auth(),
    )
    job_id = init_resp.json()["job_id"]

    # Fetch it with JWT
    resp = client.get(f"/api/v1/jobs/{job_id}", headers=jwt_auth())
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id


# ---------------------------------------------------------------------------
# POST /api/v1/upload/init — endpoint
# ---------------------------------------------------------------------------

def test_upload_init_url_is_valid_http(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=auth(),
    )
    assert resp.status_code == 201
    url = resp.json()["upload_url"]
    assert url.startswith("http")
    assert "/api/v1/dev/upload/" in url


def test_upload_init_accepts_camel_case_body(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"deviceId": "dev-001", "durationSeconds": 15},
        headers=auth(),
    )
    assert resp.status_code == 201


def test_upload_init_accepts_full_android_payload(client):
    """End-to-end: send the exact JSON the Android app sends."""
    resp = client.post(
        "/api/v1/upload/init",
        json={
            "deviceId": "abc123",
            "durationSeconds": 45,
            "title": "Ground floor walkthrough",
            "contentType": "video/mp4",
            "surveyJobId": "sj-789",
            "floorId": "floor-1",
            "roomLabel": "A - Living Room",
        },
        headers=auth(),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "job_id" in body
    assert "upload_url" in body


def test_upload_init_missing_device_id_returns_422(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"duration_seconds": 10},
        headers=auth(),
    )
    assert resp.status_code == 422


def test_upload_init_duration_over_max_returns_422(client):
    resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 999},
        headers=auth(),
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/upload/complete — endpoint
# ---------------------------------------------------------------------------

def test_upload_complete_returns_success(client):
    # Create a job
    init_resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=auth(),
    )
    job_id = init_resp.json()["job_id"]

    # Upload bytes before completing — production path validates storage exists.
    upload_body = b"0" * 1024
    upload_resp = client.put(f"/api/v1/dev/upload/{job_id}", content=upload_body)
    assert upload_resp.status_code == 200

    # Complete upload
    with patch("api.routes.enqueue_processing", return_value="pending"):
        resp = client.post(
            "/api/v1/upload/complete",
            json={"job_id": job_id, "file_size_bytes": len(upload_body)},
            headers=auth(),
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


def test_upload_complete_unknown_job_returns_404(client):
    resp = client.post(
        "/api/v1/upload/complete",
        json={"job_id": str(uuid.uuid4()), "file_size_bytes": 1024},
        headers=auth(),
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id} — endpoint
# ---------------------------------------------------------------------------

def test_get_unknown_job_returns_404(client):
    resp = client.get(f"/api/v1/jobs/{uuid.uuid4()}", headers=auth())
    assert resp.status_code == 404


def test_get_job_returns_survey_fields(client):
    """Verify survey/floor/room context is returned in job detail."""
    init_resp = client.post(
        "/api/v1/upload/init",
        json={
            "deviceId": "dev-001",
            "durationSeconds": 30,
            "surveyJobId": "sj-123",
            "floorId": "floor-2",
            "roomLabel": "Kitchen",
        },
        headers=auth(),
    )
    job_id = init_resp.json()["job_id"]

    resp = client.get(f"/api/v1/jobs/{job_id}", headers=auth())
    assert resp.status_code == 200
    body = resp.json()
    assert body["survey_job_id"] == "sj-123"
    assert body["floor_id"] == "floor-2"
    assert body["room_label"] == "Kitchen"


def test_get_job_returns_title(client):
    init_resp = client.post(
        "/api/v1/upload/init",
        json={"deviceId": "dev-001", "durationSeconds": 10, "title": "Roof inspection"},
        headers=auth(),
    )
    job_id = init_resp.json()["job_id"]

    resp = client.get(f"/api/v1/jobs/{job_id}", headers=auth())
    assert resp.status_code == 200
    assert resp.json()["title"] == "Roof inspection"


def test_get_job_no_auth_returns_403(client):
    resp = client.get(f"/api/v1/jobs/{uuid.uuid4()}")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /api/v1/jobs/{job_id}/retry — endpoint
# ---------------------------------------------------------------------------

def test_retry_completed_job_returns_409(client):
    """Cannot retry a job that isn't in 'failed' status."""
    init_resp = client.post(
        "/api/v1/upload/init",
        json={"device_id": "dev-001", "duration_seconds": 10},
        headers=auth(),
    )
    job_id = init_resp.json()["job_id"]

    resp = client.post(f"/api/v1/jobs/{job_id}/retry", headers=auth())
    assert resp.status_code == 409


def test_retry_unknown_job_returns_404(client):
    resp = client.post(f"/api/v1/jobs/{uuid.uuid4()}/retry", headers=auth())
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /health — no auth required
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "version" in resp.json()
