"""Tests for /api/ingest router: connectivity pings, multipart uploads, auth, and validation."""
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Visit
from db.session import Base, get_db
from main import app
from settings import settings


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_ingest_ping_get(client):
    """GET /api/ingest/motion without body is a connectivity ping returning 200."""
    res = client.get("/api/ingest/motion")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "kind": "ping"}


def test_ingest_ping_post_empty(client):
    """POST /api/ingest/motion without body is a connectivity ping returning 200."""
    res = client.post("/api/ingest/motion")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "kind": "ping"}


def test_ingest_valid_file_upload(client, db_session, tmp_path):
    """Valid video or image upload writes to clips dir and creates a Visit row."""
    with patch("routers.ingest.CLIPS_DIR", tmp_path):
        files = {"file": ("Birdfeeder_20260520120000.mp4", b"dummy video content", "video/mp4")}
        res = client.post("/api/ingest/motion", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["kind"] == "clip"
        assert data["bytes"] == len(b"dummy video content")

        visit = db_session.get(Visit, data["visit_id"])
        assert visit is not None
        assert "Birdfeeder_20260520120000.mp4" in visit.clip_path


def test_ingest_rejects_unsupported_file_extension(client, tmp_path):
    """Executable or unexpected script extension is rejected with 400 Bad Request."""
    with patch("routers.ingest.CLIPS_DIR", tmp_path):
        files = {"file": ("malicious_payload.sh", b"#!/bin/sh\nrm -rf /", "application/x-sh")}
        res = client.post("/api/ingest/motion", files=files)
        assert res.status_code == 400
        assert "Unsupported file extension" in res.json()["detail"]


def test_ingest_rejects_oversized_upload(client, tmp_path, monkeypatch):
    """Uploads exceeding byte limit are rejected with 413 and unlinked."""
    from routers import ingest
    monkeypatch.setattr(ingest, "MAX_INGEST_BYTES", 100)
    monkeypatch.setattr(ingest, "CLIPS_DIR", tmp_path)

    files = {"file": ("clip.mp4", b"x" * 200, "video/mp4")}
    res = client.post("/api/ingest/motion", files=files)
    assert res.status_code == 413
    assert "Upload exceeds" in res.json()["detail"]
    assert len(list(tmp_path.iterdir())) == 0


def test_ingest_auth_token_enforcement(client, monkeypatch):
    """When ingest_auth_token is configured, unauthorized requests are rejected with 401."""
    monkeypatch.setattr(settings, "ingest_auth_token", "secret-token-123")

    # Missing token
    res = client.get("/api/ingest/motion")
    assert res.status_code == 401

    # Wrong token
    res = client.get("/api/ingest/motion", headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 401

    # Correct Bearer token
    res = client.get("/api/ingest/motion", headers={"Authorization": "Bearer secret-token-123"})
    assert res.status_code == 200

    # Correct query parameter token
    res = client.get("/api/ingest/motion?token=secret-token-123")
    assert res.status_code == 200
