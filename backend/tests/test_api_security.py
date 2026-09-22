"""Security and input validation tests for public API routes:
bounds checking, DoS mitigation, and data poisoning prevention.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Detection, Visit
from db.session import Base, get_db
from main import app


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


def test_detections_limit_bounds_prevent_dos(client):
    """GET /api/detections enforces ge=1 and le=200 to prevent LIMIT -1 SQLite bypass."""
    # Negative limit must be rejected by FastAPI validation
    res_neg = client.get("/api/detections?limit=-1")
    assert res_neg.status_code == 422

    # Zero limit must be rejected
    res_zero = client.get("/api/detections?limit=0")
    assert res_zero.status_code == 422

    # Excessive limit > 200 must be rejected
    res_huge = client.get("/api/detections?limit=500")
    assert res_huge.status_code == 422

    # Valid limits within [1, 200] succeed
    res_valid = client.get("/api/detections?limit=10")
    assert res_valid.status_code == 200


def test_corrections_rejects_oversized_and_malicious_species_names(client, db_session):
    """POST /api/corrections rejects names exceeding 100 chars or containing control characters."""
    visit = Visit(clip_path="clips/v1.mp4")
    db_session.add(visit)
    db_session.commit()

    det = Detection(
        visit_id=visit.id,
        track_id=1,
        crop_path="crops/v1_t1.jpg",
        bbox=[10, 10, 50, 50],
        confidence=0.8,
    )
    db_session.add(det)
    db_session.commit()

    # Oversized species name triggers Pydantic schema validation error (422)
    huge_name = "A" * 150
    res_huge = client.post(
        "/api/corrections",
        json={"detection_id": det.id, "correct_species_name": huge_name},
    )
    assert res_huge.status_code == 422

    # Control characters / newline injection triggers custom validation (400)
    res_ctrl = client.post(
        "/api/corrections",
        json={"detection_id": det.id, "correct_species_name": "Robin\nEvil: True"},
    )
    assert res_ctrl.status_code == 400
    assert "control characters" in res_ctrl.json()["detail"]


def test_bulk_corrections_rejects_excessive_detection_ids(client):
    """POST /api/corrections/bulk rejects batch sizes exceeding 500 IDs."""
    huge_ids = list(range(1, 600))
    res = client.post(
        "/api/corrections/bulk",
        json={"detection_ids": huge_ids, "correct_species_name": "Blue Jay"},
    )
    assert res.status_code == 422


def test_ingest_filename_truncation_prevents_fs_overflow(client, tmp_path, monkeypatch):
    """Super-long filenames are truncated to prevent OS NAME_MAX crashes."""
    from routers import ingest
    monkeypatch.setattr(ingest, "CLIPS_DIR", tmp_path)

    # 300 character filename
    long_filename = "A" * 300 + ".jpg"
    files = {"file": (long_filename, b"\xff\xd8\xff dummy jpeg", "image/jpeg")}
    res = client.post("/api/ingest/motion", files=files)
    assert res.status_code == 200

    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 1
    # File name length on disk must be well under standard 255 char limit
    assert len(saved_files[0].name) < 120
