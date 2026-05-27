"""Tests for the species directory endpoint and the corrections endpoint."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Correction, Detection, Species, Visit
from db.session import Base, get_db
from db.utils import utcnow
from main import app
from pipeline import calibration


@pytest.fixture()
def db():
    # SQLite :memory: databases are per-connection by default; in a test that
    # spans multiple threads (FastAPI's TestClient dispatches in a worker
    # thread) we need StaticPool to share one connection across them all,
    # plus check_same_thread=False to permit cross-thread access.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def client(db):
    def override():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _write_calibration(path, species: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": {"yearly_years": [2025], "daily_days": 200},
        "species": species,
    }))


def test_species_endpoint_uses_calibration(client, monkeypatch, tmp_path):
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    _write_calibration(cal_path, {
        "House Sparrow": {"total": 485000, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
        "Northern Cardinal": {"total": 62000, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
        "Vagrant Bird": {"total": 2, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},  # below threshold
    })
    r = client.get("/api/species")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "calibration"
    names = [s["name"] for s in body["species"]]
    assert names == ["House Sparrow", "Northern Cardinal"]  # sorted by count desc, vagrant excluded
    assert body["species"][0]["total"] == 485000


def test_species_endpoint_falls_back_when_uncalibrated(client, monkeypatch, tmp_path):
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "missing.json")
    r = client.get("/api/species")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "fallback"
    names = [s["name"] for s in body["species"]]
    assert "Northern Cardinal" in names
    assert "Dark-eyed Junco" in names


def _seed_detection(db, species_name: str = "Northern Cardinal") -> int:
    species = Species(common_name=species_name, scientific_name="", is_rare=False)
    db.add(species)
    db.flush()
    visit = Visit(started_at=utcnow(), clip_path="clips/test.webm")
    db.add(visit)
    db.flush()
    det = Detection(
        visit_id=visit.id, species_id=species.id, confidence=0.5,
        raw_predictions=[], audio_confirmed=False,
        crop_path="crops/test.jpg", bbox=[0, 0, 10, 10], track_id=1,
    )
    db.add(det)
    db.commit()
    return det.id


def test_correction_updates_detection_species(client, db):
    det_id = _seed_detection(db, "Northern Cardinal")
    r = client.post(
        "/api/corrections",
        json={"detection_id": det_id, "correct_species_name": "White-throated Sparrow"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["species"] == "White-throated Sparrow"

    # Detection now points at the new species
    det = db.get(Detection, det_id)
    db.refresh(det)
    assert det.species.common_name == "White-throated Sparrow"

    # A Correction row was created
    correction = db.query(Correction).filter_by(detection_id=det_id).one()
    assert correction.correct_species_id == det.species_id


def test_correction_creates_species_if_missing(client, db):
    det_id = _seed_detection(db, "Northern Cardinal")
    # The corrected species isn't in the DB yet — should be created on the fly.
    pre = db.query(Species).count()
    r = client.post(
        "/api/corrections",
        json={"detection_id": det_id, "correct_species_name": "Pyrrhuloxia"},
    )
    assert r.status_code == 200
    post = db.query(Species).count()
    assert post == pre + 1


def test_correction_rejects_empty_name(client, db):
    det_id = _seed_detection(db, "Northern Cardinal")
    r = client.post(
        "/api/corrections",
        json={"detection_id": det_id, "correct_species_name": "   "},
    )
    assert r.status_code == 400


def test_correction_rejects_unknown_detection(client):
    r = client.post(
        "/api/corrections",
        json={"detection_id": 99999, "correct_species_name": "Northern Cardinal"},
    )
    assert r.status_code == 404


def test_species_endpoint_excludes_sentinels_from_calibration(client, monkeypatch, tmp_path):
    """Sentinel labels ('Not a bird', 'Unknown bird') shouldn't appear in the
    regular species list — the picker surfaces them separately pinned at the top."""
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    _write_calibration(cal_path, {
        "Northern Cardinal": {"total": 100, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
        "Not a bird": {"total": 50, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
        "Unknown bird": {"total": 30, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
    })
    r = client.get("/api/species")
    payload = r.json()
    yard_names = [s["name"] for s in payload["yard"]]
    extra_names = [s["name"] for s in payload["extra"]]
    assert "Northern Cardinal" in yard_names
    assert "Not a bird" not in yard_names and "Not a bird" not in extra_names
    assert "Unknown bird" not in yard_names and "Unknown bird" not in extra_names


def test_detections_excludes_not_a_bird_by_default(client, db):
    """The feed should not show detections the user marked as 'Not a bird'."""
    from db.models import NOT_A_BIRD_LABEL
    cardinal_det = _seed_detection(db, "Northern Cardinal")
    feeder_det = _seed_detection(db, NOT_A_BIRD_LABEL)

    r = client.get("/api/detections")
    assert r.status_code == 200
    ids = [d["id"] for d in r.json()]
    assert cardinal_det in ids
    assert feeder_det not in ids


def test_detections_include_not_a_bird_when_requested(client, db):
    """Diagnostic opt-in flag exposes everything including false positives."""
    from db.models import NOT_A_BIRD_LABEL
    cardinal_det = _seed_detection(db, "Northern Cardinal")
    feeder_det = _seed_detection(db, NOT_A_BIRD_LABEL)

    r = client.get("/api/detections?include_not_a_bird=true")
    ids = [d["id"] for d in r.json()]
    assert cardinal_det in ids
    assert feeder_det in ids


def test_detections_pagination_with_compound_cursor(client, db):
    """Infinite-scroll cursor: `before` returns only rows captured earlier
    (started_at, id) than the cursor."""
    ids = [_seed_detection(db, f"Test Species {i}") for i in range(5)]

    # All seeded with utcnow() in the same call window → tied on started_at,
    # ordered by id desc as the tiebreaker.
    page1 = client.get("/api/detections?limit=3").json()
    assert [d["id"] for d in page1] == [ids[4], ids[3], ids[2]]
    # Response includes the compound cursor and the capture timestamp.
    assert "cursor" in page1[0]
    assert "captured_at" in page1[0]

    # Second page uses the last row's cursor to get the next older slice.
    page2 = client.get(
        "/api/detections", params={"limit": 3, "before": page1[-1]["cursor"]}
    ).json()
    assert [d["id"] for d in page2] == [ids[1], ids[0]]

    # Past the end → empty.
    page3 = client.get(
        "/api/detections", params={"limit": 3, "before": page2[-1]["cursor"]}
    ).json()
    assert page3 == []


def test_detections_sorted_by_capture_time_not_processing_time(client, db):
    """During a backlog drain, an OLD visit just processed shouldn't pop to the
    top of the feed. Sort key is Visit.started_at desc, not Detection.id desc."""
    from datetime import datetime

    # Insert in id-order opposite to capture-time-order: the most recently
    # inserted Detection is the OLDEST sighting.
    older_capture = datetime(2026, 5, 26, 8, 0, 0)
    newer_capture = datetime(2026, 5, 26, 12, 0, 0)

    species = Species(common_name="Test Bird", scientific_name="", is_rare=False)
    db.add(species)
    db.flush()

    # First insert: newer-captured visit (smaller id).
    v_new = Visit(started_at=newer_capture, clip_path="clips/a.mp4")
    db.add(v_new); db.flush()
    d_new = Detection(
        visit_id=v_new.id, species_id=species.id, confidence=0.5,
        raw_predictions=[], audio_confirmed=False,
        crop_path="crops/a.jpg", bbox=[0, 0, 10, 10], track_id=1,
    )
    db.add(d_new); db.flush()

    # Second insert: older-captured visit (larger id) — simulates an old
    # backlog visit that just got processed.
    v_old = Visit(started_at=older_capture, clip_path="clips/b.mp4")
    db.add(v_old); db.flush()
    d_old = Detection(
        visit_id=v_old.id, species_id=species.id, confidence=0.5,
        raw_predictions=[], audio_confirmed=False,
        crop_path="crops/b.jpg", bbox=[0, 0, 10, 10], track_id=1,
    )
    db.add(d_old); db.commit()

    rows = client.get("/api/detections").json()
    ids_in_order = [d["id"] for d in rows]
    # Capture-time ordering: newer-captured first, regardless of id order.
    assert ids_in_order.index(d_new.id) < ids_in_order.index(d_old.id)


def test_species_endpoint_returns_yard_and_extra_groups(client, monkeypatch, tmp_path):
    """Picker needs two groups: yard (Haikubox-heard) and extra (broader NA)."""
    cal_path = tmp_path / "yard_priors.json"
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", cal_path)
    _write_calibration(cal_path, {
        "Northern Cardinal": {"total": 100, "monthly_pct": {str(m): 1 / 12 for m in range(1, 13)}},
    })
    body = client.get("/api/species").json()
    assert "yard" in body
    assert "extra" in body
    assert [s["name"] for s in body["yard"]] == ["Northern Cardinal"]
    extra_names = [s["name"] for s in body["extra"]]
    # Common NA species the user might want to label but the Haikubox missed.
    assert "Rock Pigeon" in extra_names
    assert "Red-tailed Hawk" in extra_names
    # Should NOT duplicate yard species in the extra list.
    assert "Northern Cardinal" not in extra_names


def test_correction_to_not_a_bird_removes_from_feed(client, db):
    """End-to-end: user marks a misidentified Ovenbird as Not a bird,
    and the next GET /api/detections excludes it."""
    from db.models import NOT_A_BIRD_LABEL
    det_id = _seed_detection(db, "Ovenbird")
    assert det_id in [d["id"] for d in client.get("/api/detections").json()]

    r = client.post(
        "/api/corrections",
        json={"detection_id": det_id, "correct_species_name": NOT_A_BIRD_LABEL},
    )
    assert r.status_code == 200
    assert r.json()["species"] == NOT_A_BIRD_LABEL

    assert det_id not in [d["id"] for d in client.get("/api/detections").json()]
    assert det_id in [d["id"] for d in client.get("/api/detections?include_not_a_bird=true").json()]
