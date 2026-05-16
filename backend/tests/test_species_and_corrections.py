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
    visit = Visit(started_at=datetime.utcnow(), clip_path="clips/test.webm")
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
    det = db.query(Detection).get(det_id)
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
