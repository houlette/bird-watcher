"""Tests for Phase 2: Diversity-First feed mode and Today's Yard Story endpoint."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Detection, Species, Visit
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


def test_diversity_feed_collapses_per_species_day(client, db_session, monkeypatch):
    """GET /api/detections?diversity=true returns only 1 best detection per species per day."""
    monkeypatch.setattr(settings, "camera_timezone", "America/New_York")

    cardinal = Species(common_name="Northern Cardinal", scientific_name="Cardinalis cardinalis", is_rare=False)
    sparrow = Species(common_name="House Sparrow", scientific_name="Passer domesticus", is_rare=False)
    db_session.add_all([cardinal, sparrow])
    db_session.commit()

    # Day 1: May 1 2026 EDT
    t1 = datetime(2026, 5, 1, 14, 0, 0)
    v1 = Visit(started_at=t1, clip_path="clips/v1.mp4")
    v2 = Visit(started_at=t1 + timedelta(minutes=10), clip_path="clips/v2.mp4")
    db_session.add_all([v1, v2])
    db_session.commit()

    # 3 Sparrow detections in v1 and v2
    d_sp1 = Detection(
        visit_id=v1.id, species_id=sparrow.id, confidence=0.85,
        sharpness=100.0, crop_area_px=1000, bbox=[0, 0, 10, 10], track_id=1,
        crop_path="crops/sp1.jpg", raw_predictions=[], audio_confirmed=False, created_at=t1,
    )
    d_sp2 = Detection(
        visit_id=v2.id, species_id=sparrow.id, confidence=0.95,
        sharpness=500.0, crop_area_px=2000, bbox=[0, 0, 10, 10], track_id=1,
        crop_path="crops/sp2.jpg", raw_predictions=[], audio_confirmed=False, created_at=t1,
    )
    # 1 Cardinal detection
    d_card1 = Detection(
        visit_id=v1.id, species_id=cardinal.id, confidence=0.92,
        sharpness=300.0, crop_area_px=1500, bbox=[0, 0, 10, 10], track_id=2,
        crop_path="crops/card1.jpg", raw_predictions=[], audio_confirmed=False, created_at=t1,
    )
    db_session.add_all([d_sp1, d_sp2, d_card1])
    db_session.commit()

    # Normal feed returns all 3 detections
    res_normal = client.get("/api/detections")
    assert res_normal.status_code == 200
    assert len(res_normal.json()) == 3

    # Diversity feed returns 2 detections (1 per species)
    res_div = client.get("/api/detections?diversity=true")
    assert res_div.status_code == 200
    items = res_div.json()
    assert len(items) == 2

    by_species = {item["species"]: item for item in items}
    assert "House Sparrow" in by_species
    assert "Northern Cardinal" in by_species

    # Sparrow daily_count should be 2, and the chosen detection is d_sp2 (higher confidence 0.95)
    sp_item = by_species["House Sparrow"]
    assert sp_item["daily_count"] == 2
    assert sp_item["id"] == d_sp2.id

    # Cardinal daily_count should be 1
    card_item = by_species["Northern Cardinal"]
    assert card_item["daily_count"] == 1
    assert card_item["id"] == d_card1.id


def test_get_daily_story_endpoint(client, db_session, monkeypatch):
    """GET /api/detections/daily_story returns stats, hero bird, and species highlights."""
    monkeypatch.setattr(settings, "camera_timezone", "America/New_York")

    cardinal = Species(common_name="Northern Cardinal", scientific_name="Cardinalis cardinalis", is_rare=False)
    blue_jay = Species(common_name="Blue Jay", scientific_name="Cyanocitta cristata", is_rare=False)
    db_session.add_all([cardinal, blue_jay])
    db_session.commit()

    # Target day: 2026-06-15
    t = datetime(2026, 6, 15, 12, 0, 0)
    v = Visit(started_at=t, clip_path="clips/v.mp4")
    db_session.add(v)
    db_session.commit()

    d1 = Detection(
        visit_id=v.id, species_id=cardinal.id, confidence=0.98,
        sharpness=850.0, crop_area_px=2000, bbox=[0, 0, 10, 10], track_id=1,
        crop_path="crops/c.jpg", raw_predictions=[], audio_confirmed=False, created_at=t,
    )
    d2 = Detection(
        visit_id=v.id, species_id=blue_jay.id, confidence=0.91,
        sharpness=400.0, crop_area_px=1800, bbox=[0, 0, 10, 10], track_id=2,
        crop_path="crops/bj.jpg", raw_predictions=[], audio_confirmed=False, created_at=t,
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    res = client.get("/api/detections/daily_story?target_date=2026-06-15")
    assert res.status_code == 200
    data = res.json()

    assert data["date"] == "2026-06-15"
    assert data["has_data"] is True
    assert data["total_detections"] == 2
    assert data["species_count"] == 2
    assert data["hero"]["species"] == "Northern Cardinal"  # highest score
    assert len(data["species_highlights"]) == 2
