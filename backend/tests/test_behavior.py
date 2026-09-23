"""Tests for Feeder Science: plumage dimorphism, pair dynamics, and dwell analytics."""
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Detection, Species, Visit
from db.session import Base, get_db
from main import app
from pipeline.stats import compute_feeder_behavior


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


def test_compute_feeder_behavior(db_session):
    # Setup species: Cardinal (dimorphic) and Chickadee (monomorphic)
    cardinal = Species(common_name="Northern Cardinal", scientific_name="Cardinalis cardinalis")
    chickadee = Species(common_name="Black-capped Chickadee", scientific_name="Poecile atricapillus")
    db_session.add_all([cardinal, chickadee])
    db_session.commit()

    now = datetime(2026, 6, 1, 10, 0, 0)

    # Visit 1: Pair visit! Male and Female Cardinal together
    v1 = Visit(started_at=now, clip_path="clips/v1.mp4")
    db_session.add(v1)
    db_session.commit()

    d_male = Detection(
        visit_id=v1.id, species_id=cardinal.id, confidence=0.95,
        sex="male", track_id=1, crop_path="crops/card_m.jpg", bbox=[0,0,10,10],
        track_frames=[0, 1, 2, 3, 4, 5],  # 6 frames = 2.0s
    )
    d_female = Detection(
        visit_id=v1.id, species_id=cardinal.id, confidence=0.92,
        sex="female", track_id=2, crop_path="crops/card_f.jpg", bbox=[0,0,10,10],
        track_frames=[0, 1, 2],  # 3 frames = 1.0s
    )
    db_session.add_all([d_male, d_female])
    db_session.commit()

    # Visit 2: Quick Chickadee visits (5 detections so it meets the >=5 threshold for dwell rankings)
    for i in range(5):
        v = Visit(started_at=now + timedelta(minutes=i*5), clip_path=f"clips/chk_{i}.mp4")
        db_session.add(v)
        db_session.commit()
        d_chk = Detection(
            visit_id=v.id, species_id=chickadee.id, confidence=0.90,
            sex=None, track_id=1, crop_path=f"crops/chk_{i}.jpg", bbox=[0,0,10,10],
            track_frames=[0, 1],  # 2 frames = 0.67s
        )
        db_session.add(d_chk)
    db_session.commit()

    # Compute behavior
    behavior = compute_feeder_behavior(db_session)

    # 1. Dimorphic results
    card_info = next(d for d in behavior["dimorphic_species"] if d["species"] == "Northern Cardinal")
    assert card_info["male"] == 1
    assert card_info["female"] == 1
    assert card_info["pair_visits"] == 1
    assert card_info["male_pct"] == 50
    assert card_info["female_pct"] == 50

    # 2. Pair highlights
    assert len(behavior["pair_highlights"]) == 1
    pair = behavior["pair_highlights"][0]
    assert pair["species"] == "Northern Cardinal"
    assert pair["visit_id"] == v1.id
    assert len(pair["crops"]) == 2

    # 3. Dwell rankings
    chk_dwell = next(d for d in behavior["dwell_rankings"] if d["species"] == "Black-capped Chickadee")
    assert chk_dwell["sample_count"] == 5
    assert chk_dwell["avg_seconds"] < 1.0
    assert chk_dwell["style"] == "Quick Forager"


def test_api_stats_behavior_endpoint(client, db_session):
    cardinal = Species(common_name="Northern Cardinal", scientific_name="Cardinalis cardinalis")
    db_session.add(cardinal)
    db_session.commit()

    v = Visit(started_at=datetime.utcnow(), clip_path="clips/v.mp4")
    db_session.add(v)
    db_session.commit()

    d = Detection(
        visit_id=v.id, species_id=cardinal.id, confidence=0.90,
        sex="male", track_id=1, crop_path="crops/c.jpg", bbox=[0,0,10,10],
    )
    db_session.add(d)
    db_session.commit()

    res = client.get("/api/stats/behavior")
    assert res.status_code == 200
    data = res.json()
    assert "dimorphic_species" in data
    assert "dwell_rankings" in data
    assert "pair_highlights" in data


def test_list_detections_includes_sex_and_pair(client, db_session):
    cardinal = Species(common_name="Northern Cardinal", scientific_name="Cardinalis cardinalis")
    db_session.add(cardinal)
    db_session.commit()

    now = datetime(2026, 6, 1, 12, 0, 0)
    v = Visit(started_at=now, clip_path="clips/v.mp4")
    db_session.add(v)
    db_session.commit()

    d1 = Detection(
        visit_id=v.id, species_id=cardinal.id, confidence=0.95,
        sex="male", track_id=1, crop_path="crops/m.jpg", bbox=[0,0,10,10],
    )
    d2 = Detection(
        visit_id=v.id, species_id=cardinal.id, confidence=0.93,
        sex="female", track_id=2, crop_path="crops/f.jpg", bbox=[0,0,10,10],
    )
    db_session.add_all([d1, d2])
    db_session.commit()

    res = client.get("/api/detections")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 2
    for item in items:
        assert item["sex"] in ("male", "female")
        assert item["is_pair"] is True
