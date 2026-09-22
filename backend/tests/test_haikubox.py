"""Tests for Haikubox audio ingest, payload parsing, deduplication, and recorrelation."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Detection, HaikuboxDetection, Species, Visit
from db.session import Base
from db.utils import utcnow
from ingest import haikubox
from settings import settings


@pytest.fixture()
def db():
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


def test_pick_helper():
    """_pick returns first matching non-null key value."""
    obj = {"b": None, "c": "found", "d": "later"}
    assert haikubox._pick(obj, ("a", "b", "c", "d")) == "found"
    assert haikubox._pick(obj, ("x", "y")) is None


def test_parse_timestamp_formats():
    """_parse_timestamp handles ISO strings, epoch seconds, epoch milliseconds, and invalids."""
    # ISO strings
    dt = haikubox._parse_timestamp("2026-05-20T12:00:00Z")
    assert dt == datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)

    dt2 = haikubox._parse_timestamp("2026-05-20T12:00:00+00:00")
    assert dt2 == datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Epoch seconds
    dt3 = haikubox._parse_timestamp(1747742400)
    assert dt3.tzinfo is not None

    # Epoch milliseconds (> 1e12)
    dt4 = haikubox._parse_timestamp(1747742400000)
    assert dt4.tzinfo is not None
    assert dt3 == dt4

    # Invalid / None
    assert haikubox._parse_timestamp(None) is None
    assert haikubox._parse_timestamp("not-a-date") is None


def test_extract_detections_shapes():
    """_extract_detections handles lists and various dictionary wrapping keys."""
    raw_list = [{"cn": "Blue Jay"}]
    assert haikubox._extract_detections(raw_list) == raw_list

    assert haikubox._extract_detections({"detections": raw_list}) == raw_list
    assert haikubox._extract_detections({"results": raw_list}) == raw_list
    assert haikubox._extract_detections({"data": raw_list}) == raw_list
    assert haikubox._extract_detections({"items": raw_list}) == raw_list
    assert haikubox._extract_detections({"unknown": 123}) == []


def test_fetch_recent_detections_mocked():
    """fetch_recent_detections calls the correct API endpoint and handles errors gracefully."""
    mock_client = MagicMock()

    # When credentials are empty, returns empty list
    with patch.object(settings, "haikubox_api_key", ""), patch.object(settings, "haikubox_serial", ""):
        assert haikubox.fetch_recent_detections(mock_client) == []

    with patch.object(settings, "haikubox_api_key", "fake-key"), patch.object(settings, "haikubox_serial", "box123"):
        # 401 unauthorized
        resp_401 = MagicMock(status_code=401)
        mock_client.get.return_value = resp_401
        assert haikubox.fetch_recent_detections(mock_client) == []

        # 404 not found
        resp_404 = MagicMock(status_code=404)
        mock_client.get.return_value = resp_404
        assert haikubox.fetch_recent_detections(mock_client) == []

        # 200 OK
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = [{"cn": "Northern Cardinal", "dt": "2026-05-20T12:00:00Z"}]
        mock_client.get.return_value = resp_200
        result = haikubox.fetch_recent_detections(mock_client)
        assert len(result) == 1
        assert result[0]["cn"] == "Northern Cardinal"


def test_upsert_detections_windowed_dedup(db):
    """upsert_detections correctly deduplicates within the timestamp window and inserts new records."""
    t0 = datetime(2026, 5, 20, 12, 0, 0)
    raw = [
        {"cn": "Blue Jay", "dt": t0.isoformat() + "Z", "score": 0.85},
        {"cn": "American Goldfinch", "dt": (t0 + timedelta(minutes=1)).isoformat() + "Z", "confidence": "0.92"},
        {"common_name": "Mourning Dove", "timestamp": (t0 + timedelta(minutes=2)).isoformat() + "Z"},
    ]

    # Initial insert
    inserted = haikubox.upsert_detections(db, raw)
    assert inserted == 3
    assert db.query(HaikuboxDetection).count() == 3

    # Re-running same batch inserts 0 (deduplication)
    reinserted = haikubox.upsert_detections(db, raw)
    assert reinserted == 0
    assert db.query(HaikuboxDetection).count() == 3

    # Running batch with 1 new item inserts only 1
    new_raw = [
        {"cn": "Blue Jay", "dt": t0.isoformat() + "Z"},  # duplicate
        {"cn": "Black-capped Chickadee", "dt": (t0 + timedelta(minutes=3)).isoformat() + "Z", "score": 0.77},
    ]
    inserted_new = haikubox.upsert_detections(db, new_raw)
    assert inserted_new == 1
    assert db.query(HaikuboxDetection).count() == 4


def test_recorrelate_recent_detections(db):
    """_recorrelate_recent_detections sets audio_confirmed on matching detections."""
    now = utcnow()
    sp = Species(common_name="Blue Jay", scientific_name="Cyanocitta cristata")
    db.add(sp)
    db.commit()

    visit = Visit(started_at=now - timedelta(minutes=2), clip_path="clips/v1.mp4")
    db.add(visit)
    db.commit()

    det = Detection(
        visit_id=visit.id,
        track_id=1,
        species_id=sp.id,
        confidence=0.90,
        audio_confirmed=0,
        crop_path="crops/v1_t1.jpg",
        bbox=[100, 100, 200, 200],
    )
    db.add(det)

    # Add audio detection within the correlation window (e.g. 1 minute before visit)
    audio = HaikuboxDetection(
        species_common_name="Blue Jay",
        detected_at=now - timedelta(minutes=2, seconds=30),
        confidence=0.88,
    )
    db.add(audio)
    db.commit()

    # Recorrelate
    updated = haikubox._recorrelate_recent_detections(db)
    assert updated == 1
    db.refresh(det)
    assert det.audio_confirmed == 1
