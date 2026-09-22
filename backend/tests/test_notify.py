"""Unit tests for the rarity decision logic.

We don't test the actual pywebpush send path here — that's I/O against an
external service. We test the decision: 'given the detection history in
the DB, should we push?'.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Detection, PushSubscription, Species, Visit
from db.session import Base
from db.utils import utcnow
from pipeline.notify import (
    RESIDENT_SPECIES,
    build_payload,
    dispatch_for_detection,
    is_first_today,
    is_lifer,
    is_rare,
)
from settings import settings


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    s: Session = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _seed(db: Session, species_name: str = "Northern Cardinal") -> tuple[int, Visit]:
    species = Species(common_name=species_name, scientific_name="", is_rare=False)
    db.add(species)
    db.flush()
    visit = Visit(started_at=utcnow(), clip_path="clips/test.webm")
    db.add(visit)
    db.flush()
    return species.id, visit


def _add_detection(db: Session, species_id: int, visit_id: int, when: datetime) -> Detection:
    d = Detection(
        visit_id=visit_id,
        species_id=species_id,
        confidence=0.9,
        raw_predictions=[],
        audio_confirmed=False,
        crop_path="crops/test.jpg",
        bbox=[0, 0, 10, 10],
        track_id=1,
        created_at=when,
    )
    db.add(d)
    db.flush()
    return d


def test_first_ever_sighting_is_rare(db):
    species_id, visit = _seed(db)
    now = datetime(2026, 5, 1, 12, 0, 0)
    new_det = _add_detection(db, species_id, visit.id, now)
    assert is_rare(db, species_id, new_det.created_at, window_days=30) is True


def test_repeat_within_window_is_not_rare(db):
    """A second sighting of the same species 5 days later should not re-trigger a push."""
    species_id, visit = _seed(db)
    earlier = datetime(2026, 5, 1, 12, 0, 0)
    _add_detection(db, species_id, visit.id, earlier)
    later = earlier + timedelta(days=5)
    new_det = _add_detection(db, species_id, visit.id, later)
    assert is_rare(db, species_id, new_det.created_at, window_days=30) is False


def test_repeat_outside_window_is_rare_again(db):
    """A sighting more than `window_days` after the last one should re-trigger."""
    species_id, visit = _seed(db)
    last_year = datetime(2025, 5, 1, 12, 0, 0)
    _add_detection(db, species_id, visit.id, last_year)
    today = datetime(2026, 5, 1, 12, 0, 0)
    new_det = _add_detection(db, species_id, visit.id, today)
    assert is_rare(db, species_id, new_det.created_at, window_days=30) is True


def test_other_species_in_window_does_not_suppress(db):
    """A cardinal yesterday shouldn't suppress a *junco* push today."""
    cardinal_id, visit = _seed(db, "Northern Cardinal")
    junco_id, _ = _seed(db, "Dark-eyed Junco")
    yesterday = datetime(2026, 4, 30, 12, 0, 0)
    _add_detection(db, cardinal_id, visit.id, yesterday)
    today = datetime(2026, 5, 1, 12, 0, 0)
    new_junco = _add_detection(db, junco_id, visit.id, today)
    assert is_rare(db, junco_id, new_junco.created_at, window_days=30) is True


def test_same_visit_multiple_tracks_not_self_suppressed(db):
    """Two tracks of the same species from the same visit, persisted within
    a second of each other, should both still register as 'rare' (we want a
    single push per visit, but the suppression is on PRIOR detections, not
    same-instant siblings)."""
    species_id, visit = _seed(db)
    now = datetime(2026, 5, 1, 12, 0, 0)
    d1 = _add_detection(db, species_id, visit.id, now)
    d2 = _add_detection(db, species_id, visit.id, now)
    # Both should see no prior detection (the 1-second fudge in is_rare excludes
    # the just-created sibling); de-duplication of bursts is handled elsewhere.
    assert is_rare(db, species_id, d1.created_at, window_days=30) is True
    assert is_rare(db, species_id, d2.created_at, window_days=30) is True


def test_window_size_zero_means_every_sighting_is_rare(db):
    """Edge case: window=0 → no lookback → always push."""
    species_id, visit = _seed(db)
    earlier = datetime(2026, 5, 1, 12, 0, 0)
    _add_detection(db, species_id, visit.id, earlier)
    later = earlier + timedelta(days=1)
    new_det = _add_detection(db, species_id, visit.id, later)
    assert is_rare(db, species_id, new_det.created_at, window_days=0) is True


def test_long_window_suppresses_more_aggressively(db):
    """A 365-day window suppresses a same-day re-sighting just like 30 does."""
    species_id, visit = _seed(db)
    earlier = datetime(2025, 6, 1, 12, 0, 0)  # almost a year ago
    _add_detection(db, species_id, visit.id, earlier)
    today = datetime(2026, 5, 1, 12, 0, 0)
    new_det = _add_detection(db, species_id, visit.id, today)
    # With a 30-day window, the year-ago detection is outside → rare
    assert is_rare(db, species_id, new_det.created_at, window_days=30) is True
    # With a 365-day window, the year-ago detection is inside → not rare
    assert is_rare(db, species_id, new_det.created_at, window_days=365) is False


def test_is_lifer(db):
    species_id, visit = _seed(db, "Scarlet Tanager")
    t1 = datetime(2026, 5, 1, 10, 0, 0)
    d1 = _add_detection(db, species_id, visit.id, t1)
    assert is_lifer(db, species_id, d1.created_at) is True

    t2 = datetime(2026, 5, 2, 10, 0, 0)
    d2 = _add_detection(db, species_id, visit.id, t2)
    assert is_lifer(db, species_id, d2.created_at) is False


def test_is_first_today(db):
    species_id, visit = _seed(db, "Northern Cardinal")
    # Morning of day 1
    t1 = datetime(2026, 5, 1, 8, 0, 0)
    d1 = _add_detection(db, species_id, visit.id, t1)
    assert is_first_today(db, species_id, d1.created_at) is True

    # Afternoon of day 1
    t2 = datetime(2026, 5, 1, 15, 0, 0)
    d2 = _add_detection(db, species_id, visit.id, t2)
    assert is_first_today(db, species_id, d2.created_at) is False

    # Morning of day 2
    t3 = datetime(2026, 5, 2, 7, 0, 0)
    d3 = _add_detection(db, species_id, visit.id, t3)
    assert is_first_today(db, species_id, d3.created_at) is True


def test_build_payload_tiers(db):
    species_id, visit = _seed(db, "Indigo Bunting")
    species = db.get(Species, species_id)
    det = _add_detection(db, species_id, visit.id, datetime(2026, 5, 1, 12, 0, 0))

    lifer_payload = build_payload(det, species, tier="yard_lifer")
    assert "New Yard Visitor" in lifer_payload["title"]
    assert "First time recorded" in lifer_payload["body"]
    assert lifer_payload["data"]["tier"] == "yard_lifer"

    rare_payload = build_payload(det, species, tier="rare", window_days=45)
    assert "Rare Visitor" in rare_payload["title"]
    assert "45+ days" in rare_payload["body"]
    assert rare_payload["data"]["tier"] == "rare"

    first_today_payload = build_payload(det, species, tier="daily_first")
    assert first_today_payload["title"] == "Indigo Bunting"
    assert "First visit today" in first_today_payload["body"]
    assert first_today_payload["data"]["tier"] == "daily_first"


def test_dispatch_resident_muting(db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "vapid_public_key", "test_key")
    key_file = tmp_path / "vapid.pem"
    key_file.write_text("dummy-pem")
    monkeypatch.setattr(settings, "vapid_private_pem_path", str(key_file))

    pushed_payloads = []

    def mock_send(sub, payload, pem):
        pushed_payloads.append((sub, json.loads(payload)))
        return True

    monkeypatch.setattr("pipeline.notify._send_push", mock_send)

    dove_id, visit = _seed(db, "Mourning Dove")
    assert "Mourning Dove" in RESIDENT_SPECIES

    # Sub 1: muting enabled (default)
    sub_muted = PushSubscription(
        endpoint="https://push.example.com/sub/muted",
        p256dh="k1",
        auth="a1",
        mute_residents=True,
        notify_daily_first=True,
    )
    # Sub 2: muting disabled
    sub_unmuted = PushSubscription(
        endpoint="https://push.example.com/sub/unmuted",
        p256dh="k2",
        auth="a2",
        mute_residents=False,
        notify_daily_first=True,
    )
    db.add_all([sub_muted, sub_unmuted])
    db.commit()

    det = _add_detection(db, dove_id, visit.id, datetime(2026, 5, 1, 9, 0, 0))
    sent = dispatch_for_detection(db, det)
    # Only sub_unmuted should receive the push
    assert sent == 1
    assert len(pushed_payloads) == 1
    assert pushed_payloads[0][0].endpoint == "https://push.example.com/sub/unmuted"


def test_dispatch_daily_first_vs_repeat(db, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "vapid_public_key", "test_key")
    key_file = tmp_path / "vapid.pem"
    key_file.write_text("dummy-pem")
    monkeypatch.setattr(settings, "vapid_private_pem_path", str(key_file))

    pushed_payloads = []

    def mock_send(sub, payload, pem):
        pushed_payloads.append((sub, json.loads(payload)))
        return True

    monkeypatch.setattr("pipeline.notify._send_push", mock_send)

    cardinal_id, visit = _seed(db, "Northern Cardinal")
    # Seen yesterday -> not rare for a 30-day window
    _add_detection(db, cardinal_id, visit.id, datetime(2026, 5, 1, 10, 0, 0))

    # Subscriber with daily first enabled
    sub = PushSubscription(
        endpoint="https://push.example.com/sub/daily",
        p256dh="k",
        auth="a",
        notify_window_days=30,
        mute_residents=True,
        notify_daily_first=True,
    )
    db.add(sub)
    db.commit()

    # First arrival today (May 2)
    det1 = _add_detection(db, cardinal_id, visit.id, datetime(2026, 5, 2, 8, 0, 0))
    sent1 = dispatch_for_detection(db, det1)
    assert sent1 == 1
    assert len(pushed_payloads) == 1
    assert pushed_payloads[0][1]["data"]["tier"] == "daily_first"

    # Second arrival today (May 2) -> suppressed
    pushed_payloads.clear()
    det2 = _add_detection(db, cardinal_id, visit.id, datetime(2026, 5, 2, 11, 0, 0))
    sent2 = dispatch_for_detection(db, det2)
    assert sent2 == 0
    assert len(pushed_payloads) == 0

