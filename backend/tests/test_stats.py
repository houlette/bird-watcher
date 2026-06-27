"""Tests for the daily-stats aggregation + /api/stats endpoint."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (
    NOT_A_BIRD_LABEL,
    UNKNOWN_BIRD_LABEL,
    Correction,
    Detection,
    Species,
    Visit,
)
from db.session import Base, get_db
from main import app
from pipeline.stats import (
    compute_daily_stats,
    compute_global_stats,
    compute_species_activity,
    upsert_daily_stats,
)


@pytest.fixture()
def db():
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


def _species(db, name: str) -> Species:
    sp = db.query(Species).filter_by(common_name=name).one_or_none()
    if sp is None:
        sp = Species(common_name=name, scientific_name="", is_rare=False)
        db.add(sp)
        db.flush()
    return sp


def _visit(db, captured: datetime, *, processing_error: str | None = None) -> Visit:
    v = Visit(
        started_at=captured,
        clip_path=f"clips/{captured.isoformat()}.mp4",
        processed_at=captured if processing_error is not None else captured,
        processing_error=processing_error,
    )
    db.add(v)
    db.flush()
    return v


def _detection(
    db,
    visit: Visit,
    *,
    species: Species | None = None,
    track_id: int = 1,
    yolo_confidence: float | None = None,
    raw_top1: str | None = None,
    audio_confirmed: bool = False,
) -> Detection:
    raw = [{"species": raw_top1, "p": 0.9}] if raw_top1 else []
    d = Detection(
        visit_id=visit.id,
        species_id=species.id if species else None,
        confidence=0.5 if species else 0.0,
        yolo_confidence=yolo_confidence,
        raw_predictions=raw,
        audio_confirmed=audio_confirmed,
        crop_path=f"crops/{visit.id}_{track_id}.jpg",
        bbox=[0, 0, 10, 10],
        track_id=track_id,
    )
    db.add(d)
    db.flush()
    return d


def _correct(db, det: Detection, species_name: str) -> Correction:
    sp = _species(db, species_name)
    det.species_id = sp.id  # mirrors corrections.py behavior
    c = Correction(detection_id=det.id, correct_species_id=sp.id)
    db.add(c)
    db.flush()
    return c


def test_compute_daily_stats_funnel(db):
    """Cover every branch of the funnel in one day: daylight-skipped clip,
    a clip with no detections, a clip with multiple detections, a NAB
    correction, an Unknown correction, a correct species correction, and
    a wrong species correction.
    """
    day = datetime(2026, 5, 30, 0, 0, 0)

    # 1. Daylight-skipped visit (still a clip, but excluded from the
    #    daylight-clip funnel stage).
    _visit(db, day.replace(hour=2), processing_error="skipped: captured outside daylight hours")

    # 2. Daylight visit, YOLO found nothing.
    _visit(db, day.replace(hour=8))

    # 3. Daylight visit with 4 detections, exercising every correction branch.
    v = _visit(db, day.replace(hour=12))
    cardinal = _species(db, "Northern Cardinal")
    jay = _species(db, "Blue Jay")

    # 3a. Classifier-labeled Cardinal, user agrees (correct).
    d_correct = _detection(db, v, species=cardinal, track_id=1,
                           yolo_confidence=0.7, raw_top1="Northern Cardinal")
    _correct(db, d_correct, "Northern Cardinal")

    # 3b. Classifier guessed Cardinal, user says Blue Jay (miss).
    d_wrong = _detection(db, v, species=cardinal, track_id=2,
                         yolo_confidence=0.6, raw_top1="Northern Cardinal")
    _correct(db, d_wrong, "Blue Jay")

    # 3c. Classifier guessed Cardinal, user says NAB.
    d_nab = _detection(db, v, species=cardinal, track_id=3,
                       yolo_confidence=0.3, raw_top1="Northern Cardinal")
    _correct(db, d_nab, NOT_A_BIRD_LABEL)

    # 3d. Classifier guessed nothing (species_id=None), user says Unknown.
    d_unk = _detection(db, v, species=None, track_id=4,
                       yolo_confidence=0.4, raw_top1=None,
                       audio_confirmed=True)
    _correct(db, d_unk, UNKNOWN_BIRD_LABEL)

    # 4. Operational signal: visit with non-skip processing error.
    _visit(db, day.replace(hour=14), processing_error="frame extraction failed")

    db.commit()

    stats = compute_daily_stats(db, day.date())

    assert stats.clips_received == 4
    assert stats.clips_daylight == 3   # 4 minus 1 daylight-skip
    assert stats.clips_with_detections == 1
    assert stats.detections_total == 4
    # 3 detections had a non-sentinel classifier species set (Cardinal x3);
    # the Unknown one was species_id=None at insert time.
    assert stats.detections_labeled_by_classifier == 3
    assert stats.detections_user_corrected == 4
    assert stats.corrections_nab == 1
    assert stats.corrections_unknown == 1
    assert stats.corrections_real_species == 2
    # Classifier-eligible = 2 (Cardinal-correct + Cardinal→BlueJay miss);
    # Classifier-correct = 1 (Cardinal agreement).
    assert stats.classifier_eligible == 2
    assert stats.classifier_correct == 1
    assert stats.visits_with_processing_error == 1
    assert stats.detections_audio_confirmed == 1

    # Hour-of-day: 4 detections all at hour 12.
    assert stats.payload["hour_of_day"][12] == 4
    assert sum(stats.payload["hour_of_day"]) == 4

    # YOLO-confidence hist: NAB bucket at 0.3 → index 3.
    nab_hist = stats.payload["yolo_confidence_hist"]["nab"]
    species_hist = stats.payload["yolo_confidence_hist"]["species"]
    assert nab_hist[3] == 1
    assert sum(nab_hist) == 1
    # Species hist: 0.7 → 7, 0.6 → 6. Unknown is sentinel, not counted.
    assert species_hist[7] == 1
    assert species_hist[6] == 1
    assert sum(species_hist) == 2


def test_compute_daily_stats_empty_day_returns_zeros(db):
    stats = compute_daily_stats(db, date(2020, 1, 1))
    assert stats.clips_received == 0
    assert stats.detections_total == 0
    assert stats.detections_user_corrected == 0


def test_upsert_daily_stats_is_idempotent(db):
    day = datetime(2026, 5, 30, 0, 0, 0)
    _visit(db, day.replace(hour=12))
    db.commit()

    a = upsert_daily_stats(db, day.date())
    assert a.clips_received == 1

    # Add another visit on the same day; upsert again should refresh.
    _visit(db, day.replace(hour=13))
    db.commit()
    b = upsert_daily_stats(db, day.date())
    assert b.clips_received == 2

    # Still one row in the table.
    from db.models import PipelineStatsDaily
    assert db.query(PipelineStatsDaily).count() == 1


def test_global_stats_pending_backlog_and_top_species(db):
    day = datetime(2026, 5, 30, 0, 0, 0)
    v = _visit(db, day.replace(hour=12))
    cardinal = _species(db, "Northern Cardinal")
    jay = _species(db, "Blue Jay")

    # Three Cardinals: one corrected, two pending.
    d1 = _detection(db, v, species=cardinal, track_id=1, raw_top1="Northern Cardinal")
    d2 = _detection(db, v, species=cardinal, track_id=2, raw_top1="Northern Cardinal")
    d3 = _detection(db, v, species=cardinal, track_id=3, raw_top1="Northern Cardinal")
    _correct(db, d1, "Northern Cardinal")

    # One NAB-labeled detection — should NOT count toward pending backlog.
    d4 = _detection(db, v, species=cardinal, track_id=4, raw_top1="Northern Cardinal")
    _correct(db, d4, NOT_A_BIRD_LABEL)
    # The NAB correction sets species_id to NAB sentinel — but it ALSO
    # has a Correction row, so the backlog query (which requires NO
    # Correction) excludes it regardless.
    db.commit()

    g = compute_global_stats(db)
    assert g["pending_backlog"] == 2   # d2 and d3
    # Top species: Cardinal has 1 real-species correction (d1).
    species_names = [s["species"] for s in g["top_species"]]
    assert "Northern Cardinal" in species_names
    # NAB sentinel must not appear.
    assert NOT_A_BIRD_LABEL not in species_names


def test_endpoint_returns_daily_with_today_recomputed(client, db):
    """The /api/stats endpoint should always include a row for today even
    if no nightly snapshot has been written yet, and the row should
    reflect live data."""
    today = datetime.now(timezone.utc).date()
    today_dt = datetime.combine(today, datetime.min.time()).replace(hour=12)
    _visit(db, today_dt)
    db.commit()

    r = client.get("/api/stats?days=7")
    assert r.status_code == 200
    body = r.json()
    assert "daily" in body and "totals" in body and "as_of" in body
    assert len(body["daily"]) == 7
    last = body["daily"][-1]
    assert last["date"] == today.isoformat()
    assert last["clips_received"] == 1
    # Derived rates surfaced.
    assert "yolo_bird_rate" in last
    assert "classifier_accuracy" in last


def test_species_activity_buckets_in_local_time(db):
    """compute_species_activity buckets detections per species by LOCAL
    hour and week, excludes sentinels, and sorts by total desc."""
    from zoneinfo import ZoneInfo

    from settings import settings

    tz = ZoneInfo(settings.camera_timezone)

    def expect(dt_utc: datetime) -> tuple[int, int]:
        local = dt_utc.replace(tzinfo=timezone.utc).astimezone(tz)
        return local.hour, min((local.timetuple().tm_yday - 1) // 7, 52)

    cardinal = _species(db, "Northern Cardinal")
    jay = _species(db, "Blue Jay")

    # Two Cardinal detections at the same instant (one visit, two birds)
    # plus a third on a different week — Cardinal should out-total the Jay.
    t1 = datetime(2026, 5, 30, 14, 0, 0)   # mid-spring
    t2 = datetime(2026, 7, 4, 23, 30, 0)   # ~5 weeks later, near local midnight
    v1 = _visit(db, t1)
    _detection(db, v1, species=cardinal, track_id=1)
    _detection(db, v1, species=cardinal, track_id=2)
    v2 = _visit(db, t2)
    _detection(db, v2, species=cardinal, track_id=1)

    # One Blue Jay, and one sentinel detection that must be ignored.
    v3 = _visit(db, t1)
    _detection(db, v3, species=jay, track_id=1)
    nab = _species(db, NOT_A_BIRD_LABEL)
    _detection(db, v3, species=nab, track_id=2)
    db.commit()

    out = compute_species_activity(db)
    by_name = {s["species"]: s for s in out["species"]}

    # Sentinel excluded entirely.
    assert NOT_A_BIRD_LABEL not in by_name
    # Sorted by total desc → Cardinal (3) before Jay (1).
    assert [s["species"] for s in out["species"]] == ["Northern Cardinal", "Blue Jay"]

    card = by_name["Northern Cardinal"]
    assert card["total"] == 3
    assert sum(card["by_hour"]) == 3
    assert sum(card["by_week"]) == 3
    h1, w1 = expect(t1)
    h2, w2 = expect(t2)
    assert card["by_hour"][h1] == 2   # the two simultaneous Cardinals
    assert card["by_hour"][h2] == 1
    assert card["by_week"][w1] == 2
    assert card["by_week"][w2] == 1

    jay_row = by_name["Blue Jay"]
    assert jay_row["total"] == 1
    assert jay_row["by_hour"][h1] == 1
    assert out["tz"] == settings.camera_timezone


def test_species_activity_endpoint(client, db):
    day = datetime(2026, 5, 30, 12, 0, 0)
    v = _visit(db, day)
    cardinal = _species(db, "Northern Cardinal")
    _detection(db, v, species=cardinal, track_id=1)
    db.commit()

    r = client.get("/api/stats/activity")
    assert r.status_code == 200
    body = r.json()
    assert "species" in body and "tz" in body and "as_of" in body
    assert body["species"][0]["species"] == "Northern Cardinal"
    assert len(body["species"][0]["by_hour"]) == 24
    assert len(body["species"][0]["by_week"]) == 53


def test_endpoint_fills_gap_days_with_zeros(client, db):
    """A 7-day window with no data should still return 7 rows, all zeroed."""
    r = client.get("/api/stats?days=7")
    assert r.status_code == 200
    daily = r.json()["daily"]
    assert len(daily) == 7
    assert all(d["clips_received"] == 0 for d in daily)
