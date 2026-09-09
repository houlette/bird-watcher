"""Tests for the plumage palette, flight-path synthesis, and /api/art."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    UNKNOWN_BIRD_LABEL,
    Detection,
    Species,
    Visit,
)
from db.session import Base, get_db
from main import app
from pipeline.palette import FRAME_H, FRAME_W, flight_points, species_style
from settings import settings


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
def client(db, monkeypatch):
    # Pin the feeder's zone: every date assertion below depends on the
    # UTC→local shift, and a machine configured for another yard shouldn't
    # change what day a visit belongs to.
    monkeypatch.setattr(settings, "camera_timezone", "America/New_York")

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


def _species(db, name, sci="Testus birdus"):
    sp = Species(common_name=name, scientific_name=sci)
    db.add(sp)
    db.commit()
    return sp


def _detection(db, started_at, species=None, n=1, **kw):
    """Insert a visit with `n` detections, each its own track."""
    visit = Visit(started_at=started_at, ended_at=started_at + timedelta(seconds=8))
    db.add(visit)
    db.commit()
    made = []
    for i in range(n):
        d = Detection(
            visit_id=visit.id,
            species_id=species.id if species else None,
            confidence=kw.pop("confidence", 0.9),
            audio_confirmed=kw.pop("audio_confirmed", 0),
            crop_path=f"crops/{visit.id}_{i}.jpg",
            bbox=[1800 + i * 40, 900, 150, 150],
            track_id=i + 1,
            **kw,
        )
        db.add(d)
        made.append(d)
    db.commit()
    return visit, made


# ── Palette ─────────────────────────────────────────────────────────────


def test_species_style_exact_match():
    dove = species_style("Mourning Dove")
    assert dove["primary"] == "#a08b76"
    assert dove["mass_g"] > 0
    assert dove["call_hz"] > 0


def test_species_style_family_label_matches_whole_word():
    # The "Sparrow" family label and any member species share the family
    # plumage; a name that merely contains the letters must not.
    assert species_style("Sparrow")["primary"] == species_style("Field Sparrow")["primary"]
    assert species_style("Sparrowhawk-Not-Real")["primary"] != species_style("Sparrow")["primary"]


def test_species_style_unknown_bird_is_its_own_neutral():
    unknown = species_style(UNKNOWN_BIRD_LABEL)
    unlabelled = species_style(None)
    assert unknown["primary"] != unlabelled["primary"]


def test_species_style_fallback_is_hex_and_stable():
    # The canvas parses hex only. An hsl() string here collapses every
    # uncatalogued species onto one fallback colour.
    a = species_style("Roseate Spoonbill")
    b = species_style("Roseate Spoonbill")
    c = species_style("Snowy Egret")
    assert a["primary"].startswith("#") and len(a["primary"]) == 7
    assert a["accent"].startswith("#") and len(a["accent"]) == 7
    assert a == b
    assert a["primary"] != c["primary"]


# ── Flight paths ────────────────────────────────────────────────────────


def test_flight_points_uses_real_track_when_present():
    track = [
        [100, 1000, 60, 60],
        [800, 700, 60, 60],
        [1600, 500, 60, 60],
        [2400, 480, 60, 60],
    ]
    points, kind = flight_points([2400, 480, 60, 60], track, seed=7)

    assert kind == "tracked"
    # Catmull-Rom passes through its control points, so the resampled path
    # still starts and ends where the bird was actually detected.
    assert points[0]["x"] == pytest.approx(130 / FRAME_W, abs=1e-3)
    assert points[0]["y"] == pytest.approx(1030 / FRAME_H, abs=1e-3)
    assert points[-1]["x"] == pytest.approx(2430 / FRAME_W, abs=1e-3)


def test_flight_points_synthesises_from_a_single_bbox():
    points, kind = flight_points([1900, 1000, 120, 120], None, seed=11)

    assert kind == "synthesized"
    assert len(points) >= 20
    # The perch is real: the dwell phase sits on the detected centre.
    dwell = [p for p in points if 0.35 <= p["t"] <= 0.63]
    assert dwell
    assert all(abs(p["x"] - (1900 + 60) / FRAME_W) < 0.05 for p in dwell)
    # t runs the full path, non-decreasing.
    assert points[0]["t"] == 0.0
    assert points[-1]["t"] == pytest.approx(1.0, abs=1e-3)
    assert all(b["t"] >= a["t"] for a, b in zip(points, points[1:]))


def test_flight_points_synthesis_is_seeded():
    a, _ = flight_points([1900, 1000, 120, 120], None, seed=11)
    b, _ = flight_points([1900, 1000, 120, 120], None, seed=11)
    c, _ = flight_points([1900, 1000, 120, 120], None, seed=12)
    assert a == b
    assert a != c


def test_flight_points_survives_junk_boxes():
    # Malformed track entries are skipped rather than crashing the day.
    points, kind = flight_points([1900, 1000, 120, 120], [[1, 2], None, "x", [0, 0, 0, 0]], seed=3)
    assert kind == "synthesized"
    assert points


# ── /api/art/trajectories ───────────────────────────────────────────────


def test_each_detection_is_its_own_flight(client, db):
    # A visit's detections are separate tracks — usually separate birds.
    # Collapsing them into one path would draw a line between two birds.
    sp = _species(db, "Northern Cardinal")
    _detection(db, datetime(2026, 5, 20, 14, 0), sp, n=3)

    body = client.get("/api/art/trajectories?date=2026-05-20").json()
    assert len(body["flights"]) == 3
    assert {f["visit_id"] for f in body["flights"]} == {1}
    assert len({f["detection_id"] for f in body["flights"]}) == 3


def test_days_are_binned_in_the_cameras_local_zone(client, db):
    sp = _species(db, "Blue Jay")
    # 23:30 on May 20 in New York is 03:30 UTC on May 21. The bird visited
    # on the 20th; a UTC-date bin would file it under the 21st.
    _detection(db, datetime(2026, 5, 21, 3, 30), sp)

    on_20th = client.get("/api/art/trajectories?date=2026-05-20").json()
    on_21st = client.get("/api/art/trajectories?date=2026-05-21").json()

    assert len(on_20th["flights"]) == 1
    assert on_20th["flights"][0]["started_at"].startswith("2026-05-20T23:30")
    assert on_20th["flights"][0]["time_of_day"] == pytest.approx(23.5 / 24, abs=1e-3)
    assert on_21st["flights"] == []


def test_hidden_labels_are_excluded_but_unknown_bird_is_kept(client, db):
    real = _species(db, "Gray Catbird")
    nab = _species(db, NOT_A_BIRD_LABEL, "n/a")
    poor = _species(db, POOR_QUALITY_LABEL, "n/a")
    unknown = _species(db, UNKNOWN_BIRD_LABEL, "n/a")

    at = datetime(2026, 5, 20, 15, 0)
    for sp in (real, nab, poor, unknown):
        _detection(db, at, sp)

    body = client.get("/api/art/trajectories?date=2026-05-20").json()
    names = {f["species"] for f in body["flights"]}
    assert names == {"Gray Catbird", UNKNOWN_BIRD_LABEL}


def test_species_filter_and_unidentified_toggle(client, db):
    sp = _species(db, "House Finch")
    at = datetime(2026, 5, 20, 15, 0)
    _detection(db, at, sp)
    _detection(db, at, None)  # classifier rejected — no species row

    both = client.get("/api/art/trajectories?date=2026-05-20").json()
    assert len(both["flights"]) == 2
    assert "Unidentified" in both["summary"]["species_counts"]

    labelled = client.get(
        "/api/art/trajectories?date=2026-05-20&include_unidentified=false"
    ).json()
    assert [f["species"] for f in labelled["flights"]] == ["House Finch"]

    filtered = client.get(
        f"/api/art/trajectories?date=2026-05-20&species_id={sp.id}"
    ).json()
    assert [f["species"] for f in filtered["flights"]] == ["House Finch"]


def test_limit_samples_across_the_whole_day(client, db):
    sp = _species(db, "Rock Pigeon")
    # UTC 10:00-21:00 is local 06:00-17:00, a plausible feeder day.
    for hour in range(10, 22):
        _detection(db, datetime(2026, 5, 20, hour, 0), sp)

    body = client.get("/api/art/trajectories?date=2026-05-20&limit=4").json()
    times = [f["started_at"] for f in body["flights"]]

    assert body["summary"]["returned"] == 4
    assert body["summary"]["total_available"] == 12
    assert body["summary"]["truncated"] is True
    # First and last flight of the day survive the thinning, so the
    # artwork still spans dawn to dusk rather than stopping mid-morning.
    assert times[0].startswith("2026-05-20T06:00")
    assert times[-1].startswith("2026-05-20T17:00")


def test_path_kind_is_reported(client, db):
    sp = _species(db, "American Robin")
    at = datetime(2026, 5, 20, 15, 0)
    _detection(db, at, sp)
    _, made = _detection(db, at, sp)
    made[0].track_bboxes = [[100, 900, 60, 60], [900, 700, 60, 60], [1700, 600, 60, 60]]
    db.commit()

    body = client.get("/api/art/trajectories?date=2026-05-20").json()
    kinds = sorted(f["path_kind"] for f in body["flights"])
    assert kinds == ["synthesized", "tracked"]
    assert body["summary"]["tracked"] == 1


def test_response_carries_zone_and_sun_times(client, db):
    sp = _species(db, "Blue Jay")
    _detection(db, datetime(2026, 5, 20, 15, 0), sp)

    body = client.get("/api/art/trajectories?date=2026-05-20").json()
    assert body["tz"] == "America/New_York"
    assert 0.0 < body["sun"]["sunrise"] < body["sun"]["sunset"] < 1.0


def test_no_date_falls_back_to_the_most_recent_active_day(client, db):
    sp = _species(db, "Blue Jay")
    _detection(db, datetime(2026, 5, 18, 15, 0), sp)
    _detection(db, datetime(2026, 5, 22, 15, 0), sp)

    body = client.get("/api/art/trajectories").json()
    assert body["date"] == "2026-05-22"


def test_bad_date_is_rejected(client, db):
    assert client.get("/api/art/trajectories?date=last-tuesday").status_code == 400


def test_empty_database_returns_an_empty_day(client, db):
    body = client.get("/api/art/trajectories").json()
    assert body["flights"] == []
    assert body["summary"]["total_available"] == 0


# ── /api/art/dates ──────────────────────────────────────────────────────


def test_dates_groups_locally_with_counts_and_top_species(client, db):
    cardinal = _species(db, "Northern Cardinal")
    dove = _species(db, "Mourning Dove")
    nab = _species(db, NOT_A_BIRD_LABEL, "n/a")

    _detection(db, datetime(2026, 5, 20, 14, 0), cardinal, n=2)
    _detection(db, datetime(2026, 5, 20, 16, 0), dove)
    _detection(db, datetime(2026, 5, 20, 16, 5), nab)  # excluded
    # 02:00 UTC on the 21st is 22:00 local on the 20th.
    _detection(db, datetime(2026, 5, 21, 2, 0), dove)
    _detection(db, datetime(2026, 5, 22, 14, 0), cardinal)

    body = client.get("/api/art/dates").json()
    by_date = {d["date"]: d for d in body["dates"]}

    assert [d["date"] for d in body["dates"]] == ["2026-05-22", "2026-05-20"]
    assert by_date["2026-05-20"]["flight_count"] == 4
    assert by_date["2026-05-20"]["top_species"][0] == {"species": "Northern Cardinal", "count": 2}
    assert body["tz"] == "America/New_York"


def test_dates_limit(client, db):
    sp = _species(db, "Blue Jay")
    for day in range(1, 8):
        _detection(db, datetime(2026, 5, day, 15, 0), sp)

    body = client.get("/api/art/dates?limit=3").json()
    assert [d["date"] for d in body["dates"]] == ["2026-05-07", "2026-05-06", "2026-05-05"]


def test_chime_note_is_pentatonic_and_ordered():
    from pipeline.palette import chime_note

    # A dove's coo lands below a goldfinch's call, and both land on notes
    # of the same scale so simultaneous arrivals stay consonant.
    dove = chime_note(species_style("Mourning Dove")["call_hz"], 0)
    finch = chime_note(species_style("American Goldfinch")["call_hz"], 0)
    assert dove < finch

    scale = {0, 2, 4, 7, 9}
    for hz in (200, 480, 1600, 4200, 9000):
        semitones = 12 * math.log2(chime_note(hz, 0) / 261.63)
        assert round(semitones) % 12 in scale


def test_flight_carries_a_chime_frequency(client, db):
    sp = _species(db, "Northern Cardinal")
    _detection(db, datetime(2026, 5, 20, 15, 0), sp)

    body = client.get("/api/art/trajectories?date=2026-05-20").json()
    assert body["flights"][0]["style"]["chime_hz"] > 0
