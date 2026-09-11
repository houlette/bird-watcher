"""Tests for zone control, the pecking order, and /api/territory."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import NOT_A_BIRD_LABEL, UNKNOWN_BIRD_LABEL, Detection, Species, Visit
from db.session import Base, get_db
from main import app
from pipeline import territory
from pipeline.palette import FRAME_H, FRAME_W
from pipeline.territory import (
    DEFAULT_ZONES,
    DISPLACE_WINDOW_FRAMES,
    FRAME_SECONDS,
    displacements,
    dwell,
    faction_for,
    load_zones,
    zone_for,
    zone_spans,
)
from settings import settings

ZONES = DEFAULT_ZONES


def _box_at(zone_id: str, nudge: float = 0.0) -> list[int]:
    """A 120 px box centred on a zone, optionally nudged right."""
    z = next(x for x in ZONES if x["id"] == zone_id)
    cx = (z["x"] + nudge) * FRAME_W
    cy = z["y"] * FRAME_H
    return [int(cx - 60), int(cy - 60), 120, 120]


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


def _detect(db, started_at, name=None, *, zone="dish", frames=None, boxes=None, visit=None):
    """One detection in one zone, optionally with a per-frame clock."""
    if visit is None:
        visit = Visit(started_at=started_at, ended_at=started_at + timedelta(seconds=8))
        db.add(visit)
        db.commit()

    species = None
    if name:
        species = db.query(Species).filter_by(common_name=name).first()
        if species is None:
            species = Species(common_name=name, scientific_name="Testus birdus")
            db.add(species)
            db.commit()

    box = _box_at(zone)
    if boxes is None and frames is not None:
        boxes = [box] * len(frames)

    d = Detection(
        visit_id=visit.id,
        species_id=species.id if species else None,
        confidence=0.9,
        crop_path=f"crops/{visit.id}.jpg",
        bbox=box,
        track_bboxes=boxes,
        track_frames=frames,
        track_id=1,
    )
    db.add(d)
    db.commit()
    return visit, d


# ── The map ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("zone_id", [z["id"] for z in DEFAULT_ZONES])
def test_every_zone_claims_its_own_centre(zone_id):
    assert zone_for(_box_at(zone_id), ZONES) == zone_id


def test_open_sky_belongs_to_no_zone():
    assert zone_for([0, 0, 100, 100], ZONES) is None


def test_malformed_boxes_do_not_raise():
    for bad in (None, [], [1, 2], ["a", "b", "c", "d"], [0, 0, 0, 0], [0, 0, -5, 5]):
        assert zone_for(bad, ZONES) is None


def test_zones_fall_back_to_the_built_in_map(monkeypatch, tmp_path):
    territory.reset_zone_cache_for_tests()
    monkeypatch.setattr(territory, "ZONES_PATH", tmp_path / "absent.json")
    zones, source = load_zones()
    assert source == "built-in"
    assert [z["id"] for z in zones] == [z["id"] for z in DEFAULT_ZONES]


def test_a_calibrated_map_replaces_the_built_in_one(monkeypatch, tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps({"zones": [{"id": "porch", "name": "The Porch", "x": 0.2, "y": 0.2, "radius": 0.1}]})
    )
    territory.reset_zone_cache_for_tests()
    monkeypatch.setattr(territory, "ZONES_PATH", path)
    zones, source = load_zones()
    assert source == "calibrated"
    assert [z["id"] for z in zones] == ["porch"]
    territory.reset_zone_cache_for_tests()


def test_a_corrupt_map_falls_back_rather_than_raising(monkeypatch, tmp_path):
    path = tmp_path / "zones.json"
    path.write_text("{not json")
    territory.reset_zone_cache_for_tests()
    monkeypatch.setattr(territory, "ZONES_PATH", path)
    zones, source = load_zones()
    assert source == "built-in"
    assert zones
    territory.reset_zone_cache_for_tests()


# ── Factions ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "species,faction",
    [
        ("Blue Jay", "syndicate"),
        ("Common Grackle", "syndicate"),
        ("Mourning Dove", "order"),
        ("Rock Pigeon", "order"),
        ("Northern Cardinal", "court"),
        ("Gray Catbird", "chorus"),
        ("Downy Woodpecker", "guild"),
        ("House Sparrow", "coalition"),
        ("Black-capped Chickadee", "scouts"),
    ],
)
def test_the_yards_regulars_have_a_faction(species, faction):
    assert faction_for(species) == faction


def test_family_labels_find_a_faction():
    """"Sparrow" and "Woodpecker" are real rows, not species."""
    assert faction_for("Sparrow") == "coalition"
    assert faction_for("Woodpecker") == "guild"


def test_an_unnamed_bird_fights_for_nobody():
    assert faction_for(None) is None
    assert faction_for("") is None


def test_the_big_odd_visitors_ride_alone():
    """Raptors and waterfowl are not feeder birds and get their own side."""
    assert faction_for("Cooper's Hawk") == "outriders"
    assert faction_for("Mallard") == "outriders"
    assert faction_for("Hawk") == "outriders"           # the family label
    assert faction_for("American Robin") == "chorus"


def test_every_named_bird_gets_a_faction_and_keeps_it():
    """The fallback has to be total, or species quietly stop being scored.

    Uncatalogued names take their mass from the palette's hash, which only
    ever returns 15 to 79 g, so they land among the lighter factions. The
    point of the assertion is that they land somewhere at all, and land
    there again on the next request.
    """
    for name in ("Greater Rhea", "Zigzag Heron", "Spangled Cotinga"):
        first = faction_for(name)
        assert first in territory.FACTIONS
        assert faction_for(name) == first


# ── Dwell ───────────────────────────────────────────────────────────────


def test_dwell_measured_spans_gaps_in_the_track():
    """The IoU matcher drops frames; the clock still knows how long it was."""
    seconds, kind = dwell([[0, 0, 10, 10]] * 4, [0, 1, 5, 6])
    assert kind == "measured"
    assert seconds == pytest.approx(7 * FRAME_SECONDS, abs=0.01)


def test_dwell_estimated_undercounts_but_says_so():
    seconds, kind = dwell([[0, 0, 10, 10]] * 4, None)
    assert kind == "estimated"
    assert seconds == pytest.approx(4 * FRAME_SECONDS, abs=0.01)


def test_dwell_unknown_for_a_row_with_no_track_history():
    assert dwell(None, None) == (0.0, "unknown")
    assert dwell([[0, 0, 10, 10]], None)[1] == "unknown"


def test_mismatched_track_columns_do_not_claim_a_measurement():
    """Lengths must agree or the boxes and the clock are not the same track."""
    assert dwell([[0, 0, 10, 10]] * 4, [0, 1])[1] == "estimated"


# ── Holding ground ──────────────────────────────────────────────────────


def test_zone_spans_needs_both_columns():
    boxes = [_box_at("dish")] * 4
    assert zone_spans(boxes, None, ZONES) == {}
    assert zone_spans(None, [0, 1, 2, 3], ZONES) == {}
    assert zone_spans(boxes, [0, 1, 2, 3], ZONES) == {"dish": (0, 3, 4)}


def test_passing_through_is_not_holding():
    """One frame in a zone is flight, not occupancy."""
    boxes = [_box_at("dish")] + [_box_at("rail")] * 4
    spans = zone_spans(boxes, [0, 1, 2, 3, 4], ZONES)
    assert "dish" not in spans
    assert spans["rail"] == (1, 4, 4)


# ── The pecking order ───────────────────────────────────────────────────


def _contender(did, species, frames, zone="dish"):
    return {
        "detection_id": did,
        "species": species,
        "track_bboxes": [_box_at(zone)] * len(frames),
        "track_frames": frames,
    }


def test_a_newcomer_that_clears_the_perch_is_a_displacement():
    dove = _contender(1, "Mourning Dove", [0, 1, 2, 3, 4])
    jay = _contender(2, "Blue Jay", [4, 5, 6, 7])

    events = displacements([dove, jay], ZONES)
    assert len(events) == 1
    e = events[0]
    assert e["winner"] == "Blue Jay" and e["winner_faction"] == "syndicate"
    assert e["loser"] == "Mourning Dove" and e["loser_faction"] == "order"
    assert e["zone"] == "dish"
    assert e["at_frame"] == 4
    assert e["at_seconds"] == pytest.approx(4 * FRAME_SECONDS, abs=0.01)


def test_sharing_a_feeder_is_not_a_displacement():
    """Both birds stay for ages; nobody was pushed off anything."""
    a = _contender(1, "Mourning Dove", list(range(0, 20)))
    b = _contender(2, "Blue Jay", list(range(5, 20)))
    assert displacements([a, b], ZONES) == []


def test_the_bird_that_was_there_first_cannot_be_the_winner():
    a = _contender(1, "Mourning Dove", [0, 1, 2, 3])
    b = _contender(2, "Blue Jay", [10, 11, 12, 13])
    # B arrives long after A has gone, so nobody displaced anybody.
    assert displacements([a, b], ZONES) == []


def test_leaving_well_after_the_newcomer_lands_is_not_a_displacement():
    late = DISPLACE_WINDOW_FRAMES + 2
    a = _contender(1, "Mourning Dove", list(range(0, 5 + late)))
    b = _contender(2, "Blue Jay", list(range(5, 12)))
    assert displacements([a, b], ZONES) == []


def test_birds_in_different_zones_never_contest():
    a = _contender(1, "Mourning Dove", [0, 1, 2, 3, 4], zone="dish")
    b = _contender(2, "Blue Jay", [4, 5, 6, 7], zone="rail")
    assert displacements([a, b], ZONES) == []


def test_a_track_with_no_clock_can_never_win_or_lose_a_perch():
    """This is the whole reason track_frames was added."""
    a = {"detection_id": 1, "species": "Mourning Dove", "track_bboxes": [_box_at("dish")] * 5, "track_frames": None}
    b = _contender(2, "Blue Jay", [4, 5, 6, 7])
    assert displacements([a, b], ZONES) == []


def test_an_unnamed_bird_can_still_take_a_perch():
    """It holds ground for no faction, but the event is real footage."""
    a = _contender(1, None, [0, 1, 2, 3, 4])
    b = _contender(2, None, [4, 5, 6, 7])
    events = displacements([a, b], ZONES)
    assert len(events) == 1
    assert events[0]["winner_faction"] is None
    assert events[0]["loser_faction"] is None


# ── The endpoint ────────────────────────────────────────────────────────


def test_day_scores_zones_from_where_birds_actually_were(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    _detect(db, when, "Mourning Dove", zone="ground")
    _detect(db, when + timedelta(minutes=1), "Mourning Dove", zone="ground")
    _detect(db, when + timedelta(minutes=2), "Blue Jay", zone="dish")

    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    zones = {z["id"]: z for z in data["zones"]}
    assert zones["ground"]["holder"] == "order"
    assert zones["dish"]["holder"] == "syndicate"
    assert zones["cage"]["holder"] is None
    assert data["summary"]["detections"] == 3


def test_control_falls_back_to_appearances_without_a_clock(client, db):
    """An archive day has no dwell, so seconds would rank everyone at zero."""
    when = datetime(2026, 5, 20, 14, 0)
    for _ in range(3):
        _detect(db, when, "Mourning Dove", zone="dish")
    _detect(db, when, "Blue Jay", zone="dish")

    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    assert data["control_basis"] == "appearances"
    dish = next(z for z in data["zones"] if z["id"] == "dish")
    assert dish["holder"] == "order"
    assert dish["control"][0]["holds"] == 3
    assert dish["seconds"] == 0.0


def test_control_uses_seconds_once_tracks_carry_one(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    # One long-staying jay outweighs two brief doves on time, and loses on
    # head count, which is what makes this the interesting case.
    _detect(db, when, "Blue Jay", zone="dish", frames=list(range(0, 25)))
    _detect(db, when + timedelta(minutes=1), "Mourning Dove", zone="dish", frames=[0, 1, 2])
    _detect(db, when + timedelta(minutes=2), "Mourning Dove", zone="dish", frames=[0, 1, 2])

    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    assert data["control_basis"] == "seconds"
    dish = next(z for z in data["zones"] if z["id"] == "dish")
    assert dish["holder"] == "syndicate"
    assert data["summary"]["timed"] == 3


def test_a_few_timed_birds_do_not_decide_the_whole_day(client, db):
    """The transitional case, and the one that quietly erases zones.

    With seconds as the measure, every untimed bird scores zero. A perch
    with two real landings would read as "nobody held this" while a perch
    with one timed visitor took the day. Control stays on appearances
    until timings actually cover the day.
    """
    when = datetime(2026, 5, 20, 14, 0)
    _detect(db, when, "Blue Jay", zone="dish", frames=list(range(0, 20)))
    for i in range(5):
        _detect(db, when + timedelta(minutes=i + 1), "Mourning Dove", zone="rail")

    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    assert data["control_basis"] == "appearances"
    rail = next(z for z in data["zones"] if z["id"] == "rail")
    assert rail["holder"] == "order", "a zone with real landings must not vanish"
    assert rail["control"][0]["holds"] == 5


def test_unidentified_birds_are_unclaimed_not_assigned(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    _detect(db, when, None, zone="dish")
    _detect(db, when + timedelta(minutes=1), None, zone="dish")
    _detect(db, when + timedelta(minutes=2), "Blue Jay", zone="dish")

    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    dish = next(z for z in data["zones"] if z["id"] == "dish")
    assert dish["unclaimed"] == 2
    assert dish["visits"] == 3
    assert [c["faction"] for c in dish["control"]] == ["syndicate"]
    assert data["summary"]["unclaimed"] == 2


def test_a_confirmed_bird_without_a_species_still_holds_ground(client, db):
    """"Unknown bird" is a human confirming a real bird sat there."""
    when = datetime(2026, 5, 20, 14, 0)
    _detect(db, when, UNKNOWN_BIRD_LABEL, zone="rail")
    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    rail = next(z for z in data["zones"] if z["id"] == "rail")
    assert rail["visits"] == 1


def test_not_a_bird_holds_nothing(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    _detect(db, when, NOT_A_BIRD_LABEL, zone="dish")
    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    assert data["summary"]["detections"] == 0
    assert all(z["visits"] == 0 for z in data["zones"])


def test_day_reports_how_much_could_be_judged(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    visit = Visit(started_at=when, ended_at=when + timedelta(seconds=8))
    db.add(visit)
    db.commit()
    _detect(db, when, "Mourning Dove", zone="dish", frames=[0, 1, 2, 3, 4], visit=visit)
    _detect(db, when, "Blue Jay", zone="dish", frames=[4, 5, 6, 7], visit=visit)
    # A second visit with two birds but no clock: contested, never eligible.
    other = Visit(started_at=when + timedelta(minutes=5), ended_at=when + timedelta(minutes=5, seconds=8))
    db.add(other)
    db.commit()
    _detect(db, when, "Mourning Dove", zone="dish", visit=other)
    _detect(db, when, "Blue Jay", zone="dish", visit=other)

    data = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    s = data["summary"]
    assert s["contested_visits"] == 2
    assert s["eligible_visits"] == 1
    assert len(data["displacements"]) == 1
    e = data["displacements"][0]
    assert e["winner"] == "Blue Jay"
    assert e["zone_name"] == "The Dish"
    assert e["winner_faction_name"] == "The Jay Syndicate"


def test_day_bins_by_camera_local_day(client, db):
    # 02:30 UTC on the 21st is 22:30 on the 20th in New York.
    _detect(db, datetime(2026, 5, 21, 2, 30), "Blue Jay", zone="dish")
    _detect(db, datetime(2026, 5, 21, 14, 0), "Mourning Dove", zone="dish")

    twentieth = client.get("/api/territory/day", params={"date": "2026-05-20"}).json()
    assert twentieth["summary"]["detections"] == 1
    assert next(z for z in twentieth["zones"] if z["id"] == "dish")["holder"] == "syndicate"

    twenty_first = client.get("/api/territory/day", params={"date": "2026-05-21"}).json()
    assert next(z for z in twenty_first["zones"] if z["id"] == "dish")["holder"] == "order"


def test_a_quiet_day_does_not_invent_a_battle(client, db):
    _detect(db, datetime(2026, 5, 20, 14, 0), "Blue Jay", zone="dish")
    data = client.get("/api/territory/day", params={"date": "2026-05-21"}).json()
    assert data["summary"]["quiet"] is True
    assert data["standings"] == []
    assert "Nothing held the yard today" in data["dispatch"]


def test_day_rejects_a_bad_date(client):
    assert client.get("/api/territory/day", params={"date": "yesterday"}).status_code == 400


def test_dates_reports_which_days_carry_a_clock(client, db):
    _detect(db, datetime(2026, 5, 20, 14, 0), "Blue Jay", zone="dish", frames=[0, 1, 2])
    _detect(db, datetime(2026, 5, 20, 15, 0), "Mourning Dove", zone="dish")
    _detect(db, datetime(2026, 5, 22, 14, 0), "Mourning Dove", zone="dish")

    data = client.get("/api/territory/dates").json()
    assert [d["date"] for d in data["dates"]] == ["2026-05-22", "2026-05-20"]
    twentieth = data["dates"][1]
    assert twentieth["detection_count"] == 2
    assert twentieth["timed_count"] == 1
