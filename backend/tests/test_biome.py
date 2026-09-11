"""Tests for acoustic botany and the /api/biome garden endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (
    NOT_A_BIRD_LABEL,
    UNKNOWN_BIRD_LABEL,
    Detection,
    HaikuboxDetection,
    Species,
    Visit,
)
from db.session import Base, get_db
from main import app
from pipeline.biome import (
    ENERGY_SOURCE_CALLS,
    ENERGY_SOURCE_CALLS_SCORE,
    FORMS,
    MAX_SYMBOLS,
    depth_for,
    expanded_length,
    pitch_position,
    plant,
    plant_form,
    vitality,
)
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
    # Pin the feeder's zone. Every date assertion below turns on the
    # UTC-to-local shift, and in late May New York is four hours behind.
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


def _heard(db, name, when, n=1, confidence=None, step_minutes=1):
    for i in range(n):
        db.add(
            HaikuboxDetection(
                species_common_name=name,
                detected_at=when + timedelta(minutes=i * step_minutes),
                confidence=confidence,
            )
        )
    db.commit()


def _seen(db, name, started_at, *, audio_confirmed=0, sci="Testus birdus"):
    sp = db.query(Species).filter_by(common_name=name).first()
    if sp is None:
        sp = Species(common_name=name, scientific_name=sci)
        db.add(sp)
        db.commit()
    visit = Visit(started_at=started_at, ended_at=started_at + timedelta(seconds=8))
    db.add(visit)
    db.commit()
    d = Detection(
        visit_id=visit.id,
        species_id=sp.id,
        confidence=0.9,
        audio_confirmed=audio_confirmed,
        crop_path=f"crops/{visit.id}.jpg",
        bbox=[1800, 900, 150, 150],
        track_id=1,
    )
    db.add(d)
    db.commit()
    return d


# ── Botany ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hz,form",
    [
        (400, "moss"),      # Mourning Dove
        (1199, "moss"),
        (1200, "spire"),    # band edges are inclusive at the bottom
        (1600, "spire"),    # Blue Jay
        (2200, "vine"),     # Northern Cardinal
        (3300, "frond"),    # House Sparrow
        (3699, "frond"),
        (3800, "floret"),   # Black-capped Chickadee, just inside
        (8500, "floret"),   # Blackpoll Warbler
    ],
)
def test_pitch_picks_the_form(hz, form):
    assert plant_form(hz)["id"] == form


def test_pitch_position_is_monotonic_and_clamped():
    assert pitch_position(100) == 0.0       # below the floor
    assert pitch_position(20000) == 1.0     # above the ceiling
    rising = [pitch_position(hz) for hz in (500, 1000, 2000, 4000, 8000)]
    assert rising == sorted(rising)
    # Logarithmic, so every octave is one equal step of the ramp.
    steps = [rising[i + 1] - rising[i] for i in range(4)]
    assert max(steps) - min(steps) < 1e-9


def test_depth_grows_with_calls():
    assert depth_for(1) == 2
    assert depth_for(5) == 3
    assert depth_for(40) == 4
    assert depth_for(4000) == 5


def test_vitality_says_which_inputs_it_had():
    quiet, source = vitality(1, 500, 0.0)
    assert source == ENERGY_SOURCE_CALLS
    loud, _ = vitality(500, 500, 0.9)
    assert loud > quiet

    # A BirdNET score, if one ever arrives, changes both the value and the
    # label. Nothing in the cache carries one today.
    scored, source = vitality(500, 500, 0.9, 0.85)
    assert source == ENERGY_SOURCE_CALLS_SCORE
    assert scored != loud


def test_vitality_is_log_scaled_not_linear():
    """One species out-calling the yard forty to one must not flatten it."""
    small, _ = vitality(10, 700, 0.5)
    assert small > 0.4  # a linear share would put this near 0.01


def test_expanded_length_matches_a_real_expansion():
    for form in FORMS.values():
        s = form["axiom"]
        for depth in range(1, 4):
            s = "".join(form["rules"].get(ch, ch) for ch in s)
            assert expanded_length(form["axiom"], form["rules"], depth) == len(s)


def test_every_plant_stays_under_the_symbol_cap():
    """The cap is what keeps a cushion moss from outgrowing the tab."""
    for name in ("Mourning Dove", "Blue Jay", "Northern Cardinal", "House Sparrow"):
        p = plant(name, calls=9000, busiest=9000, spread=1.0)
        assert expanded_length(p["axiom"], p["rules"], p["depth"]) <= MAX_SYMBOLS


def test_plant_is_deterministic():
    a = plant("House Sparrow", calls=40, busiest=100, spread=0.5)
    b = plant("House Sparrow", calls=40, busiest=100, spread=0.5)
    assert a == b


def test_low_callers_bloom_dark_and_high_ones_pale():
    dove = plant("Mourning Dove", calls=10, busiest=10, spread=0.5)
    warbler = plant("Blackpoll Warbler", calls=10, busiest=10, spread=0.5)
    assert dove["bloom"]["hue"] < warbler["bloom"]["hue"]
    assert dove["bloom"]["light"] < warbler["bloom"]["light"]
    # A 120 g dove grows a stouter stem than a 12 g warbler whatever else
    # they have in common.
    assert dove["girth"] > warbler["girth"]


def test_audio_regulars_have_real_palette_entries():
    """Species the box hears constantly should not fall back to a hash.

    Chimney Swift is the second-loudest voice in this yard's audio cache
    and the camera has never caught one, so the palette is the only place
    its pitch can come from.
    """
    for name in ("Chimney Swift", "Northern Parula", "Cedar Waxwing", "Fish Crow"):
        p = plant(name, calls=10, busiest=10, spread=0.5)
        assert p["form"] == plant_form(p["call_hz"])["id"]
    assert plant("Fish Crow", calls=1, busiest=1, spread=0.0)["form"] == "moss"
    assert plant("Chimney Swift", calls=1, busiest=1, spread=0.0)["form"] == "floret"


# ── The garden endpoint ─────────────────────────────────────────────────


def test_garden_groups_by_species_not_by_call(client, db):
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 14, 0), n=30)
    _heard(db, "Blue Jay", datetime(2026, 5, 20, 15, 0), n=3)

    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    assert data["summary"]["calls"] == 33
    assert data["summary"]["species_heard"] == 2
    assert len(data["plants"]) == 2

    sparrow = next(p for p in data["plants"] if p["species"] == "House Sparrow")
    assert sparrow["calls"] == 30
    assert sparrow["share"] == pytest.approx(30 / 33, abs=1e-3)
    # The busiest species anchors the density scale, so it is the liveliest.
    jay = next(p for p in data["plants"] if p["species"] == "Blue Jay")
    assert sparrow["energy"] > jay["energy"]


def test_garden_bins_by_camera_local_day(client, db):
    # 02:30 UTC on the 21st is 22:30 on the 20th in New York, so this call
    # belongs to the 20th's garden and not the 21st's.
    _heard(db, "Barred Owl", datetime(2026, 5, 21, 2, 30))
    _heard(db, "House Sparrow", datetime(2026, 5, 21, 14, 0))

    twentieth = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    assert [p["species"] for p in twentieth["plants"]] == ["Barred Owl"]
    assert twentieth["plants"][0]["hours"][22] == 1

    twenty_first = client.get("/api/biome/garden", params={"date": "2026-05-21"}).json()
    assert [p["species"] for p in twenty_first["plants"]] == ["House Sparrow"]


def test_garden_is_ordered_low_pitch_to_high(client, db):
    for name in ("Black-capped Chickadee", "Mourning Dove", "Northern Cardinal"):
        _heard(db, name, datetime(2026, 5, 20, 14, 0), n=4)

    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    pitches = [p["pitch"] for p in data["plants"]]
    assert pitches == sorted(pitches)
    assert data["plants"][0]["species"] == "Mourning Dove"


def test_garden_reports_the_chorus_and_its_peak(client, db):
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 10, 0), n=5)   # 06:00 local
    _heard(db, "American Robin", datetime(2026, 5, 20, 22, 0), n=2)  # 18:00 local

    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    assert sum(data["chorus"]) == 7
    assert data["chorus"][6] == 5
    assert data["chorus"][18] == 2
    assert data["summary"]["peak_hour"] == 6


def test_min_calls_and_limit_trim_the_garden(client, db):
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 14, 0), n=20)
    _heard(db, "Blue Jay", datetime(2026, 5, 20, 15, 0), n=1)

    trimmed = client.get(
        "/api/biome/garden", params={"date": "2026-05-20", "min_calls": 5}
    ).json()
    assert [p["species"] for p in trimmed["plants"]] == ["House Sparrow"]
    # A trimmed plant is still part of the day's totals: it happened.
    assert trimmed["summary"]["calls"] == 21
    assert trimmed["summary"]["truncated"] is True

    capped = client.get(
        "/api/biome/garden", params={"date": "2026-05-20", "limit": 1}
    ).json()
    assert [p["species"] for p in capped["plants"]] == ["House Sparrow"]


def test_pollinators_prefer_the_confirmed_sighting(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    _heard(db, "Northern Cardinal", when, n=4)
    _heard(db, "Gray Catbird", when, n=4)
    _seen(db, "Northern Cardinal", when, audio_confirmed=1)
    _seen(db, "Gray Catbird", when, audio_confirmed=0)

    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    by_species = {v["species"]: v for v in data["pollinators"]}
    assert by_species["Northern Cardinal"]["confirmed"] is True
    assert by_species["Gray Catbird"]["confirmed"] is False

    plants = {p["species"]: p for p in data["plants"]}
    assert plants["Northern Cardinal"]["confirmed"] is True
    assert plants["Gray Catbird"]["flowering"] is True
    assert plants["Gray Catbird"]["confirmed"] is False


def test_a_bird_only_heard_never_flowers(client, db):
    _heard(db, "Chimney Swift", datetime(2026, 5, 20, 14, 0), n=9)
    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    assert data["pollinators"] == []
    assert data["plants"][0]["flowering"] is False


def test_sentinels_cannot_pollinate(client, db):
    when = datetime(2026, 5, 20, 14, 0)
    _heard(db, NOT_A_BIRD_LABEL, when, n=2)
    _heard(db, UNKNOWN_BIRD_LABEL, when, n=2)
    _seen(db, NOT_A_BIRD_LABEL, when)
    _seen(db, UNKNOWN_BIRD_LABEL, when)

    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    assert data["pollinators"] == []


def test_default_date_follows_the_microphone_not_the_camera(client, db):
    """The box can be offline for days while the feeder stays busy."""
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 14, 0), n=3)
    _seen(db, "Mourning Dove", datetime(2026, 5, 25, 14, 0))

    data = client.get("/api/biome/garden").json()
    assert data["date"] == "2026-05-20"


def test_a_silent_day_is_an_empty_garden(client, db):
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 14, 0), n=3)
    data = client.get("/api/biome/garden", params={"date": "2026-05-21"}).json()
    assert data["plants"] == []
    assert data["summary"]["quiet"] is True
    assert data["summary"]["peak_hour"] is None
    assert data["summary"]["energy_source"] is None


def test_garden_rejects_a_bad_date(client):
    assert client.get("/api/biome/garden", params={"date": "last tuesday"}).status_code == 400


def test_energy_source_names_what_was_measured(client, db):
    """No score in the cache means the page must not imply one."""
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 14, 0), n=3)
    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    assert data["summary"]["energy_source"] == ENERGY_SOURCE_CALLS
    assert data["plants"][0]["mean_confidence"] is None

    _heard(db, "Blue Jay", datetime(2026, 5, 20, 15, 0), n=3, confidence=0.8)
    data = client.get("/api/biome/garden", params={"date": "2026-05-20"}).json()
    jay = next(p for p in data["plants"] if p["species"] == "Blue Jay")
    assert jay["mean_confidence"] == pytest.approx(0.8)
    assert jay["energy_source"] == ENERGY_SOURCE_CALLS_SCORE


# ── The dates endpoint ──────────────────────────────────────────────────


def test_dates_lists_days_with_audio_newest_first(client, db):
    _heard(db, "House Sparrow", datetime(2026, 5, 20, 14, 0), n=4)
    _heard(db, "Blue Jay", datetime(2026, 5, 20, 15, 0), n=1)
    _heard(db, "American Robin", datetime(2026, 5, 22, 14, 0), n=2)

    data = client.get("/api/biome/dates").json()
    assert [d["date"] for d in data["dates"]] == ["2026-05-22", "2026-05-20"]
    twentieth = data["dates"][1]
    assert twentieth["call_count"] == 5
    assert twentieth["species_count"] == 2
    assert twentieth["top_species"][0] == {"species": "House Sparrow", "count": 4}


def test_dates_is_empty_when_the_box_has_never_reported(client):
    assert client.get("/api/biome/dates").json()["dates"] == []
