"""Tests for the tavern rules and the /api/tavern endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from pipeline.tavern import (
    COVER_CHARGE,
    FOUNDING_PURSE_CAP,
    STRANGER_PAYMENT,
    UPGRADES_BY_ID,
    archetype_for,
    arrival_line,
    dwell_beats,
    lore_for,
    order_for,
    patron_payment,
    rarity_tier,
    rarity_tiers,
    spent_on,
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


def _arrive(db, minutes_ago=5, species=None, n=1, seconds=8, audio=0):
    """Insert a visit `minutes_ago` minutes back, with `n` detections."""
    started = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=minutes_ago)
    visit = Visit(started_at=started, ended_at=started + timedelta(seconds=seconds))
    db.add(visit)
    db.commit()
    made = []
    for i in range(n):
        d = Detection(
            visit_id=visit.id,
            species_id=species.id if species else None,
            confidence=0.9,
            audio_confirmed=audio,
            crop_path=f"crops/{visit.id}_{i}.jpg",
            bbox=[1800, 900, 150, 150],
            track_id=i + 1,
        )
        db.add(d)
        made.append(d)
    db.commit()
    return visit, made


# ── Archetypes ──────────────────────────────────────────────────────────


def test_archetype_explicit_table_wins():
    assert archetype_for("Blue Jay")["id"] == "adventurer"
    assert archetype_for("Mourning Dove")["id"] == "local"
    assert archetype_for("Black-capped Chickadee")["id"] == "messenger"
    assert archetype_for("Cooper's Hawk")["id"] == "hunter"


def test_archetype_unlabelled_bird_is_a_stranger():
    # The classifier declining to commit is not a sighting of anything, and
    # the room should say so rather than inventing a species for it.
    assert archetype_for(None)["id"] == "stranger"
    assert archetype_for("")["id"] == "stranger"


def test_archetype_unknown_bird_label_is_not_a_stranger():
    # "Unknown bird" is a human confirming a bird without naming it, which
    # is a real guest, unlike a NULL species.
    assert archetype_for(UNKNOWN_BIRD_LABEL)["id"] != "stranger"


def test_archetype_family_word_matches_whole_word_only():
    assert archetype_for("Field Sparrow")["id"] == "local"
    assert archetype_for("Pine Warbler")["id"] == "minstrel"
    # Mass fallback, not the "Hawk" family: the word has to stand alone.
    assert archetype_for("Sparrowhawk-Not-Real", mass_g=30.0)["id"] != "hunter"


def test_archetype_mass_fallback_for_uncatalogued_species():
    assert archetype_for("Invented Greatbird", mass_g=900.0)["id"] == "wanderer"
    assert archetype_for("Invented Middlebird", mass_g=70.0)["id"] == "adventurer"
    assert archetype_for("Invented Smallbird", mass_g=9.0)["id"] == "messenger"
    assert archetype_for("Invented Midling", mass_g=30.0)["id"] == "local"


# ── Rarity and payment ──────────────────────────────────────────────────


def test_rarity_tier_bands():
    assert rarity_tier(0.4) == "common"
    assert rarity_tier(0.03) == "regular"
    assert rarity_tier(0.01) == "uncommon"
    assert rarity_tier(0.0001) == "rare"


def test_rarity_tiers_handles_an_empty_yard():
    assert rarity_tiers({}) == {}
    assert rarity_tiers({"Ovenbird": 0})["Ovenbird"] == "rare"


def test_stranger_pays_a_flat_shilling_however_long_they_stay():
    # Otherwise the unlabelled crops, which outnumber named birds several
    # to one, would buy every upgrade in the catalogue on their own.
    long_stay = patron_payment(
        archetype_id="stranger", tier="rare", duration_seconds=600, audio_confirmed=True
    )
    assert long_stay == STRANGER_PAYMENT


def test_payment_rises_with_dwell_then_stops():
    quick = patron_payment(
        archetype_id="local", tier="common", duration_seconds=0, audio_confirmed=False
    )
    longer = patron_payment(
        archetype_id="local", tier="common", duration_seconds=45, audio_confirmed=False
    )
    capped = patron_payment(
        archetype_id="local", tier="common", duration_seconds=6000, audio_confirmed=False
    )
    forever = patron_payment(
        archetype_id="local", tier="common", duration_seconds=99999, audio_confirmed=False
    )
    assert quick == COVER_CHARGE
    assert longer > quick
    assert capped == forever


def test_payment_rewards_rarity_and_being_heard():
    common = patron_payment(
        archetype_id="local", tier="common", duration_seconds=10, audio_confirmed=False
    )
    rare = patron_payment(
        archetype_id="local", tier="rare", duration_seconds=10, audio_confirmed=False
    )
    heard = patron_payment(
        archetype_id="local", tier="common", duration_seconds=10, audio_confirmed=True
    )
    assert rare > common
    assert heard > common


def test_spent_is_derived_and_survives_a_retired_upgrade_id():
    assert spent_on([]) == 0
    assert spent_on(["tallow_candles"]) == UPGRADES_BY_ID["tallow_candles"]["cost"]
    assert spent_on(["tallow_candles", "no_such_thing"]) == UPGRADES_BY_ID["tallow_candles"]["cost"]


# ── Flavour determinism ─────────────────────────────────────────────────


def test_flavour_is_stable_per_detection():
    assert arrival_line("local", 42) == arrival_line("local", 42)
    assert order_for("adventurer", 7) == order_for("adventurer", 7)
    assert lore_for("Gray Catbird", "minstrel") == lore_for("Gray Catbird", "minstrel")


def test_dwell_scales_with_the_real_visit_but_keeps_the_archetype_order():
    short_visit = dwell_beats("local", 2)
    long_visit = dwell_beats("local", 90)
    assert long_visit > short_visit
    # A dove that sat briefly still outstays a chickadee that sat a while.
    assert dwell_beats("local", 0) > dwell_beats("messenger", 90)


# ── /state ──────────────────────────────────────────────────────────────


def test_state_seats_recent_arrivals(client, db):
    jay = _species(db, "Blue Jay", "Cyanocitta cristata")
    _arrive(db, minutes_ago=5, species=jay)

    r = client.get("/api/tavern/state")
    assert r.status_code == 200
    body = r.json()

    assert body["quiet"] is False
    assert len(body["patrons"]) == 1
    patron = body["patrons"][0]
    assert patron["species"] == "Blue Jay"
    assert patron["archetype"]["id"] == "adventurer"
    assert patron["payment"] > 0
    assert patron["stale"] is False
    assert patron["line"]
    assert patron["order"]["drink"]


def test_state_falls_back_to_the_last_company_and_says_so(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=60 * 30, species=dove)

    body = client.get("/api/tavern/state?window_minutes=60").json()
    assert body["quiet"] is True
    assert len(body["patrons"]) == 1
    assert body["patrons"][0]["stale"] is True


def test_state_turns_not_a_bird_into_a_mishap_not_a_guest(client, db):
    nab = _species(db, NOT_A_BIRD_LABEL, "n/a")
    _arrive(db, minutes_ago=3, species=nab)

    body = client.get("/api/tavern/state").json()
    assert body["patrons"] == []
    assert len(body["events"]) == 1
    assert body["events"][0]["title"]
    assert body["events"][0]["line"]


def test_state_never_seats_a_retired_crop(client, db):
    poor = _species(db, POOR_QUALITY_LABEL, "n/a")
    _arrive(db, minutes_ago=3, species=poor)

    body = client.get("/api/tavern/state").json()
    assert body["patrons"] == []
    assert body["events"] == []
    assert body["ledger"]["patrons_served"] == 0


def test_hiding_strangers_empties_the_room_but_not_the_till(client, db):
    _arrive(db, minutes_ago=4, n=3)  # three unlabelled crops

    shown = client.get("/api/tavern/state").json()
    hidden = client.get("/api/tavern/state?strangers=false").json()

    assert len(shown["patrons"]) == 3
    # No named guests to fall back on, so the room is empty and quiet.
    assert hidden["patrons"] == []
    assert hidden["quiet"] is True
    # The ledger is the same either way: a display toggle is not an economy.
    assert hidden["ledger"]["earned"] == shown["ledger"]["earned"]
    assert shown["ledger"]["earned"] == 3 * STRANGER_PAYMENT
    assert shown["ledger"]["named_patrons"] == 0


def test_every_detection_in_a_visit_is_its_own_patron(client, db):
    # A visit with three tracks is three birds. Treating the visit as one
    # guest would under-count the room and the takings alike.
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=2, species=dove, n=3)

    body = client.get("/api/tavern/state").json()
    assert len(body["patrons"]) == 3
    assert body["ledger"]["patrons_served"] == 3
    assert len({p["detection_id"] for p in body["patrons"]}) == 3


def test_state_reports_today_in_the_cameras_own_day(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=1, species=dove, n=2)

    body = client.get("/api/tavern/state").json()
    assert body["tz"] == "America/New_York"
    assert body["hearth"]["phase"] in {"dawn", "day", "dusk", "night"}
    # Arrivals are offset-aware and must not be re-stamped as UTC.
    assert body["patrons"][0]["arrived_at"].endswith(("-05:00", "-04:00"))


# ── /arrivals ───────────────────────────────────────────────────────────


def test_arrivals_returns_only_what_is_new_oldest_first(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _, first = _arrive(db, minutes_ago=9, species=dove)
    _, rest = _arrive(db, minutes_ago=2, species=dove, n=2)

    body = client.get(f"/api/tavern/arrivals?after={first[0].id}").json()
    ids = [p["detection_id"] for p in body["patrons"]]

    assert ids == sorted(ids)
    assert first[0].id not in ids
    assert ids == [d.id for d in rest]
    assert body["earned"] == sum(p["payment"] for p in body["patrons"])
    assert body["cursor"] == rest[-1].id


def test_arrivals_separates_mishaps_from_guests(client, db):
    nab = _species(db, NOT_A_BIRD_LABEL, "n/a")
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=4, species=nab)
    _arrive(db, minutes_ago=3, species=dove)

    body = client.get("/api/tavern/arrivals?after=0").json()
    assert len(body["patrons"]) == 1
    assert len(body["events"]) == 1


# ── /unlock ─────────────────────────────────────────────────────────────


def test_unlock_refuses_an_upgrade_the_house_cannot_afford(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=5, species=dove)

    r = client.post("/api/tavern/unlock", json={"upgrade_id": "painted_sign"})
    assert r.status_code == 402
    assert "shillings" in r.json()["detail"]


def test_unlock_spends_the_balance_and_stays_spent(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=5, species=dove, n=40, seconds=60)

    before = client.get("/api/tavern/state").json()["ledger"]
    cost = UPGRADES_BY_ID["tallow_candles"]["cost"]
    assert before["balance"] >= cost

    r = client.post("/api/tavern/unlock", json={"upgrade_id": "tallow_candles"})
    assert r.status_code == 200
    after = r.json()["ledger"]

    assert after["spent"] == cost
    assert after["balance"] == before["balance"] - cost
    assert after["earned"] == before["earned"]

    # And it survives the next read, which is the part a JSON column
    # mutated in place would quietly fail.
    reread = client.get("/api/tavern/state").json()
    assert "tallow_candles" in reread["unlocked"]
    assert reread["ledger"]["spent"] == cost


def test_unlock_rejects_a_second_purchase_and_an_unknown_id(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=5, species=dove, n=40, seconds=60)

    assert client.post("/api/tavern/unlock", json={"upgrade_id": "tallow_candles"}).status_code == 200
    again = client.post("/api/tavern/unlock", json={"upgrade_id": "tallow_candles"})
    assert again.status_code == 409

    missing = client.post("/api/tavern/unlock", json={"upgrade_id": "gold_plated_nothing"})
    assert missing.status_code == 404


# ── /guestbook and /seen ────────────────────────────────────────────────


def test_guestbook_counts_visits_and_withholds_lore_until_the_ledger_is_bought(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    jay = _species(db, "Blue Jay", "Cyanocitta cristata")
    _arrive(db, minutes_ago=200, species=dove, n=80, seconds=90)
    _arrive(db, minutes_ago=5, species=jay)

    body = client.get("/api/tavern/guestbook").json()
    assert body["has_ledger"] is False
    names = [e["species"] for e in body["entries"]]
    assert names[0] == "Mourning Dove"  # busiest first
    assert all(e["lore"] is None for e in body["entries"])
    assert body["entries"][0]["visits"] == 80
    assert body["entries"][0]["first_seen"] is not None

    bought = client.post("/api/tavern/unlock", json={"upgrade_id": "guest_ledger"})
    assert bought.status_code == 200, bought.json()
    after = client.get("/api/tavern/guestbook").json()
    assert after["has_ledger"] is True
    assert after["entries"][0]["lore"]["backstory"]


def test_guestbook_leaves_out_the_things_that_were_not_birds(client, db):
    nab = _species(db, NOT_A_BIRD_LABEL, "n/a")
    _arrive(db, minutes_ago=5, species=nab, n=3)

    body = client.get("/api/tavern/guestbook").json()
    assert body["entries"] == []


def test_seen_marker_only_moves_forward(client, db):
    assert client.post("/api/tavern/seen", json={"detection_id": 50}).json()[
        "last_seen_detection_id"
    ] == 50
    # A second tab replaying older arrivals must not drag the marker back.
    assert client.post("/api/tavern/seen", json={"detection_id": 10}).json()[
        "last_seen_detection_id"
    ] == 50


# ── The founding purse ──────────────────────────────────────────────────


def test_history_pays_into_a_capped_purse(client, db):
    # A yard with a long archive should open on a good start, not on every
    # upgrade in the catalogue already affordable.
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=600, species=dove, n=400, seconds=90)

    ledger = client.get("/api/tavern/state").json()["ledger"]
    assert ledger["earned_all_time"] > FOUNDING_PURSE_CAP
    assert ledger["founding_purse"] == FOUNDING_PURSE_CAP
    assert ledger["since_opening"] == 0
    assert ledger["balance"] == FOUNDING_PURSE_CAP


def test_arrivals_after_opening_pay_in_full(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _arrive(db, minutes_ago=600, species=dove, n=2, seconds=30)

    opening = client.get("/api/tavern/state").json()["ledger"]
    assert opening["since_opening"] == 0

    # A negative "minutes ago" is a visit stamped after the house changed
    # hands, which is what every real new arrival is.
    _arrive(db, minutes_ago=-1, species=dove, n=2, seconds=30)

    after = client.get("/api/tavern/state").json()["ledger"]
    assert after["since_opening"] > 0
    assert after["founding_purse"] == opening["founding_purse"]
    assert after["balance"] == opening["balance"] + after["since_opening"]


def test_state_counts_what_arrived_while_the_page_was_shut(client, db):
    dove = _species(db, "Mourning Dove", "Zenaida macroura")
    _, first = _arrive(db, minutes_ago=30, species=dove)

    # Nothing has been watched yet, so there is no backlog to report.
    assert client.get("/api/tavern/state").json()["since_last_seen"] == 0

    client.post("/api/tavern/seen", json={"detection_id": first[0].id})
    _arrive(db, minutes_ago=5, species=dove, n=3)

    body = client.get("/api/tavern/state").json()
    assert body["since_last_seen"] == 3
    assert body["last_seen_detection_id"] == first[0].id
