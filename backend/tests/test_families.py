"""Tests for family-level catch-all labels.

Covers:
  - The reverse-index helper API (members / contains / is_family_label).
  - DB seeding via ensure_family_species_rows() (idempotent + flag flip).
  - Species API exposes the families group.
  - Submitting a correction to a family name persists correctly.
  - Stats: classifier-accuracy partial credit when the user said "Sparrow"
    and the model's top-1 was a member species.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.families import (
    FAMILY_MEMBERS,
    FAMILY_NAMES,
    ensure_family_species_rows,
    family_contains,
    family_members,
    is_family_label,
    species_to_families,
)
from db.models import Correction, Detection, Species, Visit
from db.session import Base, get_db
from main import app
from pipeline.stats import compute_daily_stats


# ─── Pure-function helpers ──────────────────────────────────────────────


def test_family_helpers_round_trip():
    assert is_family_label("Sparrow")
    assert not is_family_label("Northern Cardinal")
    assert "House Sparrow" in family_members("Sparrow")
    assert family_contains("Sparrow", "House Sparrow")
    assert not family_contains("Sparrow", "Rock Pigeon")
    # Junco is a New World sparrow — verify the curation.
    assert family_contains("Sparrow", "Dark-eyed Junco")
    # Reverse lookup.
    assert "Sparrow" in species_to_families("White-throated Sparrow")
    assert species_to_families("Rock Pigeon") == []


def test_family_helpers_known_families_present():
    """The four user-chosen starter families should all be defined."""
    for name in ("Sparrow", "Warbler", "Woodpecker", "Finch"):
        assert name in FAMILY_NAMES
        assert len(FAMILY_MEMBERS[name]) >= 3


# ─── DB seeding ────────────────────────────────────────────────────────


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


def test_ensure_family_species_rows_seeds_each_family(db):
    created = ensure_family_species_rows(db)
    assert created == len(FAMILY_MEMBERS)
    for name in FAMILY_MEMBERS:
        sp = db.query(Species).filter(Species.common_name == name).one()
        assert sp.is_family is True or sp.is_family == 1


def test_ensure_family_species_rows_is_idempotent(db):
    ensure_family_species_rows(db)
    second = ensure_family_species_rows(db)
    assert second == 0
    # Still exactly one row per family.
    for name in FAMILY_MEMBERS:
        assert db.query(Species).filter(Species.common_name == name).count() == 1


def test_ensure_family_species_rows_promotes_existing_unflagged(db):
    """If a Species row already exists with the family's name (e.g., user
    typed 'Sparrow' before this feature existed), promote it in place
    rather than creating a duplicate."""
    db.add(Species(common_name="Sparrow", scientific_name="", is_rare=False, is_family=False))
    db.commit()
    ensure_family_species_rows(db)
    rows = db.query(Species).filter(Species.common_name == "Sparrow").all()
    assert len(rows) == 1
    assert rows[0].is_family is True or rows[0].is_family == 1


# ─── /api/species ─────────────────────────────────────────────────────


def test_species_endpoint_exposes_families(client):
    body = client.get("/api/species").json()
    assert "families" in body
    names = [f["name"] for f in body["families"]]
    for fam in ("Sparrow", "Warbler", "Woodpecker", "Finch"):
        assert fam in names
    # Each family carries its member-species list.
    sparrow = next(f for f in body["families"] if f["name"] == "Sparrow")
    assert "House Sparrow" in sparrow["members"]


def test_species_endpoint_excludes_families_from_yard_and_extra(client):
    body = client.get("/api/species").json()
    for s in body["yard"] + body["extra"]:
        assert s["name"] not in FAMILY_NAMES


# ─── Corrections to a family ──────────────────────────────────────────


def _seed_detection(db, species_name: str = "Northern Cardinal") -> int:
    sp = Species(common_name=species_name, scientific_name="", is_rare=False)
    db.add(sp)
    db.flush()
    v = Visit(started_at=datetime.utcnow(), clip_path="clips/x.mp4")
    db.add(v)
    db.flush()
    d = Detection(
        visit_id=v.id, species_id=sp.id, confidence=0.5,
        raw_predictions=[{"species": species_name, "p": 0.5}],
        audio_confirmed=False,
        crop_path="crops/x.jpg", bbox=[100, 100, 80, 80], track_id=1,
    )
    db.add(d)
    db.commit()
    return d.id


def test_correction_to_family_persists(client, db):
    """A correction to 'Sparrow' should resolve to the family Species row,
    creating it on the fly via the families seeder."""
    ensure_family_species_rows(db)
    det_id = _seed_detection(db, "House Sparrow")
    r = client.post(
        "/api/corrections",
        json={"detection_id": det_id, "correct_species_name": "Sparrow"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["species"] == "Sparrow"
    # Detection points at the Sparrow family Species row.
    det = db.get(Detection, det_id)
    db.refresh(det)
    assert det.species.common_name == "Sparrow"
    assert det.species.is_family is True or det.species.is_family == 1


# ─── Stats partial credit ─────────────────────────────────────────────


def test_classifier_accuracy_gives_partial_credit_for_family(db):
    """User says 'Sparrow', classifier said 'House Sparrow' → counted as
    correct in the daily classifier-accuracy metric."""
    ensure_family_species_rows(db)
    fam = db.query(Species).filter(Species.common_name == "Sparrow").one()
    # A non-member species so we can also test the wrong-family case.
    cardinal = Species(common_name="Northern Cardinal", scientific_name="", is_rare=False)
    db.add(cardinal)
    db.flush()

    day = datetime(2026, 5, 30, 12, 0, 0)
    v = Visit(started_at=day, clip_path="clips/a.mp4",
              processed_at=day, processing_error=None)
    db.add(v)
    db.flush()

    # Detection #1: classifier top-1 = House Sparrow; user said Sparrow. HIT.
    d1 = Detection(
        visit_id=v.id, species_id=fam.id, confidence=0.0,
        raw_predictions=[{"species": "House Sparrow", "p": 0.4}],
        audio_confirmed=False,
        crop_path="crops/a.jpg", bbox=[0, 0, 80, 80], track_id=1,
    )
    db.add(d1)
    db.flush()
    db.add(Correction(detection_id=d1.id, correct_species_id=fam.id))

    # Detection #2: classifier top-1 = Northern Cardinal; user said Sparrow. MISS.
    d2 = Detection(
        visit_id=v.id, species_id=fam.id, confidence=0.0,
        raw_predictions=[{"species": "Northern Cardinal", "p": 0.5}],
        audio_confirmed=False,
        crop_path="crops/b.jpg", bbox=[0, 0, 80, 80], track_id=2,
    )
    db.add(d2)
    db.flush()
    db.add(Correction(detection_id=d2.id, correct_species_id=fam.id))

    db.commit()

    stats = compute_daily_stats(db, day.date())
    assert stats.classifier_eligible == 2
    assert stats.classifier_correct == 1
