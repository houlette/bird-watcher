"""Unit tests for Bayesian fusion of visual + audio + seasonal priors.

These tests use an in-memory SQLite DB so we can seed HaikuboxDetection rows
without touching the real session.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import HaikuboxDetection
from db.session import Base
from pipeline.fuse import AUDIO_BOOST, fuse


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


def test_renormalizes_to_one(db, monkeypatch):
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "")
    fused = fuse([("Northern Cardinal", 0.6), ("Tufted Titmouse", 0.3), ("Blue Jay", 0.1)], db=db)
    assert pytest.approx(sum(f.probability for f in fused), abs=1e-9) == 1.0


def test_no_audio_no_season_preserves_order(db, monkeypatch):
    """With no audio and no listed seasonal priors, fusion is identity (just renorm)."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "")
    fused = fuse([("Northern Cardinal", 0.6), ("Tufted Titmouse", 0.3), ("Blue Jay", 0.1)], db=db)
    species_order = [f.species for f in fused]
    assert species_order == ["Northern Cardinal", "Tufted Titmouse", "Blue Jay"]
    # No species is audio_confirmed when there are no audio detections.
    assert all(not f.audio_confirmed for f in fused)


def test_audio_match_flips_close_call(db, monkeypatch):
    """When the visual is close (55/40) and audio agrees with the runner-up, it should win."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Purple Finch", detected_at=now - timedelta(seconds=30)))
    db.commit()
    fused = fuse([("House Finch", 0.55), ("Purple Finch", 0.40), ("Pine Siskin", 0.05)], db=db, when=now)
    assert fused[0].species == "Purple Finch"
    assert fused[0].audio_confirmed is True
    # House Finch is still in the running, just not first.
    assert any(f.species == "House Finch" and not f.audio_confirmed for f in fused)


def test_strong_visual_resists_audio_for_different_species(db, monkeypatch):
    """A clear visual (95%) shouldn't get overridden by audio of a different bird."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Tufted Titmouse", detected_at=now - timedelta(seconds=30)))
    db.commit()
    fused = fuse([("Northern Cardinal", 0.95), ("Tufted Titmouse", 0.03), ("Blue Jay", 0.02)], db=db, when=now)
    assert fused[0].species == "Northern Cardinal"


def test_audio_outside_window_is_ignored(db, monkeypatch):
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_window_seconds", 90)
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_lookahead_seconds", 30)
    now = datetime(2026, 5, 1, 12, 0, 0)
    # 10 minutes ago — outside the 90s window
    db.add(HaikuboxDetection(species_common_name="Purple Finch", detected_at=now - timedelta(minutes=10)))
    db.commit()
    fused = fuse([("House Finch", 0.55), ("Purple Finch", 0.40)], db=db, when=now)
    assert fused[0].species == "House Finch"
    assert all(not f.audio_confirmed for f in fused)


def test_audio_during_visit_is_matched(db, monkeypatch):
    """Audio 15 seconds after visit start (during the ~20s clip) should match."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_window_seconds", 90)
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_lookahead_seconds", 30)
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Purple Finch", detected_at=now + timedelta(seconds=15)))
    db.commit()
    fused = fuse([("House Finch", 0.55), ("Purple Finch", 0.40)], db=db, when=now)
    assert fused[0].species == "Purple Finch"
    assert fused[0].audio_confirmed is True


def test_audio_just_after_visit_is_matched(db, monkeypatch):
    """Audio 25 seconds after visit start (departure call within 30s lookahead) should match."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_window_seconds", 90)
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_lookahead_seconds", 30)
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Purple Finch", detected_at=now + timedelta(seconds=25)))
    db.commit()
    fused = fuse([("House Finch", 0.55), ("Purple Finch", 0.40)], db=db, when=now)
    assert fused[0].species == "Purple Finch"
    assert fused[0].audio_confirmed is True


def test_audio_beyond_lookahead_is_ignored(db, monkeypatch):
    """Audio well after the visit (>30s lookahead) should not match."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_window_seconds", 90)
    monkeypatch.setattr("pipeline.fuse.settings.audio_correlation_lookahead_seconds", 30)
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Purple Finch", detected_at=now + timedelta(seconds=60)))
    db.commit()
    fused = fuse([("House Finch", 0.55), ("Purple Finch", 0.40)], db=db, when=now)
    assert fused[0].species == "House Finch"
    assert all(not f.audio_confirmed for f in fused)


def test_seasonal_prior_boosts_winter_junco(db, monkeypatch):
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "")
    january = datetime(2026, 1, 15, 12, 0, 0)
    # Visually the classifier slightly prefers Song Sparrow (no seasonal entry,
    # so prior is 1.0); the junco has a 2.0 January multiplier and should win.
    fused = fuse([("Song Sparrow", 0.50), ("Dark-eyed Junco", 0.35)], db=db, when=january)
    assert fused[0].species == "Dark-eyed Junco"


def test_seasonal_prior_suppresses_summer_junco(db, monkeypatch):
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "")
    july = datetime(2026, 7, 15, 12, 0, 0)
    # In July the junco multiplier is 0.1 — Song Sparrow should win comfortably.
    fused = fuse([("Song Sparrow", 0.50), ("Dark-eyed Junco", 0.50)], db=db, when=july)
    assert fused[0].species == "Song Sparrow"


def test_audio_boost_magnitude():
    """Sanity check: AUDIO_BOOST is configured at a level that can flip close calls."""
    # If a runner-up at 40% gets boosted by 3.0 → 120 (unnormalized), and the
    # winner at 55% stays at 55, the runner-up wins. The exact tuning is
    # subjective but we encode the design intent.
    assert AUDIO_BOOST >= 2.0
    assert AUDIO_BOOST <= 5.0


def test_is_regional_species():
    from pipeline.fuse import is_regional_species

    # Common Eastern NA backyard / Massachusetts species
    assert is_regional_species("Northern Cardinal") is True
    assert is_regional_species("Mourning Dove") is True
    assert is_regional_species("Black-capped Chickadee") is True
    assert is_regional_species("Blue Jay") is True
    assert is_regional_species("Song Sparrow") is True

    # Generic family labels
    assert is_regional_species("Sparrow") is True
    assert is_regional_species("Woodpecker") is True
    assert is_regional_species("Hawk") is True

    # Desert / Southwestern / Western birds not in regional yard baseline
    assert is_regional_species("Inca Dove") is False
    assert is_regional_species("White-winged Dove") is False
    assert is_regional_species("Harris's Sparrow") is False
    assert is_regional_species("Pyrrhuloxia") is False
    assert is_regional_species("Phainopepla") is False
    assert is_regional_species("Pelagic Cormorant") is False

    # Edge cases
    assert is_regional_species("") is False
    assert is_regional_species(None) is False


def test_out_of_range_downweight_allows_local_species_to_win(db, monkeypatch):
    """When an out-of-range species (Inca Dove) leads visually over a local species (Mourning Dove),
    the 20x vagrant penalty (0.05) lets the true local species win."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "")

    # Inca Dove @ 45% vs Mourning Dove @ 35%
    fused = fuse([("Inca Dove", 0.45), ("Mourning Dove", 0.35), ("Rock Pigeon", 0.20)], db=db)
    assert fused[0].species == "Mourning Dove"
    assert fused[0].probability > 0.60
    # Inca Dove is heavily penalized
    inca = next(f for f in fused if f.species == "Inca Dove")
    assert inca.probability < 0.10


def test_out_of_range_audio_confirmed_wins(db, monkeypatch):
    """If Haikubox confirms the rare visitor, it receives the audio boost and no vagrant penalty."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Inca Dove", detected_at=now - timedelta(seconds=10)))
    db.commit()

    fused = fuse([("Inca Dove", 0.45), ("Mourning Dove", 0.35)], db=db, when=now)
    assert fused[0].species == "Inca Dove"
    assert fused[0].audio_confirmed is True

