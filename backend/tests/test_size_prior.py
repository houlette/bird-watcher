"""Tests for the per-species size prior and its integration in fuse()."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.session import Base
from pipeline import size_prior
from pipeline.fuse import fuse
from pipeline.size_prior import SIZE_FLOOR


def _write_priors(path, species: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proxy": "max_wh",
        "min_labels": 20,
        "min_log_std": 0.10,
        "species": species,
    }))


def test_size_multiplier_no_calibration_returns_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", tmp_path / "missing.json")
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Northern Cardinal", 150.0) == 1.0


def test_size_multiplier_unknown_species_returns_identity(monkeypatch, tmp_path):
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    # Species not in the priors → no prior, no penalty.
    assert size_prior.size_multiplier("Sandhill Crane", 160.0) == 1.0


def test_size_multiplier_peaks_at_geometric_mean(monkeypatch, tmp_path):
    """A bird the exact size of the species's geometric mean should get
    mult ≈ 1.0 (the prior says 'totally consistent')."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Mourning Dove", 160.0) == pytest.approx(1.0, abs=1e-6)


def test_size_multiplier_penalizes_off_size_observation(monkeypatch, tmp_path):
    """A bbox 2σ above the species mean should be penalized but not zeroed."""
    path = tmp_path / "size_priors.json"
    sigma = 0.15
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": sigma},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    two_sigma_high = math.exp(math.log(160) + 2 * sigma)  # ≈ 216 px
    mult = size_prior.size_multiplier("Mourning Dove", two_sigma_high)
    # exp(-2) ≈ 0.135, but our floor is SIZE_FLOOR=0.33 → clipped up.
    assert mult == pytest.approx(SIZE_FLOOR, abs=1e-6)


def test_size_multiplier_floors_for_extreme_observation(monkeypatch, tmp_path):
    """A 10× too-big observation shouldn't infinitely penalize — floor caps."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Mourning Dove", 1600.0) == SIZE_FLOOR


def test_size_multiplier_degenerate_input_returns_identity(monkeypatch, tmp_path):
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Mourning Dove", 0.0) == 1.0
    assert size_prior.size_multiplier("Mourning Dove", -10.0) == 1.0
    assert size_prior.size_multiplier("Mourning Dove", None) == 1.0


# ─── Integration: fuse() with bbox + priors ─────────────────────────────


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


def test_fuse_without_bbox_skips_size_prior(monkeypatch, tmp_path, db):
    """Calling fuse() with no bbox preserves the old behavior — size_mult=1.0
    for every candidate even when a calibration file exists. This is the
    fallback path for any caller that hasn't been updated to pass bbox."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()

    out = fuse(
        [("Mourning Dove", 0.6), ("Rock Pigeon", 0.4)],
        db=db,
        when=datetime(2026, 5, 30, 12, 0, 0),
    )
    for r in out:
        assert r.size_mult == 1.0


def test_fuse_with_bbox_applies_size_prior_to_known_species(monkeypatch, tmp_path, db):
    """An observed bbox matching Mourning Dove's mean but very-off for
    Rock Pigeon should re-rank the posterior so Dove dominates."""
    path = tmp_path / "size_priors.json"
    # Dove fit at 160 px, Pigeon at 240 px — observe 160 → Dove perfect fit,
    # Pigeon many σ away.
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.10},
        "Rock Pigeon": {"n": 100, "log_mean": math.log(240), "log_std": 0.10},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()

    # bbox 160 px square at origin → max(w,h)=160.
    out = fuse(
        [("Mourning Dove", 0.45), ("Rock Pigeon", 0.55)],
        db=db,
        when=datetime(2026, 5, 30, 12, 0, 0),
        bbox=(0, 0, 160, 160),
    )
    by_sp = {r.species: r for r in out}
    # Dove's size_mult ≈ 1.0; Pigeon's is floored at SIZE_FLOOR.
    assert by_sp["Mourning Dove"].size_mult == pytest.approx(1.0, abs=1e-6)
    assert by_sp["Rock Pigeon"].size_mult == SIZE_FLOOR
    # The classifier favored Pigeon (0.55 vs 0.45) but size flips the rank.
    assert out[0].species == "Mourning Dove"


def test_fuse_size_prior_no_op_for_uncalibrated_species(monkeypatch, tmp_path, db):
    """A candidate that isn't in the priors gets mult=1.0 — same as before
    the prior was added. Important for new/rare species we haven't labeled
    enough times to fit yet (we don't want to penalize them out of
    existence)."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.10},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()

    out = fuse(
        [("Baltimore Oriole", 0.55), ("Mourning Dove", 0.45)],
        db=db,
        when=datetime(2026, 5, 30, 12, 0, 0),
        bbox=(0, 0, 200, 60),  # max_wh = 200 — way off Dove's 160
    )
    by_sp = {r.species: r for r in out}
    assert by_sp["Baltimore Oriole"].size_mult == 1.0
    # Dove penalized, oriole unaffected → oriole keeps its visual lead.
    assert out[0].species == "Baltimore Oriole"
