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


def _write_priors(
    path,
    species: dict,
    *,
    aspect_bounds: tuple[float, float] = (0.40, 2.50),
    perch_scales: list[dict] | None = None,
    image_height: int = 2160,
) -> None:
    """Write a calibration JSON shaped like calibrate_size_priors.py emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    perch_payload = {
        "image_height_px": image_height,
        "n_bins": len(perch_scales) if perch_scales else 5,
        "anchor_species": "Mourning Dove",
        "min_per_bin": 50,
        "scales": perch_scales or [],
    }
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proxy": "max_wh",
        "min_labels": 20,
        "min_log_std": 0.10,
        "aspect_ratio_bounds": list(aspect_bounds),
        "perch": perch_payload,
        "species": species,
    }))


def _bbox(w: float, h: float, x: float = 0, y: float = 0) -> tuple:
    return (x, y, w, h)


# ─── Loader / identity behaviors ────────────────────────────────────────


def test_size_multiplier_no_calibration_returns_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", tmp_path / "missing.json")
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Northern Cardinal", _bbox(150, 150)) == 1.0


def test_size_multiplier_unknown_species_returns_identity(monkeypatch, tmp_path):
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Sandhill Crane", _bbox(160, 160)) == 1.0


def test_size_multiplier_peaks_at_geometric_mean(monkeypatch, tmp_path):
    """A bird the exact size of the species's geometric mean should get
    mult ≈ 1.0 (the prior says 'totally consistent')."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Mourning Dove", _bbox(160, 160)) == pytest.approx(1.0, abs=1e-6)


def test_size_multiplier_penalizes_off_size_observation(monkeypatch, tmp_path):
    path = tmp_path / "size_priors.json"
    sigma = 0.15
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": sigma},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    two_sigma_high = math.exp(math.log(160) + 2 * sigma)  # ≈ 216 px
    mult = size_prior.size_multiplier("Mourning Dove", _bbox(two_sigma_high, two_sigma_high))
    # exp(-2) ≈ 0.135, but our floor is SIZE_FLOOR=0.33 → clipped up.
    assert mult == pytest.approx(SIZE_FLOOR, abs=1e-6)


def test_size_multiplier_floors_for_extreme_observation(monkeypatch, tmp_path):
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    # 1600 / 160 = 10x — way past 5σ, floored.
    assert size_prior.size_multiplier("Mourning Dove", _bbox(1600, 1000)) == SIZE_FLOOR


def test_size_multiplier_degenerate_input_returns_identity(monkeypatch, tmp_path):
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    assert size_prior.size_multiplier("Mourning Dove", _bbox(0, 0)) == 1.0
    assert size_prior.size_multiplier("Mourning Dove", _bbox(-10, 10)) == 1.0
    assert size_prior.size_multiplier("Mourning Dove", None) == 1.0
    assert size_prior.size_multiplier("Mourning Dove", [1, 2]) == 1.0


# ─── Aspect-ratio gating ────────────────────────────────────────────────


def test_size_multiplier_skips_aspect_ratio_outliers(monkeypatch, tmp_path):
    """A wings-extended frame (very wide) gets no prior — the bbox isn't
    a meaningful size signal."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    }, aspect_bounds=(0.40, 2.50))
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    # max_wh = 320 here would normally floor the multiplier; but aspect=8/1
    # is way out of bounds, so we expect a clean identity instead of penalty.
    assert size_prior.size_multiplier("Mourning Dove", _bbox(320, 40)) == 1.0
    # A narrow-vertical bird (aspect=0.25) is also out of bounds.
    assert size_prior.size_multiplier("Mourning Dove", _bbox(40, 320)) == 1.0


def test_size_multiplier_accepts_borderline_aspect(monkeypatch, tmp_path):
    """Bbox exactly at the aspect-ratio boundary should not be gated out
    (it's the >/< not ≥/≤ check we want)."""
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.15},
    }, aspect_bounds=(0.40, 2.50))
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    # aspect = 2.5 exactly; max_wh = 200 → log diff = log(200) - log(160) ≈ 0.22
    # z ≈ 1.49 → exp(-z) ≈ 0.22, floored at SIZE_FLOOR.
    mult = size_prior.size_multiplier("Mourning Dove", _bbox(200, 80))
    assert mult > 0.0  # not gated out → goes through scoring path
    assert mult == SIZE_FLOOR  # well off-mean → floored


# ─── Perch scaling ──────────────────────────────────────────────────────


def test_size_multiplier_applies_perch_scale(monkeypatch, tmp_path):
    """A bird in a "small-bbox bin" gets its observation scaled UP to match
    the global anchor scale before being compared to the species fit."""
    path = tmp_path / "size_priors.json"
    perch_scales = [
        # Top half: shrink-zone (birds appear ~80% as big here).
        {"y_min": 0,    "y_max": 1080, "n": 80, "scale": 1.25},  # 1 / 0.80
        # Bottom half: identity (the anchor region).
        {"y_min": 1080, "y_max": 2160, "n": 200, "scale": 1.00},
    ]
    _write_priors(
        path,
        {"Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.10}},
        perch_scales=perch_scales,
    )
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()

    # Same observed bbox (max_wh=128), foot at y=500 (top bin) → scaled to 160 → matches mean.
    top_bbox = _bbox(128, 128, y=500 - 128)  # foot = 500
    assert size_prior.size_multiplier("Mourning Dove", top_bbox) == pytest.approx(1.0, abs=1e-6)

    # Same observed bbox (max_wh=128), foot at y=1500 (bottom bin) → no scale → penalized.
    bot_bbox = _bbox(128, 128, y=1500 - 128)  # foot = 1500
    assert size_prior.size_multiplier("Mourning Dove", bot_bbox) < 0.5


def test_size_multiplier_no_perch_scales_defaults_to_identity(monkeypatch, tmp_path):
    """When the calibration has no perch_scales (older file or thin data),
    behavior reverts to the simple max(w,h) prior."""
    path = tmp_path / "size_priors.json"
    _write_priors(
        path,
        {"Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.10}},
        perch_scales=[],
    )
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()
    # max_wh=160 anywhere in frame → mult ≈ 1.0 (no scaling).
    assert size_prior.size_multiplier("Mourning Dove", _bbox(160, 160, y=0)) == pytest.approx(1.0, abs=1e-6)
    assert size_prior.size_multiplier("Mourning Dove", _bbox(160, 160, y=1500)) == pytest.approx(1.0, abs=1e-6)


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
    path = tmp_path / "size_priors.json"
    _write_priors(path, {
        "Mourning Dove": {"n": 100, "log_mean": math.log(160), "log_std": 0.10},
        "Rock Pigeon": {"n": 100, "log_mean": math.log(240), "log_std": 0.10},
    })
    monkeypatch.setattr(size_prior, "CALIBRATION_PATH", path)
    size_prior.reset_cache_for_tests()

    out = fuse(
        [("Mourning Dove", 0.45), ("Rock Pigeon", 0.55)],
        db=db,
        when=datetime(2026, 5, 30, 12, 0, 0),
        bbox=(0, 0, 160, 160),
    )
    by_sp = {r.species: r for r in out}
    assert by_sp["Mourning Dove"].size_mult == pytest.approx(1.0, abs=1e-6)
    assert by_sp["Rock Pigeon"].size_mult == SIZE_FLOOR
    # The classifier favored Pigeon (0.55 vs 0.45) but size flips the rank.
    assert out[0].species == "Mourning Dove"


def test_fuse_size_prior_no_op_for_uncalibrated_species(monkeypatch, tmp_path, db):
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
        bbox=(0, 0, 200, 60),  # aspect 3.33 → out of bounds → mult 1.0 for both
    )
    by_sp = {r.species: r for r in out}
    assert by_sp["Baltimore Oriole"].size_mult == 1.0
    assert by_sp["Mourning Dove"].size_mult == 1.0  # gated out by aspect ratio
    # Oriole keeps its visual lead.
    assert out[0].species == "Baltimore Oriole"
