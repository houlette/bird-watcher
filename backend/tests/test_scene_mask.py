"""Unit tests for the scene-mask spatial filter."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import (
    NOT_A_BIRD_LABEL,
    Base,
    Correction,
    Detection,
    Species,
    Visit,
)
from pipeline import scene_mask


@dataclass
class _FakeDet:
    """Mirrors BirdDetection's interface enough for filter_detections()."""
    bbox: tuple[int, int, int, int]
    confidence: float


@pytest.fixture(autouse=True)
def reset_cache():
    """Each test starts with a cold cache so prior runs don't leak through."""
    scene_mask._cached_zones = None
    scene_mask._cached_at = None
    yield
    scene_mask._cached_zones = None
    scene_mask._cached_at = None


@pytest.fixture()
def session_maker(monkeypatch):
    """In-memory DB with the schema applied; monkeypatches SessionLocal so
    scene_mask._compute_hot_zones reads from it."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    monkeypatch.setattr(scene_mask, "SessionLocal", SessionLocal)
    return SessionLocal


def _seed_nab(db, bbox):
    """Insert one Detection + Correction(NAB) with the given bbox."""
    species = db.query(Species).filter_by(common_name=NOT_A_BIRD_LABEL).one_or_none()
    if species is None:
        species = Species(common_name=NOT_A_BIRD_LABEL, scientific_name="", is_rare=False)
        db.add(species); db.flush()
    visit = Visit(clip_path="clips/x.mp4")
    db.add(visit); db.flush()
    det = Detection(
        visit_id=visit.id, species_id=species.id, confidence=0.0,
        raw_predictions=[], audio_confirmed=False,
        crop_path="crops/x.jpg", bbox=list(bbox), track_id=1,
    )
    db.add(det); db.flush()
    corr = Correction(detection_id=det.id, correct_species_id=species.id)
    db.add(corr); db.commit()
    return det


def test_is_masked_returns_false_when_no_hot_zones():
    """Empty hot-zone set means nothing is ever masked."""
    assert scene_mask.is_masked((100, 100, 50, 50), set()) is False


def test_is_masked_detects_center_inside_cell():
    """A bbox centered at (1750, 550) lands in cell (17, 5) with GRID_PX=100."""
    hot = {(17, 5)}
    assert scene_mask.is_masked((1710, 510, 80, 80), hot) is True  # center ~(1750, 550)


def test_is_masked_outside_hot_zone():
    hot = {(17, 5)}
    # Center (1850, 550) is in cell (18, 5) — adjacent but not hot.
    assert scene_mask.is_masked((1810, 510, 80, 80), hot) is False


def test_filter_drops_low_conf_detection_in_hot_zone():
    hot = {(17, 5)}
    d = _FakeDet(bbox=(1710, 510, 80, 80), confidence=0.30)
    kept, suppressed = scene_mask.filter_detections([d], hot)
    assert kept == []
    assert suppressed == 1


def test_filter_preserves_high_conf_detection_in_hot_zone():
    """User wants a hummingbird AT the feeder to still be reported."""
    hot = {(17, 5)}
    d = _FakeDet(bbox=(1710, 510, 80, 80), confidence=0.90)
    kept, suppressed = scene_mask.filter_detections([d], hot)
    assert kept == [d]
    assert suppressed == 0


def test_filter_preserves_detections_outside_hot_zone():
    hot = {(17, 5)}
    d = _FakeDet(bbox=(100, 100, 80, 80), confidence=0.30)
    kept, suppressed = scene_mask.filter_detections([d], hot)
    assert kept == [d]
    assert suppressed == 0


def test_filter_with_no_hot_zones_is_passthrough():
    d1 = _FakeDet(bbox=(0, 0, 10, 10), confidence=0.30)
    d2 = _FakeDet(bbox=(1000, 1000, 10, 10), confidence=0.90)
    kept, suppressed = scene_mask.filter_detections([d1, d2], set())
    assert kept == [d1, d2]
    assert suppressed == 0


def test_compute_hot_zones_clusters_dense_nabs(session_maker):
    """≥MIN_NABS_PER_CELL labels in one cell yields a hot zone."""
    db = session_maker()
    try:
        # Seed enough NABs in cell (17, 5) to clear the threshold.
        for _ in range(scene_mask.MIN_NABS_PER_CELL):
            _seed_nab(db, (1710, 510, 80, 80))
        # One isolated NAB elsewhere — below threshold, shouldn't make a cell hot.
        _seed_nab(db, (100, 100, 50, 50))
    finally:
        db.close()

    zones = scene_mask._compute_hot_zones()
    assert (17, 5) in zones
    assert (1, 1) not in zones


def test_compute_hot_zones_returns_empty_when_no_nabs(session_maker):
    """No NABs in the DB → no hot zones (clean install case)."""
    zones = scene_mask._compute_hot_zones()
    assert zones == set()


def test_get_hot_zones_caches_result(session_maker, monkeypatch):
    """Second call within the TTL should not re-query the DB."""
    calls = {"n": 0}
    real = scene_mask._compute_hot_zones

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(scene_mask, "_compute_hot_zones", counting)
    scene_mask.get_hot_zones()
    scene_mask.get_hot_zones()
    assert calls["n"] == 1


def test_get_hot_zones_force_refresh_bypasses_cache(session_maker, monkeypatch):
    calls = {"n": 0}
    real = scene_mask._compute_hot_zones

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(scene_mask, "_compute_hot_zones", counting)
    scene_mask.get_hot_zones()
    scene_mask.get_hot_zones(force_refresh=True)
    assert calls["n"] == 2
