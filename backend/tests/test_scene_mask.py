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


def _seed_labeled(db, bbox, common_name):
    """Insert one Detection labeled `common_name` with NO Correction row.

    This is what the binary filter produces: it overwrites species_id in
    place and never writes a Correction, which is precisely why the old
    hot-zone query could not see any of its work.
    """
    species = db.query(Species).filter_by(common_name=common_name).one_or_none()
    if species is None:
        species = Species(common_name=common_name, scientific_name="", is_rare=False)
        db.add(species); db.flush()
    visit = Visit(clip_path="clips/x.mp4")
    db.add(visit); db.flush()
    det = Detection(
        visit_id=visit.id, species_id=species.id, confidence=0.5,
        raw_predictions=[], audio_confirmed=False,
        crop_path="crops/x.jpg", bbox=list(bbox), track_id=1,
    )
    db.add(det); db.commit()
    return det


def test_pipeline_nab_verdicts_alone_can_make_a_cell_hot(session_maker):
    """The regression this fixes: the user stops labeling, and the mask used
    to empty out even while the binary filter kept calling the same spot NAB
    hundreds of times."""
    db = session_maker()
    try:
        for _ in range(scene_mask.MIN_MACHINE_NABS_PER_CELL):
            _seed_labeled(db, (1710, 510, 80, 80), NOT_A_BIRD_LABEL)
    finally:
        db.close()

    assert (17, 5) in scene_mask._compute_hot_zones()


def test_pipeline_nab_below_machine_threshold_is_not_hot(session_maker):
    """A machine verdict is weaker evidence than a user label, so the count
    that qualifies a cell by hand is deliberately not enough on its own."""
    db = session_maker()
    try:
        for _ in range(scene_mask.MIN_NABS_PER_CELL):
            _seed_labeled(db, (1710, 510, 80, 80), NOT_A_BIRD_LABEL)
    finally:
        db.close()

    assert scene_mask._compute_hot_zones() == set()


def test_real_perch_is_not_masked_despite_many_nab_verdicts(session_maker):
    """A busy perch collects NAB verdicts from blurred birds. It must survive.

    Modeled on the real feeder cell, which ran 30 NAB against 23 sparrows in
    the fortnight to 2026-09-11 and must never go dark."""
    db = session_maker()
    try:
        for _ in range(30):
            _seed_labeled(db, (1410, 1810, 80, 80), NOT_A_BIRD_LABEL)
        for _ in range(23):
            _seed_labeled(db, (1410, 1810, 80, 80), "House Sparrow")
    finally:
        db.close()

    assert scene_mask._compute_hot_zones() == set()


def test_bird_label_cap_outranks_purity(session_maker):
    """A cell can pass the purity ratio on volume alone; the absolute cap on
    bird labels is what stops a heavily-trafficked spot qualifying."""
    db = session_maker()
    try:
        for _ in range(200):
            _seed_labeled(db, (1410, 1810, 80, 80), NOT_A_BIRD_LABEL)
        for _ in range(scene_mask.MAX_BIRD_LABELS_IN_HOT_CELL + 1):
            _seed_labeled(db, (1410, 1810, 80, 80), "House Sparrow")
    finally:
        db.close()

    # 200 vs 6 is 97% pure, comfortably past MACHINE_NAB_PURITY.
    assert scene_mask._compute_hot_zones() == set()


def test_user_labels_still_qualify_at_the_lower_threshold(session_maker):
    """The hand-label path is unchanged and does not have to clear the
    machine threshold or the purity test."""
    db = session_maker()
    try:
        for _ in range(scene_mask.MIN_NABS_PER_CELL):
            _seed_nab(db, (1710, 510, 80, 80))
        # Plenty of real birds in the same cell: a user label is authoritative
        # and is not diluted by them.
        for _ in range(50):
            _seed_labeled(db, (1710, 510, 80, 80), "House Sparrow")
    finally:
        db.close()

    assert (17, 5) in scene_mask._compute_hot_zones()


def test_backfill_corrections_do_not_hold_a_cell_hot(session_maker):
    """The backfill writes NAB corrections of its own. If those counted as
    user labels, a cell could keep itself hot for a fortnight after the
    artifact justifying it had gone."""
    db = session_maker()
    try:
        species = db.query(Species).filter_by(common_name=NOT_A_BIRD_LABEL).one_or_none()
        if species is None:
            species = Species(common_name=NOT_A_BIRD_LABEL, scientific_name="", is_rare=False)
            db.add(species); db.flush()
        for _ in range(scene_mask.MIN_NABS_PER_CELL * 2):
            visit = Visit(clip_path="clips/x.mp4")
            db.add(visit); db.flush()
            det = Detection(
                visit_id=visit.id, species_id=species.id, confidence=0.0,
                raw_predictions=[], audio_confirmed=False,
                crop_path="crops/x.jpg", bbox=[1710, 510, 80, 80], track_id=1,
            )
            db.add(det); db.flush()
            db.add(Correction(detection_id=det.id, correct_species_id=species.id,
                              source=scene_mask.BACKFILL_SOURCE))
        db.commit()
    finally:
        db.close()

    # The detections themselves are NAB-labeled, so the machine path sees 20
    # of them and that alone is enough. What must not happen is the human
    # path counting the backfill's corrections at its lower threshold.
    zones = scene_mask._compute_hot_zones()
    assert (17, 5) in zones  # machine path, on the NAB labels themselves

    # With only half as many, the machine threshold is not met and the
    # backfill corrections must not carry the cell on their own.
    db = session_maker()
    try:
        for d in db.query(Detection).limit(scene_mask.MIN_MACHINE_NABS_PER_CELL // 2).all():
            db.delete(db.query(Correction).filter_by(detection_id=d.id).one())
            db.delete(d)
        db.commit()
    finally:
        db.close()
    assert scene_mask._compute_hot_zones() == set()
