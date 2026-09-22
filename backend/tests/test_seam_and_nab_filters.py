"""Unit tests for tile-seam crop expansion, squirrel/NAB suppression, and vagrant penalties."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import NOT_A_BIRD_LABEL, HaikuboxDetection, Visit
from db.session import Base
from pipeline.detect import BirdDetection
from pipeline.fuse import fuse
from pipeline.process import _extract_crop_from_image, process_visit
from settings import settings


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


def test_tile_seam_top_crop_extension():
    """A bird clipped at the top tile seam (y=820, such as a pigeon on the feeder)
    should extend upward across the seam to capture the rest of the bird."""
    full_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    # Stubby tail/rump detection at y=820
    det = BirdDetection(bbox=(920, 820, 143, 58), confidence=0.50, frame_index=0)
    crop = _extract_crop_from_image(det, full_frame)
    # Without seam extension, height would only be ~92-123 px.
    # With seam extension across y=820, height extends upward to include head & feeder body >= 250 px.
    assert crop.shape[0] >= 250
    assert crop.shape[1] >= 200


def test_tile_seam_explicit_clipped_edges():
    """BirdDetection with clipped_edges should extend even if coordinate is slightly offset."""
    full_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    det = BirdDetection(
        bbox=(500, 823, 100, 50),
        confidence=0.50,
        frame_index=0,
        clipped_edges={"top"},
    )
    crop = _extract_crop_from_image(det, full_frame)
    assert crop.shape[0] >= 220


def test_normal_bird_not_extended_unnecessarily():
    """A normal bird situated away from tile seams retains standard padding."""
    full_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    det = BirdDetection(bbox=(1500, 500, 100, 100), confidence=0.85, frame_index=0)
    crop = _extract_crop_from_image(det, full_frame)
    # 100 px + 30% padding on each side = 160 px
    assert crop.shape[0] == 160
    assert crop.shape[1] == 160


def test_vagrant_species_downweighted_in_fuse(db, monkeypatch):
    """Exotic species (e.g. Phainopepla, Pelagic Cormorant) that are not regional backyard birds
    should receive a 5x downweight in fuse unless confirmed by audio."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "")

    # Visual classifier outputs 20% Phainopepla and 15% House Finch
    predictions = [("Phainopepla", 0.20), ("House Finch", 0.15)]
    fused = fuse(predictions, db=db)

    # House Finch is a common backyard bird; Phainopepla is an exotic desert bird.
    # The vagrant penalty (0.20) should drop Phainopepla below House Finch.
    top = fused[0]
    assert top.species == "House Finch"


def test_vagrant_species_wins_when_audio_confirmed(db, monkeypatch):
    """If an exotic bird is actually heard on Haikubox, audio confirmation preserves it."""
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_api_key", "fake")
    monkeypatch.setattr("pipeline.fuse.settings.haikubox_serial", "fake")
    now = datetime(2026, 5, 1, 12, 0, 0)
    db.add(HaikuboxDetection(species_common_name="Phainopepla", detected_at=now))
    db.commit()

    predictions = [("Phainopepla", 0.30), ("House Finch", 0.30)]
    fused = fuse(predictions, db=db, when=now)
    assert fused[0].species == "Phainopepla"
    assert fused[0].audio_confirmed is True


@dataclass
class _FakeDetection:
    bbox: tuple[int, int, int, int] = (100, 100, 60, 60)
    confidence: float = 0.85
    frame_index: int = 0
    crop: Any = None
    clipped_edges: set = None


@dataclass
class _FakeTrack:
    track_id: int
    detections: list

    @property
    def best_detection(self):
        return self.detections[0]


class _FakeTracker:
    def __init__(self, tracks):
        self._tracks = tracks

    def update(self, *_args, **_kwargs):
        pass

    def finalize(self):
        return self._tracks


def _run_mock_visit(db, monkeypatch, tmp_path, species_preds, nab_fused, nab_single):
    from pipeline import process as process_module

    clip_file = tmp_path / "clips/test.mp4"
    clip_file.parent.mkdir(parents=True, exist_ok=True)
    clip_file.write_bytes(b"dummy")

    visit = Visit(clip_path="clips/test.mp4")
    db.add(visit)
    db.commit()
    db.refresh(visit)

    fake_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    tracked_det = _FakeDetection(crop=fake_crop)

    @dataclass
    class _Frame:
        index: int = 0
        timestamp: float = 0.0
        image: Any = None

    frame_img = np.zeros((200, 200, 3), dtype=np.uint8)

    monkeypatch.setattr(process_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(process_module, "extract_frames", lambda *_a, **_k: iter([_Frame(image=frame_img)]))
    monkeypatch.setattr(process_module, "detect_birds", lambda *_a, **_k: [tracked_det])
    monkeypatch.setattr(process_module, "Tracker", lambda: _FakeTracker([_FakeTrack(1, [tracked_det])]))
    monkeypatch.setattr(process_module, "classify_bird", lambda _img: species_preds)
    monkeypatch.setattr(process_module, "_save_crop", lambda _d, **_k: Path("crops/test.jpg"))
    monkeypatch.setattr(process_module, "_rank_detections", lambda track: list(track.detections))
    monkeypatch.setattr(process_module, "_extract_crop_from_image", lambda _d, _img, **_k: fake_crop)
    monkeypatch.setattr(process_module, "_save_source_frames", lambda *_a, **_k: None)
    monkeypatch.setattr(process_module, "dispatch_for_detection", lambda *_a, **_k: 0)

    monkeypatch.setattr(process_module, "binary_filter_enabled", lambda: True)
    monkeypatch.setattr(process_module, "nab_probability", lambda _img: nab_fused)
    monkeypatch.setattr(process_module, "_score_crop_variants", lambda _b, _f, _s: (nab_single, None))

    process_visit(visit, db)
    from db.models import Detection, Species
    dets = db.query(Detection).filter(Detection.visit_id == visit.id).all()
    assert len(dets) == 1
    d = dets[0]
    sp = db.get(Species, d.species_id).common_name if d.species_id else None
    return d, sp


def test_squirrel_overridden_when_single_crop_confident_nab(db, monkeypatch, tmp_path):
    """When fusion blurs moving squirrel (nab_fused=0.52), but sharp single crop is 0.85 (>0.65),
    effective NAB overrides to Not a bird."""
    from pipeline.classify import SpeciesPrediction
    species_preds = [SpeciesPrediction(species="House Sparrow", probability=0.40, raw_label="House Sparrow")]
    d, sp = _run_mock_visit(db, monkeypatch, tmp_path, species_preds, nab_fused=0.52, nab_single=0.85)
    assert sp == NOT_A_BIRD_LABEL


def test_squirrel_overridden_when_weak_visual_and_moderate_nab(db, monkeypatch, tmp_path):
    """When visual confidence is weak (<0.35) and NAB is moderate (>=0.40), override to Not a bird."""
    from pipeline.classify import SpeciesPrediction
    # Weak visual prediction on squirrel (e.g. 10% on random bird)
    species_preds = [SpeciesPrediction(species="House Sparrow", probability=0.10, raw_label="House Sparrow")]
    d, sp = _run_mock_visit(db, monkeypatch, tmp_path, species_preds, nab_fused=0.45, nab_single=0.42)
    assert sp == NOT_A_BIRD_LABEL


def test_unidentified_when_very_weak_visual_and_no_audio(db, monkeypatch, tmp_path):
    """When visual confidence is very weak (<0.20) and no audio confirmed, don't hallucinate species."""
    from pipeline.classify import SpeciesPrediction
    species_preds = [SpeciesPrediction(species="House Sparrow", probability=0.12, raw_label="House Sparrow")]
    # Very low NAB (0.15) so not overridden to NAB, but visual is too weak to identify species
    d, sp = _run_mock_visit(db, monkeypatch, tmp_path, species_preds, nab_fused=0.15, nab_single=0.15)
    assert sp is None  # Persisted as Unidentified


def test_confident_bird_not_overridden(db, monkeypatch, tmp_path):
    """A real bird with high visual confidence and low NAB is identified normally."""
    from pipeline.classify import SpeciesPrediction
    species_preds = [SpeciesPrediction(species="Northern Cardinal", probability=0.90, raw_label="Northern Cardinal")]
    d, sp = _run_mock_visit(db, monkeypatch, tmp_path, species_preds, nab_fused=0.05, nab_single=0.05)
    assert sp == "Northern Cardinal"
