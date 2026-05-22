"""Tests for the per-track classifier-rejection path in process_visit.

The full end-to-end (real YOLO, real classifier, real video file) is
covered by scripts/smoke_test.py; here we monkeypatch the heavy bits
and verify only the new behavior: tracks where the classifier rejects
every crop still produce a Detection row with species_id=None so the
user can hand-label them in the feed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Detection, Visit
from db.session import Base
from pipeline import process as process_module


@dataclass
class _FakeDetection:
    bbox: tuple[int, int, int, int] = (100, 100, 60, 60)
    confidence: float = 0.85
    frame_index: int = 0


@dataclass
class _FakeTrack:
    track_id: int
    detections: list

    @property
    def best_detection(self):
        # Mirror pipeline.track.Track's selector — area × confidence.
        return max(self.detections, key=lambda d: d.bbox[2] * d.bbox[3] * d.confidence)


class _FakeTracker:
    def __init__(self, tracks):
        self._tracks = tracks

    def update(self, *_args, **_kwargs):
        pass

    def finalize(self):
        return self._tracks


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def _make_visit(session_maker, tmp_path) -> Visit:
    # process_visit needs a real file on disk so frames.extract_frames can
    # open it. We monkeypatch extract_frames to ignore the contents but it
    # still checks existence.
    clip = tmp_path / "clip.jpg"
    clip.write_bytes(b"fake")
    session = session_maker()
    try:
        visit = Visit(clip_path=str(clip.relative_to(tmp_path)))
        session.add(visit)
        session.commit()
        session.refresh(visit)
        return visit, session
    except Exception:
        session.close()
        raise


def test_classifier_rejection_still_creates_detection(db, tmp_path, monkeypatch):
    """A YOLO-detected track whose crops are all classifier-rejected should
    still appear in the feed (as Unidentified) so the user can hand-label."""
    visit, session = _make_visit(db, tmp_path)
    try:
        # Pretend a clip with one frame and one tracked bird.
        fake_frame_image = np.zeros((100, 100, 3), dtype=np.uint8)

        @dataclass
        class _Frame:
            index: int = 0
            timestamp: float = 0.0
            image: np.ndarray = None

        monkeypatch.setattr(process_module, "DATA_DIR", tmp_path)
        monkeypatch.setattr(process_module, "extract_frames", lambda _p, target_fps=3.0: iter([_Frame(image=fake_frame_image)]))
        monkeypatch.setattr(process_module, "detect_birds", lambda _img, _idx: [_FakeDetection()])
        # Bypass the IoU tracker entirely.
        monkeypatch.setattr(process_module, "Tracker", lambda: _FakeTracker([_FakeTrack(track_id=1, detections=[_FakeDetection()])]))
        # Classifier rejects (returns []) — the new code path under test.
        monkeypatch.setattr(process_module, "classify_bird", lambda _img: [])
        # Don't actually write image files.
        monkeypatch.setattr(process_module, "_save_best_crop", lambda _t, _f, *, visit_id: (Path(f"crops/v{visit_id}_t1.jpg"), fake_frame_image))
        # Don't try to push notifications (no species anyway).
        monkeypatch.setattr(process_module, "dispatch_for_detection", lambda *_a, **_k: 0)

        process_module.process_visit(visit, session)

        dets = session.query(Detection).filter_by(visit_id=visit.id).all()
        assert len(dets) == 1
        d = dets[0]
        assert d.species_id is None
        assert d.confidence == 0.0
        assert d.crop_path == f"crops/v{visit.id}_t1.jpg"
        assert d.raw_predictions == []
    finally:
        session.close()
