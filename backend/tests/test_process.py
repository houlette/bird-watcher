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
    crop: object = None  # populated by process_visit at YOLO time in real flow


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

        # The tracked detection's crop will be populated by process_visit at YOLO
        # time, so the rest of the pipeline can read d.crop directly. We mirror
        # that by giving the same _FakeDetection instance to both detect_birds
        # and the fake Tracker (so the crop assignment is visible downstream).
        tracked_det = _FakeDetection()

        monkeypatch.setattr(process_module, "DATA_DIR", tmp_path)
        monkeypatch.setattr(process_module, "extract_frames", lambda _p, target_fps=6.0: iter([_Frame(image=fake_frame_image)]))
        monkeypatch.setattr(process_module, "detect_birds", lambda _img, _idx: [tracked_det])
        # Bypass the IoU tracker entirely — return the same det instance so
        # `det.crop` assigned in process_visit's loop is visible to ranking.
        monkeypatch.setattr(process_module, "Tracker", lambda: _FakeTracker([_FakeTrack(track_id=1, detections=[tracked_det])]))
        # Classifier rejects (returns []) — the new code path under test.
        monkeypatch.setattr(process_module, "classify_bird", lambda _img: [])
        # Don't actually write image files.
        monkeypatch.setattr(process_module, "_save_crop", lambda _d, *, visit_id, track_id: Path(f"crops/v{visit_id}_t{track_id}.jpg"))
        # Bypass the sharpness-aware ranking and just hand back the track's detections in order.
        monkeypatch.setattr(process_module, "_rank_detections", lambda track: list(track.detections))
        # _extract_crop_from_image is called per-detection during the frame loop.
        monkeypatch.setattr(process_module, "_extract_crop_from_image", lambda _d, _img, padding=0.15: fake_frame_image)
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


def test_laplacian_variance_higher_for_sharp_image():
    """Sanity check: a sharp checkerboard scores higher than uniform gray."""
    sharp = np.zeros((100, 100, 3), dtype=np.uint8)
    sharp[::2, ::2] = 255  # checkerboard pattern, lots of high-frequency content
    blurry = np.full((100, 100, 3), 128, dtype=np.uint8)  # uniform gray, no edges

    sharp_var = process_module._laplacian_variance(sharp)
    blurry_var = process_module._laplacian_variance(blurry)
    assert sharp_var > blurry_var * 10  # at least 10× ratio


def test_rank_detections_prefers_sharper_crops(tmp_path):
    """Two detections with the same bbox area and confidence should be ranked
    by sharpness — the sharper crop's source detection wins."""
    sharp_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp_frame[::2, ::2] = 255  # checkerboard — high Laplacian variance
    blurry_frame = np.full((200, 200, 3), 128, dtype=np.uint8)  # uniform gray

    d_blurry = _FakeDetection(bbox=(0, 0, 100, 100), confidence=0.9, frame_index=0, crop=blurry_frame)
    d_sharp = _FakeDetection(bbox=(0, 0, 100, 100), confidence=0.9, frame_index=1, crop=sharp_frame)
    track = _FakeTrack(track_id=1, detections=[d_blurry, d_sharp])

    ranked = process_module._rank_detections(track)
    assert ranked[0] is d_sharp
    assert ranked[1] is d_blurry
