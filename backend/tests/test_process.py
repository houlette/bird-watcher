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
        def fake_extract(_p, target_fps=3.0, stats=None):
            stats.update(source_frames_read=7, source_fps=20.0, width=100, height=100)
            return iter([_Frame(image=fake_frame_image)])

        def fake_detect(_img, _idx, stats=None, **_kwargs):
            if stats is not None:
                stats["tiles"] = stats.get("tiles", 0) + 15
            return [tracked_det]

        monkeypatch.setattr(process_module, "extract_frames", fake_extract)
        monkeypatch.setattr(process_module, "detect_birds", fake_detect)
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
        # Source-frame archive write — pre-empt the cv2.VideoCapture call since
        # our fake clip is a JPEG with .jpg fake bytes, not a real MP4.
        monkeypatch.setattr(process_module, "_save_source_frames", lambda *_a, **_k: None)
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

        # Every stage that ran is timed, the stages never exceed the total,
        # and the frame, source-frame and tile counts reach the row.
        t = session.get(Visit, visit.id).timings
        assert {"decode", "detect", "scene_mask", "recurrence", "backdrop", "crop_extract", "track",
                "rank", "save_crop", "fuse_crops", "classify", "crop_quality", "persist",
                "source_frames", "push"} <= set(t["stages_s"])
        assert all(v >= 0 for v in t["stages_s"].values())
        assert t["unaccounted_s"] >= 0
        assert t["counts"]["frames"] == 1 and t["counts"]["tiles"] == 15 and t["counts"]["tracks"] == 1
        assert t["counts"]["detect_voluntary_switches"] >= 0
        assert t["clip"]["source_frames_read"] == 7 and t["clip"]["ext"] == ".jpg"
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


# --- crop-representation experiment (Detection.nab_p_served) --------------

def _solid(h, w, value):
    import numpy as np
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_fuse_reports_how_many_crops_it_averaged():
    """fusion_n_used == 1 means fusion fell back to the anchor, so that row
    says nothing about whether fusion helps and must be excluded."""
    from pipeline import process

    stats: dict = {}
    process._fuse_crops([_solid(40, 40, 100)], stats=stats)
    assert stats["n_used"] == 1

    stats = {}
    process._fuse_crops([_solid(40, 40, 100), _solid(40, 40, 102)], stats=stats)
    assert stats["n_used"] == 2


def test_fuse_without_stats_still_works():
    from pipeline import process

    assert process._fuse_crops([_solid(40, 40, 100)]) is not None


def test_variant_scoring_reuses_the_served_probability_when_fusion_is_off(monkeypatch):
    """With fusion off the served crop IS the raw crop, so scoring it again
    would pay for the same answer twice."""
    from pipeline import process

    calls = []

    def fake(img):
        calls.append(img)
        return 0.42

    monkeypatch.setattr(process, "nab_probability", fake)

    class _Best:
        crop = _solid(40, 40, 100)

    best = _Best()
    single, polished = process._score_crop_variants(best, best.crop, 0.9)
    assert single == 0.9              # reused, not re-scored
    assert polished == 0.42
    assert len(calls) == 1            # only the polished crop was scored


def test_variant_scoring_scores_both_when_the_served_crop_is_fused(monkeypatch):
    from pipeline import process

    monkeypatch.setattr(process, "nab_probability", lambda img: 0.3)

    class _Best:
        crop = _solid(40, 40, 100)

    best = _Best()
    fused = _solid(260, 260, 101)
    single, polished = process._score_crop_variants(best, fused, 0.9)
    assert single == 0.3 and polished == 0.3


def test_variant_scoring_tolerates_a_missing_crop():
    from pipeline import process

    class _Best:
        crop = None

    assert process._score_crop_variants(_Best(), None, 0.5) == (None, None)


def test_process_visit_with_motion_gated_tiles_enabled(db, tmp_path, monkeypatch):
    """process_visit passes selected_tiles to detect_birds when motion_gated_tiles_enabled is True."""
    from settings import settings
    monkeypatch.setattr(settings, "motion_gated_tiles_enabled", True)

    visit, session = _make_visit(db, tmp_path)
    try:
        fake_frame_image = np.zeros((100, 100, 3), dtype=np.uint8)

        @dataclass
        class _Frame:
            index: int = 0
            timestamp: float = 0.0
            image: np.ndarray = None

        tracked_det = _FakeDetection()
        detected_tiles = []

        monkeypatch.setattr(process_module, "DATA_DIR", tmp_path)

        def fake_extract(_p, target_fps=3.0, stats=None):
            stats.update(source_frames_read=1, source_fps=20.0, width=100, height=100)
            return iter([_Frame(image=fake_frame_image)])

        def fake_detect(_img, _idx, stats=None, backend=None, tiles=None):
            detected_tiles.append(tiles)
            return [tracked_det]

        monkeypatch.setattr(process_module, "extract_frames", fake_extract)
        monkeypatch.setattr(process_module, "detect_birds", fake_detect)
        monkeypatch.setattr(process_module, "Tracker", lambda: _FakeTracker([_FakeTrack(track_id=1, detections=[tracked_det])]))
        monkeypatch.setattr(process_module, "classify_bird", lambda _img: [])
        monkeypatch.setattr(process_module, "_save_crop", lambda _d, *, visit_id, track_id: Path(f"crops/v{visit_id}_t{track_id}.jpg"))
        monkeypatch.setattr(process_module, "_rank_detections", lambda track: list(track.detections))
        monkeypatch.setattr(process_module, "_extract_crop_from_image", lambda _d, _img, padding=0.15: fake_frame_image)
        monkeypatch.setattr(process_module, "_save_source_frames", lambda *_a, **_k: None)
        monkeypatch.setattr(process_module, "dispatch_for_detection", lambda *_a, **_k: 0)

        process_module.process_visit(visit, session)

        assert len(detected_tiles) == 1
        # On frame 0 (keyframe), all tiles for 100x100 frame are evaluated
        assert detected_tiles[0] == [(0, 0, 100, 100)]
    finally:
        session.close()
