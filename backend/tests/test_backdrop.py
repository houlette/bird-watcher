"""Unit tests for the per-hour backdrop model."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import pytest

from pipeline import backdrop


@dataclass
class _FakeDet:
    bbox: tuple[int, int, int, int]
    confidence: float = 0.5


@pytest.fixture()
def yard(tmp_path, monkeypatch):
    """A synthetic fixed scene: textured backdrop, no foreground."""
    monkeypatch.setattr(backdrop, "BACKDROP_DIR", tmp_path / "backdrop")
    monkeypatch.setattr(backdrop, "FRAMES_DIR", tmp_path / "frames")
    backdrop._cache.clear()
    backdrop._cache_stamp.clear()
    rng = np.random.default_rng(7)
    base = rng.integers(60, 120, size=(540, 960), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (9, 9), 0)
    return base


def _write_frames(tmp_path, base, n, hour=21, with_bird_at=None):
    """n full-size frames of the scene, birds in varied places so the median
    still converges on the empty yard."""
    d = tmp_path / "frames"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    paths = []
    for i in range(n):
        frame = base.copy()
        # A bird somewhere different each time: the thing a median removes.
        bx, by = int(rng.integers(50, 800)), int(rng.integers(50, 450))
        cv2.rectangle(frame, (bx, by), (bx + 30, by + 30), 240, -1)
        if with_bird_at is not None and i == 0:
            x, y, w, h = [v // backdrop.SCALE for v in with_bird_at]
            cv2.rectangle(frame, (x, y), (x + w, y + h), 250, -1)
        big = cv2.resize(frame, (960 * backdrop.SCALE, 540 * backdrop.SCALE),
                         interpolation=cv2.INTER_NEAREST)
        p = d / f"v{i:08d}_t0001.jpg"
        cv2.imwrite(str(p), big)
        paths.append(p)
    return {hour: paths}


def test_median_removes_transient_objects(tmp_path, yard):
    """The whole premise: birds in varied positions average out."""
    frames = _write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10)
    used = backdrop.rebuild(frames)
    assert used[21] >= backdrop.MIN_FRAMES_PER_HOUR
    model = backdrop.get_backdrop(21)
    assert model is not None
    # No bright bird-sized blob survived into the model.
    assert float((model > 200).mean()) < 0.01


def test_too_few_frames_builds_nothing(tmp_path, yard):
    """A median over a handful of frames can keep a bird in it, which would
    then read as backdrop forever. Better to have no model for that hour."""
    frames = _write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR - 1)
    assert backdrop.rebuild(frames) == {}
    assert backdrop.get_backdrop(21) is None


def test_backdrop_box_scores_near_one(tmp_path, yard):
    """A box containing nothing but scene differs no more than its ring."""
    frames = _write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10)
    backdrop.rebuild(frames)
    model = backdrop.get_backdrop(21)
    gray = yard.copy()
    r = backdrop.contrast_ratio(gray, model, (2000, 1200, 300, 300))
    assert r is not None
    assert r < backdrop.MIN_CONTRAST_RATIO


def test_object_box_scores_well_above_one(tmp_path, yard):
    """A box containing something that is not in the model stands out."""
    frames = _write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10)
    backdrop.rebuild(frames)
    model = backdrop.get_backdrop(21)
    gray = yard.copy()
    cv2.rectangle(gray, (500, 300), (560, 360), 255, -1)
    r = backdrop.contrast_ratio(gray, model, (500 * backdrop.SCALE, 300 * backdrop.SCALE,
                                              60 * backdrop.SCALE, 60 * backdrop.SCALE))
    assert r > backdrop.MIN_CONTRAST_RATIO * 2


def test_filter_reports_scored_count_so_silence_is_readable(tmp_path, yard):
    """With no model for the hour, nothing is suppressed AND nothing is
    scored — the pair is what distinguishes "clean" from "not running"."""
    backdrop._cache.clear()
    frame = cv2.cvtColor(cv2.resize(yard, (960 * backdrop.SCALE, 540 * backdrop.SCALE)),
                         cv2.COLOR_GRAY2BGR)
    d = _FakeDet(bbox=(2000, 1200, 300, 300))
    kept, suppressed, scored = backdrop.filter_detections([d], frame, datetime(2026, 9, 13, 4))
    assert kept == [d] and suppressed == 0 and scored == 0


def test_filter_drops_a_pure_backdrop_box(tmp_path, yard, monkeypatch):
    from settings import settings
    monkeypatch.setattr(settings, "backdrop_filter_enabled", True)
    frames = _write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10)
    backdrop.rebuild(frames)
    frame = cv2.cvtColor(cv2.resize(yard, (960 * backdrop.SCALE, 540 * backdrop.SCALE),
                                    interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    d = _FakeDet(bbox=(2000, 1200, 300, 300))
    kept, suppressed, scored = backdrop.filter_detections([d], frame, datetime(2026, 9, 13, 21))
    assert suppressed == 1 and scored == 1 and kept == []


def test_visit_id_parsed_from_frame_name(tmp_path):
    assert backdrop.visit_id_from_frame(tmp_path / "v00251200_t0001.jpg") == 251200
    assert backdrop.visit_id_from_frame(tmp_path / "crop_12.jpg") is None


def test_frames_bin_on_capture_time_not_write_time(tmp_path):
    """The bug this replaced: a frame written at 15:00 but shot at 13:00
    belongs in the 13:00 median, or that hour's model blends two lightings."""
    shot_at_13 = tmp_path / "v00000001_t0001.jpg"
    shot_at_21 = tmp_path / "v00000002_t0001.jpg"
    captured = {
        1: datetime(2026, 9, 6, 13, 5, 5),
        2: datetime(2026, 9, 6, 21, 40, 0),
    }
    bins, dropped = backdrop.bin_by_capture_hour([shot_at_13, shot_at_21], captured)
    assert dropped == 0
    assert bins[13] == [shot_at_13]
    assert bins[21] == [shot_at_21]


def test_frame_without_a_visit_is_dropped_not_guessed(tmp_path):
    """Outside the rebuild window, or orphaned by a deleted visit. Either
    way its hour is unknown, and an unknown hour is what poisoned the
    median before."""
    bins, dropped = backdrop.bin_by_capture_hour(
        [tmp_path / "v00000009_t0001.jpg", tmp_path / "not_a_frame.jpg"], {})
    assert bins == {} and dropped == 2


def test_rebuild_prunes_an_hour_it_can_no_longer_build(tmp_path, yard):
    """A model left over from a previous rebuild reads as current forever."""
    backdrop.rebuild(_write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10, hour=21))
    assert backdrop.get_backdrop(21) is not None
    backdrop.rebuild(_write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10, hour=13))
    assert backdrop.get_backdrop(13) is not None
    assert backdrop.get_backdrop(21) is None


def test_a_rebuild_that_builds_nothing_keeps_what_is_there(tmp_path, yard):
    """An unmounted frames volume must not wipe every hour's model."""
    backdrop.rebuild(_write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10, hour=21))
    assert backdrop.rebuild({}) == {}
    assert backdrop.get_backdrop(21) is not None


def test_filter_is_off_unless_explicitly_enabled(tmp_path, yard, monkeypatch):
    """It removed 37.5% of confirmed birds to remove 34.0% of confirmed junk
    on June 2026, so it stays off until the mechanism changes."""
    from settings import settings
    frames = _write_frames(tmp_path, yard, backdrop.MIN_FRAMES_PER_HOUR + 10)
    backdrop.rebuild(frames)
    frame = cv2.cvtColor(cv2.resize(yard, (960 * backdrop.SCALE, 540 * backdrop.SCALE),
                                    interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    d = _FakeDet(bbox=(2000, 1200, 300, 300))
    monkeypatch.setattr(settings, "backdrop_filter_enabled", False)
    kept, suppressed, scored = backdrop.filter_detections([d], frame, datetime(2026, 9, 13, 21))
    # scored == 0 is the existing "did not run" signal, not "found nothing".
    assert kept == [d] and suppressed == 0 and scored == 0
    monkeypatch.setattr(settings, "backdrop_filter_enabled", True)
    kept, suppressed, scored = backdrop.filter_detections([d], frame, datetime(2026, 9, 13, 21))
    assert suppressed == 1 and scored == 1 and kept == []
