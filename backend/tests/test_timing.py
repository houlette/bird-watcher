"""Stage timing and the counts that make it interpretable."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from pipeline.frames import extract_frames
from pipeline.timing import TIMINGS_VERSION, StageTimer


def test_stages_accumulate_and_leave_the_rest_unaccounted():
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("detect"):
            time.sleep(0.01)
    time.sleep(0.02)  # untimed work
    d = timer.as_dict()

    assert d["version"] == TIMINGS_VERSION
    assert 0.03 <= d["stages_s"]["detect"] < 0.2
    assert d["unaccounted_s"] >= 0.015
    assert abs(d["total_s"] - d["stages_s"]["detect"] - d["unaccounted_s"]) < 0.002


def test_a_stage_is_charged_even_when_its_body_raises():
    timer = StageTimer()
    with pytest.raises(ValueError):
        with timer.stage("persist"):
            time.sleep(0.01)
            raise ValueError
    assert timer.as_dict()["stages_s"]["persist"] >= 0.01


def test_iterate_charges_producing_items_not_consuming_them():
    def slow_source():
        for i in range(2):
            time.sleep(0.02)
            yield i

    timer = StageTimer()
    seen = []
    for item in timer.iterate(slow_source(), "decode"):
        seen.append(item)
        time.sleep(0.03)  # the consumer's work is not decode

    stages = timer.as_dict()["stages_s"]
    assert seen == [0, 1]
    assert 0.04 <= stages["decode"] < 0.055


def test_summary_puts_the_largest_stage_first():
    timer = StageTimer()
    timer.seconds.update(detect=2.0, decode=5.0, track=0.01)
    timer.count("frames", 30)
    line = timer.summary()
    assert line.index("decode") < line.index("detect")
    assert "track" not in line and "30 frames" in line


def test_extract_frames_counts_every_source_frame_it_decodes(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("no mp4v encoder in this OpenCV build")
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()

    stats: dict = {}
    frames = list(extract_frames(path, target_fps=3.0, stats=stats))

    # 20 fps sampled every 7th frame: indices 0, 7, 14, 21, 28.
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert stats["source_frames_read"] == 30
    assert (stats["width"], stats["height"]) == (64, 48)
    assert stats["source_fps"] == pytest.approx(20.0)
