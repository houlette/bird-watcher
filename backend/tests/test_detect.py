"""Unit tests for the pure helpers in pipeline.detect (no YOLO weights needed).

Covers the Non-Maximum Merging (NMM) replacement for standard NMS, which
handles the tile-seam fragment case (two halves of one bird detected
across adjacent tiles) in addition to overlapping duplicates.
"""
from __future__ import annotations

from pipeline.detect import (
    BirdDetection,
    _box_union,
    _is_tile_fragment_pair,
    _nmm,
)


def _det(bbox, conf=0.5, frame_index=0):
    return BirdDetection(bbox=bbox, confidence=conf, frame_index=frame_index)


def test_box_union_contains_both():
    a = (10, 20, 30, 40)   # x∈[10,40], y∈[20,60]
    b = (25, 50, 50, 30)   # x∈[25,75], y∈[50,80]
    assert _box_union(a, b) == (10, 20, 65, 60)


def test_tile_fragment_pair_horizontal_seam_merged():
    """Side-by-side halves with a tiny x-gap and matching y-extent should merge."""
    left_half = (900, 500, 120, 200)    # x∈[900,1020]
    right_half = (1030, 500, 120, 200)  # x∈[1030,1150]  → gap of 10 px at the seam
    assert _is_tile_fragment_pair(left_half, right_half) is True


def test_tile_fragment_pair_vertical_seam_merged():
    """Stacked halves with a tiny y-gap and matching x-extent should merge."""
    top_half = (500, 900, 200, 120)
    bottom_half = (500, 1030, 200, 120)
    assert _is_tile_fragment_pair(top_half, bottom_half) is True


def test_tile_fragment_pair_real_separate_birds_not_merged():
    """Two birds with a real gap (> TILE_SEAM_GAP_PX) should NOT be flagged as fragments."""
    bird_a = (900, 500, 120, 200)   # x∈[900,1020]
    bird_b = (1080, 500, 120, 200)  # x∈[1080,1200]  → gap of 60 px
    assert _is_tile_fragment_pair(bird_a, bird_b) is False


def test_tile_fragment_pair_misaligned_not_merged():
    """Two boxes touching on x-axis but at very different y-positions are not fragments."""
    a = (900, 500, 120, 200)    # y∈[500,700]
    b = (1030, 900, 120, 200)   # y∈[900,1100]  → no y-overlap
    assert _is_tile_fragment_pair(a, b) is False


def test_nmm_merges_overlapping_duplicates_into_union():
    """A fully-detected bird in tile A and a partial duplicate in tile B's overlap zone
    should merge into the union bbox (not just drop the lower-confidence one)."""
    full = _det((100, 100, 200, 200), conf=0.9)
    partial = _det((150, 100, 200, 200), conf=0.6)   # high IoU with `full`
    out = _nmm([full, partial], iou_thresh=0.50)
    assert len(out) == 1
    # Union covers x∈[100,350], y∈[100,300]
    assert out[0].bbox == (100, 100, 250, 200)
    assert out[0].confidence == 0.9


def test_nmm_merges_tile_seam_fragments():
    """The motivating case: two half-bird detections from adjacent tiles
    should merge into a single bbox covering the whole bird."""
    left = _det((900, 500, 120, 200), conf=0.55)
    right = _det((1030, 500, 120, 200), conf=0.45)
    out = _nmm([left, right], iou_thresh=0.50)
    assert len(out) == 1
    # Combined extent: x∈[900,1150], y∈[500,700]
    assert out[0].bbox == (900, 500, 250, 200)


def test_nmm_keeps_genuinely_separate_birds():
    """Two birds with a real gap (no overlap, far apart) must both survive."""
    a = _det((100, 100, 100, 100), conf=0.8)
    b = _det((500, 500, 100, 100), conf=0.7)
    out = _nmm([a, b], iou_thresh=0.50)
    assert len(out) == 2


def test_nmm_empty_input():
    assert _nmm([], iou_thresh=0.50) == []
