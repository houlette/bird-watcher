"""Unit tests for Phase 0 motion-gated spatial tiling engine.

Tests select_active_tiles_hardened, seam clustering, signed illumination
invariance, velocity envelope clamping, and recent perch memory.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.detect import BirdDetection, _tile_offsets
from pipeline.motion_tiles import (
    LOWRES_HEIGHT,
    LOWRES_WIDTH,
    _box_intersects,
    select_active_tiles_hardened,
)
from pipeline.track import Track


def _det(bbox: tuple[int, int, int, int], frame_index: int = 0) -> BirdDetection:
    return BirdDetection(bbox=bbox, confidence=0.85, frame_index=frame_index)


def test_box_intersects_cases():
    # Identical
    assert _box_intersects((0, 0, 100, 100), (0, 0, 100, 100)) is True
    # Overlapping
    assert _box_intersects((0, 0, 100, 100), (50, 50, 100, 100)) is True
    # Touching edge (no overlap)
    assert _box_intersects((0, 0, 100, 100), (100, 0, 100, 100)) is False
    assert _box_intersects((0, 0, 100, 100), (0, 100, 100, 100)) is False
    # Disjoint
    assert _box_intersects((0, 0, 100, 100), (200, 200, 50, 50)) is False


def test_keyframe_and_none_history_evaluates_all_tiles():
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    assert len(all_tiles) == 15

    frame0 = np.full((h_full, w_full, 3), 128, dtype=np.uint8)
    perch_memory = {}

    # Frame 0: keyframe and prev_gray is None
    tiles, low_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory=perch_memory,
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert len(tiles) == 15
    assert low_gray.shape == (LOWRES_HEIGHT, LOWRES_WIDTH)

    # Frame 3: keyframe (index % 3 == 0) with prev_gray present
    tiles_kf, _ = select_active_tiles_hardened(
        prev_gray=low_gray,
        curr_bgr=frame0,
        frame_index=3,
        active_tracks=[],
        perch_memory=perch_memory,
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert len(tiles_kf) == 15


def test_no_motion_yields_no_tiles_between_keyframes():
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame = np.full((h_full, w_full, 3), 128, dtype=np.uint8)

    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # Frame 1: Identical frame -> zero motion
    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame,
        frame_index=1,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert tiles == []


def test_motion_in_tile_activates_that_tile():
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame0 = np.full((h_full, w_full, 3), 128, dtype=np.uint8)
    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # Introduce a 60x60 px bright patch in the interior of tile 0 (x=500, y=500)
    frame1 = frame0.copy()
    frame1[500:560, 500:560] = 220

    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame1,
        frame_index=1,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    # Tile 0 is (0, 0, 1024, 1024). Interior patch (500, 500) is far from seam margins (~205 px).
    assert tiles == [all_tiles[0]]


def test_sub_threshold_motion_area_ignored():
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame0 = np.full((h_full, w_full, 3), 128, dtype=np.uint8)
    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # 4x4 px patch in 4K -> after 4:1 INTER_AREA downscale, becomes 1 pixel
    frame1 = frame0.copy()
    frame1[500:504, 500:504] = 220

    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame1,
        frame_index=1,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert tiles == []


def test_signed_illumination_adjustment_suppresses_cloud_shifts():
    """A uniform exposure or cloud illumination shift must not trigger all tiles."""
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame0 = np.full((h_full, w_full, 3), 120, dtype=np.uint8)
    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # Whole frame brightens by +25 (e.g. camera auto-exposure or sun breaking through)
    frame1 = np.full((h_full, w_full, 3), 145, dtype=np.uint8)

    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame1,
        frame_index=1,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    # The signed median delta cancels the +25 shift, resulting in 0 active tiles
    assert tiles == []


def test_motion_with_cloud_shift_still_detected():
    """True bird motion on top of an auto-exposure shift must still be detected."""
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame0 = np.full((h_full, w_full, 3), 120, dtype=np.uint8)
    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # Frame brightens by +20 globally, PLUS a moving bird (dark patch) in tile 0
    frame1 = np.full((h_full, w_full, 3), 140, dtype=np.uint8)
    frame1[400:480, 400:480] = 50

    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame1,
        frame_index=1,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert tiles == [all_tiles[0]]


def test_seam_proximity_expands_to_neighboring_tile():
    """Motion touching the tile overlap margin (205 px) must pull in the neighboring tile."""
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    # Tile 0: (0, 0, 1024, 1024)
    # Tile 1: (819, 0, 1024, 1024) - starts at 1024 - 205 = 819
    frame0 = np.full((h_full, w_full, 3), 128, dtype=np.uint8)
    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # Motion right at x=950, y=500 (inside Tile 0, and within 205 px of Tile 0's right edge 1024)
    frame1 = frame0.copy()
    frame1[500:560, 950:1010] = 220

    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame1,
        frame_index=1,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    # Both tile 0 and tile 1 must be selected!
    assert all_tiles[0] in tiles
    assert all_tiles[1] in tiles
    assert len(tiles) == 2


def test_active_track_persistence_and_velocity_envelope():
    """Active tracks keep tiles active even without pixel motion, and expand velocity envelope."""
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame0 = np.full((h_full, w_full, 3), 128, dtype=np.uint8)
    _, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=0,
        active_tracks=[],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )

    # Track moving from (400, 400) to (700, 400) (vx = 300)
    t = Track(
        track_id=1,
        detections=[
            _det((400, 400, 80, 80), frame_index=0),
            _det((700, 400, 80, 80), frame_index=1),
        ],
    )

    tiles, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame0,  # no motion in image
        frame_index=2,
        active_tracks=[t],
        perch_memory={},
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    # Bbox is at (700, 400). Velocity is 300, so v_margin is 1.5 * 300 = 450.
    # Expanded box extends to 700 + 80 + 450 = 1230, crossing into Tile 1 (x=819).
    assert all_tiles[0] in tiles
    assert all_tiles[1] in tiles


def test_perch_memory_preserves_motionless_bird_for_30_frames():
    """A bird validated with >= 3 detections stays queued in perch memory for 30 frames."""
    h_full, w_full = 2160, 3840
    all_tiles = _tile_offsets(w_full, h_full)
    frame0 = np.full((h_full, w_full, 3), 128, dtype=np.uint8)
    perch_memory: dict[tuple[int, int, int, int], int] = {}

    # Validated track with 3 detections on Tile 0
    t = Track(
        track_id=1,
        detections=[
            _det((300, 300, 80, 80), 0),
            _det((300, 300, 80, 80), 1),
            _det((300, 300, 80, 80), 2),
        ],
    )

    # Frame 2: Process frame with active track
    tiles, prev_gray = select_active_tiles_hardened(
        prev_gray=None,
        curr_bgr=frame0,
        frame_index=2,
        active_tracks=[t],
        perch_memory=perch_memory,
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert all_tiles[0] in perch_memory
    # Expiry frame is 2 + 30 = 32
    assert perch_memory[all_tiles[0]] == 32

    # Now the track closes (bird is motionless, YOLO dropped it)
    t.closed = True
    t.missed_frames = 4

    # Frame 10 (non-keyframe): Track is closed, no motion, but perch memory holds tile 0
    tiles_f10, prev_gray = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame0,
        frame_index=10,
        active_tracks=[],
        perch_memory=perch_memory,
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert tiles_f10 == [all_tiles[0]]

    # Frame 32: Still valid on frame 32
    tiles_f32, prev_gray = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame0,
        frame_index=32,
        active_tracks=[],
        perch_memory=perch_memory,
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert tiles_f32 == [all_tiles[0]]

    # Frame 34 (non-keyframe, 34 % 3 != 0): Perch memory has expired
    tiles_f34, _ = select_active_tiles_hardened(
        prev_gray=prev_gray,
        curr_bgr=frame0,
        frame_index=34,
        active_tracks=[],
        perch_memory=perch_memory,
        all_tiles=all_tiles,
        frame_shape=(h_full, w_full),
    )
    assert tiles_f34 == []
    assert all_tiles[0] not in perch_memory
