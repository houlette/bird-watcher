"""Phase 0: Motion-Gated Spatial Tiling Engine.

Reduces per-visit YOLO inference from ~450 tiles to ~180 tiles by skipping
empty tiles between keyframes while rigorously defending against false negatives:
  1. Keyframe safety net: Every 3rd frame (1.0s at 3 fps), all 15 tiles are evaluated.
  2. Track persistence & velocity envelope: Active tracks keep their tiles queued.
  3. Recent perch memory: Any bird track with >=3 detections keeps its tile queued
     for 30 frames (10 seconds), ensuring YOLO evaluates the perch even if the bird
     sits motionless across extended drops.
  4. Signed illumination adjustment: Cancels global auto-exposure / cloud shifts
     via median delta shift before frame differencing.
  5. Seam-boundary proximity expansion: Any motion touching a tile overlap seam
     expands a Cartesian envelope to pull in the neighboring tile.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pipeline.track import Track

# Prevent OpenCV primitives from spawning internal thread pools that contend
# with OpenVINO inference threads.
cv2.setNumThreads(1)

log = logging.getLogger(__name__)

# Resolution for motion differencing: 960x540 (4:1 downscale from 4K).
# Averages 4x4 pixel blocks via cv2.INTER_AREA, preserving small-bird motion
# while naturally suppressing sensor noise.
LOWRES_WIDTH = 960
LOWRES_HEIGHT = 540

# Photometric difference threshold. Pixel delta >= 10 is considered motion.
DIFF_THRESHOLD = 10

# Minimum number of motion pixels in low-res frame to trigger a tile.
# 24 pixels at 960x540 corresponds to ~384 pixels on 4K, sensitive enough to
# catch chickadees and kinglets shifting pose between frames.
MIN_MOTION_AREA_DEFAULT = 24

# Tile overlap margin on full 4K frame (20% of 1024 px = 204.8 ≈ 205 px).
TILE_OVERLAP_MARGIN_FULL = 205

# Recent perch memory duration in frames (30 frames @ 3 fps = 10.0 seconds).
PERCH_MEMORY_FRAMES = 30


def _box_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True if xywh rectangles a and b overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def select_active_tiles_hardened(
    prev_gray: np.ndarray | None,
    curr_bgr: np.ndarray,
    frame_index: int,
    active_tracks: list[Track],
    perch_memory: dict[tuple[int, int, int, int], int],  # tile -> expiry_frame
    all_tiles: list[tuple[int, int, int, int]],
    frame_shape: tuple[int, int],  # (height, width)
    keyframe_interval: int = 3,
    min_motion_area: int = MIN_MOTION_AREA_DEFAULT,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Production motion-gated tile selector with seam clustering, signed illumination
    invariance, and recent perch memory.

    Returns:
        (selected_tiles, curr_gray_lowres)
    """
    h_full, w_full = frame_shape

    # 1. Downscale to 960x540 grayscale via INTER_AREA (4:1, averages 4x4 blocks)
    curr_gray_raw = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    curr_low = cv2.resize(curr_gray_raw, (LOWRES_WIDTH, LOWRES_HEIGHT), interpolation=cv2.INTER_AREA)

    # Clean up expired perch memory
    expired = [tile for tile, expiry in perch_memory.items() if frame_index > expiry]
    for e in expired:
        del perch_memory[e]

    # Keyframe refresh (every 1.0s / 3rd frame) or missing history: run all 15 tiles
    if frame_index % keyframe_interval == 0 or prev_gray is None:
        # If there are active validated tracks, refresh their perch memory on keyframes too
        for t in active_tracks:
            if not t.closed and hasattr(t, "detections") and len(t.detections) >= 3:
                bx, by, bw, bh = t.last_bbox
                tbox = (bx, by, bw, bh)
                for tile in all_tiles:
                    if _box_intersects(tile, tbox):
                        perch_memory[tile] = frame_index + PERCH_MEMORY_FRAMES
        return list(all_tiles), curr_low

    scale_x = curr_low.shape[1] / float(w_full)
    scale_y = curr_low.shape[0] / float(h_full)

    selected_tiles: set[tuple[int, int, int, int]] = set()

    # 2. TRACK PERSISTENCE & VELOCITY ENVELOPE (With Proper Clamped Geometry)
    for t in active_tracks:
        if not t.closed or (hasattr(t, "missed_frames") and t.missed_frames <= 3):
            if not getattr(t, "detections", None):
                continue
            bx, by, bw, bh = t.last_bbox
            v_margin = 300
            if len(t.detections) >= 2:
                d1, d2 = t.detections[-2], t.detections[-1]
                c1x = d1.bbox[0] + d1.bbox[2] / 2.0
                c1y = d1.bbox[1] + d1.bbox[3] / 2.0
                c2x = d2.bbox[0] + d2.bbox[2] / 2.0
                c2y = d2.bbox[1] + d2.bbox[3] / 2.0
                vx = abs(c2x - c1x)
                vy = abs(c2y - c1y)
                v_margin = int(min(800, max(300, 1.5 * max(vx, vy))))

            # Properly clamped Cartesian bounding box geometry
            x1 = max(0, bx - v_margin)
            y1 = max(0, by - v_margin)
            x2 = min(w_full, bx + bw + v_margin)
            y2 = min(h_full, by + bh + v_margin)
            expanded_box = (x1, y1, x2 - x1, y2 - y1)

            for tile in all_tiles:
                if _box_intersects(tile, expanded_box):
                    selected_tiles.add(tile)
                    if len(t.detections) >= 3:
                        perch_memory[tile] = frame_index + PERCH_MEMORY_FRAMES

    # 3. RECENT PERCH MEMORY (Protect motionless perched birds across extended YOLO drops)
    for tile in perch_memory:
        selected_tiles.add(tile)

    # 4. SIGNED GLOBAL ILLUMINATION ADJUSTMENT (Immunity to cloud transitions / AE shifts)
    signed_diff = curr_low.astype(np.int16) - prev_gray.astype(np.int16)
    global_signed_delta = int(np.median(signed_diff))

    if abs(global_signed_delta) >= 2:
        prev_adjusted = np.clip(prev_gray.astype(np.int16) + global_signed_delta, 0, 255).astype(np.uint8)
    else:
        prev_adjusted = prev_gray

    diff = cv2.absdiff(curr_low, prev_adjusted)
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 5. PRECISE TILE-ZONE & SEAM-PROXIMITY EVALUATION (Evaluates ALL tiles to avoid blind spots)
    seam_margin_low = int(TILE_OVERLAP_MARGIN_FULL * scale_x)
    for tile in all_tiles:
        x, y, w, h = tile
        tx, ty = int(x * scale_x), int(y * scale_y)
        tw, th = int(w * scale_x), int(h * scale_y)
        tile_diff = thresh[ty : min(curr_low.shape[0], ty + th), tx : min(curr_low.shape[1], tx + tw)]

        pts = cv2.findNonZero(tile_diff)
        if pts is not None and len(pts) >= min_motion_area:
            selected_tiles.add(tile)
            # Find bounding box of actual motion inside tile
            mbx, mby, mbw, mbh = cv2.boundingRect(pts)
            th_actual, tw_actual = tile_diff.shape[:2]

            # Check if motion touches the tile overlap seam boundary
            touches_left = mbx <= seam_margin_low
            touches_right = (mbx + mbw) >= (tw_actual - seam_margin_low)
            touches_top = mby <= seam_margin_low
            touches_bottom = (mby + mbh) >= (th_actual - seam_margin_low)

            if touches_left or touches_right or touches_top or touches_bottom:
                # Convert motion bbox to full-frame coords
                full_mbx = x + int(mbx / scale_x)
                full_mby = y + int(mby / scale_y)
                full_mbw = int(mbw / scale_x)
                full_mbh = int(mbh / scale_y)

                # Expand only the motion box by seam margin using correct Cartesian math
                sx1 = max(0, full_mbx - TILE_OVERLAP_MARGIN_FULL)
                sy1 = max(0, full_mby - TILE_OVERLAP_MARGIN_FULL)
                sx2 = min(w_full, full_mbx + full_mbw + TILE_OVERLAP_MARGIN_FULL)
                sy2 = min(h_full, full_mby + full_mbh + TILE_OVERLAP_MARGIN_FULL)
                motion_envelope = (sx1, sy1, sx2 - sx1, sy2 - sy1)

                for adj_tile in all_tiles:
                    if adj_tile not in selected_tiles and _box_intersects(adj_tile, motion_envelope):
                        selected_tiles.add(adj_tile)

    return sorted(selected_tiles, key=lambda t: (t[1], t[0])), curr_low
