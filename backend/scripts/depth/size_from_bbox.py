"""Pure helpers for sampling depth at a detection bbox and computing a
scale-free real-world size proxy.

The validation phase uses these helpers offline; they're written to be
importable by a future production `pipeline/depth.py` loader without
modification — no DB access, no logging, no side effects beyond what
the caller passes in.
"""
from __future__ import annotations

import numpy as np


# Bbox foot sample: a (FOOT_PATCH × FOOT_PATCH) median around the
# bottom-center of the bbox. The bottom-center is where a perched bird's
# feet meet the perch — exactly the depth we want. Median over a 5×5
# patch denoises the inevitable depth-map jitter at edges.
FOOT_PATCH = 5

# Local-background sample: the median of a NEIGHBORHOOD × NEIGHBORHOOD
# patch centered on the bbox foot, used to detect "in-flight" birds
# whose foot pixel is significantly closer than its surroundings.
NEIGHBORHOOD = 100

# Flight threshold: if the foot's depth is more than this much CLOSER
# than the local background median, the bird is in front of its
# background → in flight → skip size estimation.
#   0.30 means "foot is at least 30 % closer than local median."
# Tuned conservatively: better to skip a perched bird than to ship a
# wrong size for a flier.
FLIGHT_RELATIVE_CLOSER = 0.30


def bbox_foot(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Bottom-center of the bbox in pixel coords. Bbox is (x, y, w, h)."""
    x, y, w, h = bbox
    return int(round(x + w / 2.0)), int(round(y + h))


def sample_patch(depth: np.ndarray, cx: int, cy: int, patch: int) -> float | None:
    """Return the median depth in a `patch×patch` square centered on
    (cx, cy). Out-of-bounds points (the bbox falls partly off-frame)
    return None."""
    H, W = depth.shape
    half = patch // 2
    x0, x1 = max(0, cx - half), min(W, cx + half + 1)
    y0, y1 = max(0, cy - half), min(H, cy + half + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch_vals = depth[y0:y1, x0:x1]
    finite = patch_vals[np.isfinite(patch_vals)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def is_in_flight(
    depth: np.ndarray,
    cx: int,
    cy: int,
    *,
    foot_depth: float,
    relative_closer: float = FLIGHT_RELATIVE_CLOSER,
) -> bool:
    """Heuristic for "this detection is a bird in flight" → its bbox
    foot is significantly closer than the local background."""
    neighborhood_depth = sample_patch(depth, cx, cy, NEIGHBORHOOD)
    if neighborhood_depth is None or neighborhood_depth <= 0:
        return False
    return foot_depth < neighborhood_depth * (1.0 - relative_closer)


def bbox_diagonal_px(bbox: tuple[float, float, float, float]) -> float:
    """Diagonal of the bbox in pixels. Diagonal — not max(w, h) — because
    body-length birds can be horizontal (perched), vertical (hovering),
    or in between; diagonal is the most rotation-robust 1-D scalar."""
    _, _, w, h = bbox
    return float(np.sqrt(w * w + h * h))


def size_proxy(
    bbox: tuple[float, float, float, float],
    depth: np.ndarray,
) -> tuple[float | None, str]:
    """Compute the scale-free size proxy for a detection.

    Returns `(proxy, status)`:
      - `proxy` is `bbox_diagonal_px × foot_depth_m`. Proportional to
        true body length up to a constant (focal length / unit
        conversion). None when we can't get a clean estimate.
      - `status` is one of:
          'ok'         — clean perched-bird estimate
          'in_flight'  — foot depth notably closer than surroundings
          'off_frame'  — bbox foot is off-image
          'invalid'    — depth NaN / negative at the sample point

    The status field lets the caller report rejection reasons in
    aggregate without re-running the logic.
    """
    cx, cy = bbox_foot(bbox)
    H, W = depth.shape
    if not (0 <= cx < W and 0 <= cy < H):
        return None, "off_frame"

    foot_depth = sample_patch(depth, cx, cy, FOOT_PATCH)
    if foot_depth is None or not np.isfinite(foot_depth) or foot_depth <= 0:
        return None, "invalid"

    if is_in_flight(depth, cx, cy, foot_depth=foot_depth):
        return None, "in_flight"

    proxy = bbox_diagonal_px(bbox) * foot_depth
    return proxy, "ok"
