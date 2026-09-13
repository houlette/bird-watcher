"""Per-hour model of the fixed backdrop, and a test for "is this any of it".

The camera never moves, so nearly every pixel is the same yard it was
yesterday. A per-pixel median over many frames converges on the empty scene:
birds, squirrels and blowing leaves all average out, and what is left is the
wall, the downspout, the ledge and the pots.

That is the one instrument that reaches the junk the other defences cannot.
The scene mask needs an artifact to sit in one 100px cell and collect NAB
labels. The recurrence detector needs it at the same box on three separate
days. A stretch of concrete step, a leaf, or a length of downspout satisfies
neither, yet each is provably nothing but backdrop.

Two things had to be right before it worked at all:

**Bin by hour.** A median across the whole day is useless here, because
shadows crossing a stone wall differ from it as much as a bird does. Scored
against a single-hour model instead, junk and birds separate.

**Measure locally.** An absolute difference threshold trips on any lighting
change. Scoring a box against the ring around it cancels that: whatever the
light does, it does to both.

Measured on 130 frames from the 21:00 UTC hour, NAB detections sit at a
median ratio of 1.46 against 2.01 for the ones the filter calls birds, and
a cut at 1.5 catches 51% of the NAB rows while touching 7.9% of the others —
and that second figure is an overestimate, since roughly two thirds of what
the filter calls a bird in this yard is junk too.

What it will not catch is the lens flare. A flare genuinely differs from the
backdrop; it is a light, not a fixture.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BACKDROP_DIR = DATA_DIR / "backdrop"
FRAMES_DIR = DATA_DIR / "frames"

# Model resolution. 1/4 of 4K is 960x540, fine enough to resolve a small
# bird and cheap enough that a rebuild over hundreds of frames is seconds.
SCALE = 4

# Width of the comparison ring around a detection box, in model pixels.
RING_PX = 40

# Below this, a box differs from the backdrop no more than its own
# surroundings do, so it is backdrop. Set well under the measured 1.46
# median for NAB rows: this is deliberately the conservative end of the
# curve, catching a quarter of the junk rather than half, because the
# alternative costs birds in a yard that does not have many.
MIN_CONTRAST_RATIO = 1.20

# Frames needed in an hour bin before its model is trusted. Below this the
# median has not converged and a bird can survive into the backdrop.
MIN_FRAMES_PER_HOUR = 25

# How far back a rebuild looks for source frames.
REBUILD_WINDOW_DAYS = 21

_lock = threading.Lock()
_cache: dict[str, np.ndarray] = {}
_cache_stamp: dict[str, float] = {}


def _path_for(hour: int) -> Path:
    return BACKDROP_DIR / f"hour_{hour:02d}.png"


def _downscale_gray(img) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small


def rebuild(frame_paths_by_hour: dict[int, list[Path]]) -> dict[int, int]:
    """Write one median backdrop per hour. Returns frames used per hour."""
    BACKDROP_DIR.mkdir(parents=True, exist_ok=True)
    used: dict[int, int] = {}
    for hour, paths in sorted(frame_paths_by_hour.items()):
        if len(paths) < MIN_FRAMES_PER_HOUR:
            continue
        grays = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                grays.append(_downscale_gray(img))
        if len(grays) < MIN_FRAMES_PER_HOUR:
            continue
        shapes = {g.shape for g in grays}
        if len(shapes) > 1:
            # A camera resolution change invalidates the whole model; better
            # to skip the hour than to median two different framings.
            log.warning("Backdrop hour %02d: mixed frame shapes %s; skipping", hour, shapes)
            continue
        median = np.median(np.stack(grays), axis=0).astype(np.uint8)
        cv2.imwrite(str(_path_for(hour)), median)
        used[hour] = len(grays)
    with _lock:
        _cache.clear()
        _cache_stamp.clear()
    log.info("Backdrop rebuilt for %d hour(s): %s", len(used), used)
    return used


def get_backdrop(hour: int) -> np.ndarray | None:
    """Load an hour's backdrop, cached until the file on disk changes."""
    key = f"{hour:02d}"
    path = _path_for(hour)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    with _lock:
        if _cache.get(key) is not None and _cache_stamp.get(key) == stamp:
            return _cache[key]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _cache[key] = img
        _cache_stamp[key] = stamp
        return img


def contrast_ratio(frame_gray_small, backdrop, box) -> float | None:
    """How much more a box differs from the backdrop than its ring does.

    Around 1 means the box moved exactly as much as its surroundings, which
    is what backdrop does under changing light. A real object sits above it.
    """
    if backdrop is None or frame_gray_small is None:
        return None
    if frame_gray_small.shape != backdrop.shape:
        return None
    x, y, w, h = [int(v) // SCALE for v in box[:4]]
    H, W = backdrop.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    diff = cv2.absdiff(frame_gray_small, backdrop).astype(np.float32)
    inner = diff[y0:y1, x0:x1]
    rx0, ry0 = max(0, x0 - RING_PX), max(0, y0 - RING_PX)
    rx1, ry1 = min(W, x1 + RING_PX), min(H, y1 + RING_PX)
    ring = diff[ry0:ry1, rx0:rx1].copy()
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = np.nan
    with np.errstate(invalid="ignore"):
        ring_mean = float(np.nanmean(ring))
    if not np.isfinite(ring_mean) or ring_mean < 1.0:
        ring_mean = 1.0
    return float(inner.mean() / ring_mean)


def prepare(frame_bgr, captured_at: datetime):
    """Per-frame setup shared by every detection in it: the hour's backdrop
    and the downscaled grayscale frame. Returns (backdrop, gray) or None."""
    backdrop = get_backdrop(captured_at.hour)
    if backdrop is None:
        return None
    gray = _downscale_gray(frame_bgr)
    if gray.shape != backdrop.shape:
        return None
    return backdrop, gray


def filter_detections(detections, frame_bgr, captured_at: datetime,
                      min_ratio: float = MIN_CONTRAST_RATIO):
    """Drop detections whose box is indistinguishable from the backdrop.

    Returns `(kept, suppressed_count, scored_count)`. When no backdrop
    exists for the hour, everything is kept and scored_count is 0, so the
    metric never confuses "nothing suppressed" with "not running".
    """
    ready = prepare(frame_bgr, captured_at)
    if ready is None:
        return list(detections), 0, 0
    backdrop, gray = ready
    kept, suppressed, scored = [], 0, 0
    for d in detections:
        r = contrast_ratio(gray, backdrop, d.bbox)
        if r is None:
            kept.append(d)
            continue
        scored += 1
        if r < min_ratio:
            suppressed += 1
            continue
        kept.append(d)
    if suppressed:
        log.info("Backdrop filter suppressed %d of %d scored detection(s)", suppressed, scored)
    return kept, suppressed, scored


def cut_out(crop_bgr, box, captured_at: datetime):
    """Return the crop with backdrop pixels blacked out, or None.

    Not wired into classification yet: changing what the classifier sees
    needs its own before-and-after on the eval set, not a guess.
    """
    backdrop = get_backdrop(captured_at.hour)
    if backdrop is None:
        return None
    x, y, w, h = [int(v) // SCALE for v in box[:4]]
    H, W = backdrop.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    back = cv2.resize(backdrop[y0:y1, x0:x1], (crop_bgr.shape[1], crop_bgr.shape[0]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.medianBlur(cv2.absdiff(gray, back), 3)
    mask = cv2.morphologyEx((diff > 26).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(crop_bgr, crop_bgr, mask=mask)


def frames_by_hour(since: datetime, frames_dir: Path = FRAMES_DIR) -> dict[int, list[Path]]:
    """Group recent preserved frames by the hour they were written.

    File mtime stands in for capture hour. The frames are written minutes
    after capture, which is inside the bin except for the handful that land
    either side of the hour, and one stray frame does not move a median.
    """
    out: dict[int, list[Path]] = defaultdict(list)
    if not frames_dir.exists():
        return out
    cutoff = since.timestamp()
    for p in frames_dir.glob("*.jpg"):
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_mtime < cutoff:
            continue
        out[datetime.utcfromtimestamp(st.st_mtime).hour].append(p)
    return out


def rebuild_from_recent(days: int = REBUILD_WINDOW_DAYS) -> dict[int, int]:
    return rebuild(frames_by_hour(datetime.utcnow() - timedelta(days=days)))
