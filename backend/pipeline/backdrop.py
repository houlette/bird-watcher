"""Per-hour model of the fixed backdrop, and a test for "is this any of it".

OFF BY DEFAULT since 2026-09-13, because the premise does not hold in this
yard. See `settings.backdrop_filter_enabled` to put it back, and read the
measurement below first.

The idea was that the camera never moves, so a per-pixel median over many
frames converges on the empty scene: birds, squirrels and blowing leaves
average out, and what is left is the wall, the downspout, the ledge and the
pots. That would have reached junk the other defences cannot, since the
scene mask needs an artifact to sit in one 100px cell and collect NAB
labels, and the recurrence detector needs it at the same box on three
separate days, while a stretch of concrete step or a length of downspout
satisfies neither.

**What it actually costs.** Scored against a June-2026 model built with
capture-time binning, over 630 user-confirmed birds and 930 user-confirmed
junk, the ratio barely separates the two: birds sit at a median of 1.382 and
junk at 1.353. At the 1.20 cut that ran in production it removed 34.0% of
the junk and 37.5% of the birds, and 30.4% of the 23 independently
audio-confirmed birds. Bird loss tracks junk catch at every threshold from
0.90 to 1.50, so it was slightly worse than a coin flip, and suppression
happens before persistence so everything it took is gone.

**Why, as far as it is understood.** The yard is not a fixed backdrop at a
21-day timescale. Per-pixel median absolute deviation runs 6 to 25 grey
levels by hour, and even the calmest 5% of pixels move 3.5 to 14, so the
median is a blend of three weeks of different yards rather than a picture of
an empty one. The typical pixel sits 24 grey levels off the model, and a
gain-plus-bias fit over every pixel recovers only 2.3 of those, so it is not
an exposure problem that normalisation could fix. Scoring a box against the
ring around it was meant to cancel lighting, and it does, but what remains
on both sides is the scene changing rather than an object arriving.

Two details are still right and worth keeping if this is ever revived. Bin
by hour, on capture time and not on file mtime; see `frames_by_hour` for
what mtime binning cost. And measure locally, since an absolute difference
threshold trips on any lighting change.

The model itself is still rebuilt nightly, because `cut_out()` wants it and
because a variance-aware successor would start from the same stack.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from db.models import Visit
from db.session import SessionLocal
from settings import settings

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
    # Drop models for hours this rebuild could not build. Without this a
    # stale file is indistinguishable from a current one and keeps being
    # scored against: the mtime-binned models covered all 24 hours, night
    # included, out of frames that were all shot in daylight. Skipped when
    # nothing built at all, so an unmounted frames volume cannot wipe the
    # set.
    pruned = []
    if used:
        for hour in range(24):
            if hour not in used and _path_for(hour).exists():
                _path_for(hour).unlink()
                pruned.append(hour)
    with _lock:
        _cache.clear()
        _cache_stamp.clear()
    log.info("Backdrop rebuilt for %d hour(s): %s; pruned %s", len(used), used, pruned)
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
    if not settings.backdrop_filter_enabled:
        return list(detections), 0, 0
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


# Frames are written as v{visit_id:08d}_t{track_id:04d}.jpg by
# pipeline/process.py, so the visit, and through it the capture time, is
# recoverable from the name.
_FRAME_NAME = re.compile(r"^v(\d+)_t\d+\.jpg$")

# SQLite caps the variables in one statement, so the id lookup goes in
# chunks well under that limit.
_ID_CHUNK = 500


def visit_id_from_frame(path: Path) -> int | None:
    m = _FRAME_NAME.match(path.name)
    return int(m.group(1)) if m else None


def bin_by_capture_hour(paths, captured_by_visit) -> tuple[dict[int, list[Path]], int]:
    """Sort frame paths into UTC hour bins by their visit's capture time.

    A frame whose visit is missing from the map is dropped rather than
    guessed at, because a frame in the wrong bin is the thing this exists
    to prevent. Returns the bins and how many were dropped.
    """
    out: dict[int, list[Path]] = defaultdict(list)
    dropped = 0
    for p in paths:
        vid = visit_id_from_frame(p)
        started = captured_by_visit.get(vid) if vid is not None else None
        if started is None:
            dropped += 1
            continue
        out[started.hour].append(p)
    return out, dropped


def frames_by_hour(db, since: datetime, frames_dir: Path = FRAMES_DIR) -> dict[int, list[Path]]:
    """Group recent preserved frames by the hour they were captured.

    Capture time comes from the frame's visit. It used to come from the
    file mtime, on the reasoning that frames are written minutes after
    capture. Measured over 2,766 frames from late August 2026, the write
    trails capture by a median of 131 minutes and by more than 17 hours at
    p95, which put 1,481 of them, 54%, in an hour other than the one they
    were shot in. Every hour's median was therefore a blend of several
    lightings including night, and the ring around a detection sat a median
    of 40 grey levels off the model instead of near zero.
    """
    if not frames_dir.exists():
        return {}
    paths = list(frames_dir.glob("*.jpg"))
    ids = sorted({v for v in (visit_id_from_frame(p) for p in paths) if v is not None})
    captured: dict[int, datetime] = {}
    for i in range(0, len(ids), _ID_CHUNK):
        rows = (
            db.query(Visit.id, Visit.started_at)
            .filter(Visit.id.in_(ids[i:i + _ID_CHUNK]), Visit.started_at >= since)
            .all()
        )
        captured.update({vid: started for vid, started in rows if started is not None})
    out, dropped = bin_by_capture_hour(paths, captured)
    if dropped:
        log.info("Backdrop: %d of %d frame(s) fell outside the window or had no visit",
                 dropped, len(paths))
    return out


def rebuild_from_recent(days: int = REBUILD_WINDOW_DAYS) -> dict[int, int]:
    db = SessionLocal()
    try:
        return rebuild(frames_by_hour(db, datetime.utcnow() - timedelta(days=days)))
    finally:
        db.close()
