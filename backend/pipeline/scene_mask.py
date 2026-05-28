"""Spatial filter that drops YOLO detections in known false-positive regions.

The user's labeled NABs tell us, geographically, where birds DON'T live in
this yard's frame — typically a hummingbird feeder, a wind-spinner, or a
permanently bird-shaped piece of background scenery that YOLO can't help
firing on. We aggregate those labels into a coarse grid and use the result
as a per-frame suppression mask, applied between detection and tracking.

Two important design choices:

1. **High-YOLO-confidence override.** If YOLO is unusually sure about a
   detection (≥ OVERRIDE_YOLO_CONFIDENCE), we let it through even inside
   a hot zone. The user explicitly wants this — a real hummingbird ON the
   hummingbird feeder should still appear in the feed. Static FPs are
   usually low/medium confidence (0.20–0.50); a hummingbird actually at
   the feeder reads much higher.

2. **Rolling 14-day window.** Old labels age out so moving the feeder
   only blinds the system for a couple of weeks rather than forever.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timedelta

from db.models import NOT_A_BIRD_LABEL, Correction, Detection, Species
from db.session import SessionLocal

log = logging.getLogger(__name__)

# Cells are GRID_PX × GRID_PX in the 4K source frame. 100 px = ~2.6 % of
# frame width / ~4.6 % of height per cell — coarse enough to be robust to
# YOLO's bbox jitter, fine enough that one feeder doesn't mask off a whole
# quadrant of the yard.
GRID_PX = 100

# A cell needs at least this many NAB labels to be treated as "hot." Lower
# = more aggressive masking, more risk of false suppression. Tuned for the
# current scene where the top cell has 259 labels and the long tail trails
# off below 10.
MIN_NABS_PER_CELL = 10

# Older NABs are dropped from the mask computation so that moving objects
# (the feeder, an ornament) cause the mask to update within two weeks
# rather than being remembered forever.
LOOKBACK_DAYS = 14

# YOLO confidence above which we IGNORE the scene mask. A real bird in a
# masked region (esp. a hummingbird AT the hummingbird feeder, which is
# exactly the masked region) shouldn't be dropped just because the
# location usually contains a static FP. Tuned just above typical FP-
# detection confidence (~0.20-0.50) but below confident-classifier-bird.
OVERRIDE_YOLO_CONFIDENCE = 0.65

# Hourly refresh of the cached hot-zone set. Labels accumulate slowly; this
# is overkill but the query is cheap.
REFRESH_INTERVAL_SECONDS = 60 * 60

_cache_lock = threading.Lock()
_cached_zones: set[tuple[int, int]] | None = None
_cached_at: datetime | None = None


def _bbox_to_cell(bbox) -> tuple[int, int]:
    """Map a YOLO bbox to the grid cell containing its center."""
    cx = (bbox[0] + bbox[2] // 2) // GRID_PX
    cy = (bbox[1] + bbox[3] // 2) // GRID_PX
    return (cx, cy)


def _compute_hot_zones() -> set[tuple[int, int]]:
    """Query the DB for recent NAB-labeled detections and return the cells
    where they cluster densely enough to suppress."""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
        rows = (
            db.query(Detection.bbox)
            .join(Correction, Correction.detection_id == Detection.id)
            .join(Species, Correction.correct_species_id == Species.id)
            .filter(Species.common_name == NOT_A_BIRD_LABEL)
            .filter(Detection.created_at >= cutoff)
            .all()
        )
    finally:
        db.close()

    counter: Counter[tuple[int, int]] = Counter()
    for (bbox,) in rows:
        if not bbox or len(bbox) < 4:
            continue
        counter[_bbox_to_cell(bbox)] += 1

    hot = {cell for cell, n in counter.items() if n >= MIN_NABS_PER_CELL}
    log.info(
        "Scene mask: %d hot cell(s) from %d recent NAB label(s) (threshold=%d, lookback=%dd)",
        len(hot), sum(counter.values()), MIN_NABS_PER_CELL, LOOKBACK_DAYS,
    )
    return hot


def get_hot_zones(force_refresh: bool = False) -> set[tuple[int, int]]:
    """Return the cached set of hot cells, refreshing on TTL or on demand."""
    global _cached_zones, _cached_at
    with _cache_lock:
        now = datetime.utcnow()
        stale = (
            _cached_zones is None
            or _cached_at is None
            or (now - _cached_at).total_seconds() > REFRESH_INTERVAL_SECONDS
        )
        if force_refresh or stale:
            try:
                _cached_zones = _compute_hot_zones()
                _cached_at = now
            except Exception:
                log.exception("Scene mask refresh failed; reusing prior cache")
                if _cached_zones is None:
                    _cached_zones = set()  # safe empty default
        return _cached_zones


def is_masked(bbox, hot_zones: set[tuple[int, int]]) -> bool:
    """True if the bbox center sits inside any hot cell."""
    if not hot_zones:
        return False
    return _bbox_to_cell(bbox) in hot_zones


def filter_detections(detections, hot_zones: set[tuple[int, int]] | None = None):
    """Drop detections inside hot zones unless their YOLO confidence
    overrides the mask. Returns the kept list (mutation-free).

    Pass an explicit `hot_zones` set in tests to make the function fully
    deterministic; production calls let it consult the cached set.
    """
    if hot_zones is None:
        hot_zones = get_hot_zones()
    if not hot_zones:
        return detections
    kept = []
    suppressed = 0
    for d in detections:
        # High-confidence override: a confident bird gets through even at a
        # location historically labeled NAB.
        if d.confidence >= OVERRIDE_YOLO_CONFIDENCE:
            kept.append(d)
            continue
        if is_masked(d.bbox, hot_zones):
            suppressed += 1
            continue
        kept.append(d)
    if suppressed:
        log.debug("Scene mask suppressed %d detection(s)", suppressed)
    return kept
