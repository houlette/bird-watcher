"""Spatial filter that drops YOLO detections in known false-positive regions.

NAB labels tell us, geographically, where birds DON'T live in this yard's
frame — typically a hummingbird feeder, a wind-spinner, a lens flare that
arrives with a particular sun angle, or a permanently bird-shaped piece of
background scenery that YOLO can't help firing on. We aggregate those labels
into a coarse grid and use the result as a per-frame suppression mask,
applied between detection and tracking.

Labels come from two places: the user's corrections, and the binary
filter's own NAB verdicts. The second source was added because the first
one dries up. Nothing forces the user to label anything on a given
fortnight, and when they stop, a mask built only from their corrections
empties out and every fixed artifact starts reaching the feed again.

Three important design choices:

1. **High-YOLO-confidence override.** If YOLO is unusually sure about a
   detection (≥ OVERRIDE_YOLO_CONFIDENCE), we let it through even inside
   a hot zone. The user explicitly wants this — a real hummingbird ON the
   hummingbird feeder should still appear in the feed. Static FPs are
   usually low/medium confidence (0.20–0.50); a hummingbird actually at
   the feeder reads much higher.

2. **Rolling 14-day window.** Old labels age out so moving the feeder
   only blinds the system for a couple of weeks rather than forever.

3. **Machine verdicts have to dominate a cell.** A user correction qualifies
   a cell on count alone. A pipeline verdict needs a higher count, and the
   cell has to be almost entirely NAB, so a real perch where blurred birds
   collect the odd NAB verdict never goes dark.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timedelta

from db.models import NOT_A_BIRD_LABEL, SENTINEL_LABELS, Correction, Detection, Species
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

# The pipeline's own NAB verdicts count too, at a higher bar.
#
# Hand labels used to be the only input here, which meant the mask decayed to
# nothing whenever labeling paused for longer than LOOKBACK_DAYS: on
# 2026-09-11 it had been reporting "0 hot cell(s) from 0 recent NAB label(s)"
# hourly for weeks, so every fixed artifact it used to suppress by position
# was reaching the feed. Meanwhile the binary filter was producing hundreds
# of NAB verdicts a day that nothing read. These thresholds let those count
# without letting a machine mistake blind a real perch:
#
#   - MIN_MACHINE: 20 rather than 10, because one verdict is weaker evidence
#     than one of the user's.
#   - PURITY: the cell has to be almost entirely NAB. A genuine perch is not.
#     The feeder cell (14, 18) ran 30 NAB against 23 birds, mostly sparrows,
#     and is correctly left alone by this.
#   - MAX_BIRD_LABELS: an absolute cap as well as a ratio, so a busy cell with
#     a long NAB tail can never qualify on ratio alone.
#
# Checked against the 14 days to 2026-09-11: 10 cells qualify, suppressing 465
# NAB detections and touching 8 bird-labeled rows, which is 1% of the window's
# bird labels, and those 8 are singletons in cells running 100-to-2 against
# them. The YOLO confidence override below is still the escape hatch for a
# real bird that lands in one.
MIN_MACHINE_NABS_PER_CELL = 20
MACHINE_NAB_PURITY = 0.90
MAX_BIRD_LABELS_IN_HOT_CELL = 5

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
    """Return the cells where NAB detections cluster densely enough to suppress.

    Two independent paths into the hot set. The user's own NAB corrections
    qualify a cell on count alone, since a hand label is authoritative. The
    pipeline's NAB verdicts need a higher count and have to dominate the
    cell, which is what keeps a real perch out of the set even though blurred
    birds there collect NAB verdicts of their own.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
        human_rows = (
            db.query(Detection.bbox)
            .join(Correction, Correction.detection_id == Detection.id)
            .join(Species, Correction.correct_species_id == Species.id)
            .filter(Species.common_name == NOT_A_BIRD_LABEL)
            .filter(Detection.created_at >= cutoff)
            .all()
        )
        # Every labeled detection in the window, NAB or not, so the machine
        # path can measure how much of each cell is NAB rather than just how
        # many. Sentinels other than NAB (Poor Quality, Unknown bird) and
        # unlabeled rows count as neither, so they neither qualify a cell nor
        # protect one.
        labeled_rows = (
            db.query(Detection.bbox, Species.common_name)
            .join(Species, Detection.species_id == Species.id)
            .filter(Detection.created_at >= cutoff)
            .all()
        )
    finally:
        db.close()

    human: Counter[tuple[int, int]] = Counter()
    for (bbox,) in human_rows:
        if bbox and len(bbox) >= 4:
            human[_bbox_to_cell(bbox)] += 1

    nab: Counter[tuple[int, int]] = Counter()
    bird: Counter[tuple[int, int]] = Counter()
    for bbox, name in labeled_rows:
        if not bbox or len(bbox) < 4:
            continue
        cell = _bbox_to_cell(bbox)
        if name == NOT_A_BIRD_LABEL:
            nab[cell] += 1
        elif name not in SENTINEL_LABELS:
            bird[cell] += 1

    hot_human = {cell for cell, n in human.items() if n >= MIN_NABS_PER_CELL}
    hot_machine = {
        cell
        for cell, n in nab.items()
        if n >= MIN_MACHINE_NABS_PER_CELL
        and bird[cell] <= MAX_BIRD_LABELS_IN_HOT_CELL
        and n / (n + bird[cell]) >= MACHINE_NAB_PURITY
    }
    hot = hot_human | hot_machine
    log.info(
        "Scene mask: %d hot cell(s) — %d from %d user NAB label(s) (threshold=%d), "
        "%d from %d pipeline NAB verdict(s) (threshold=%d, purity=%.2f); lookback=%dd",
        len(hot), len(hot_human), sum(human.values()), MIN_NABS_PER_CELL,
        len(hot_machine), sum(nab.values()), MIN_MACHINE_NABS_PER_CELL,
        MACHINE_NAB_PURITY, LOOKBACK_DAYS,
    )
    if not hot:
        # Worth saying out loud: an empty mask is the state this module spent
        # weeks in without anyone noticing.
        log.warning("Scene mask is EMPTY — no cell qualified; nothing is being suppressed by position")
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
    overrides the mask. Returns `(kept, suppressed_count)` — the count
    is bubbled up to the worker so it can persist on the Visit row and
    aggregate into the daily stats funnel.

    Pass an explicit `hot_zones` set in tests to make the function fully
    deterministic; production calls let it consult the cached set.
    """
    if hot_zones is None:
        hot_zones = get_hot_zones()
    if not hot_zones:
        return list(detections), 0
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
        log.info("Scene mask suppressed %d detection(s)", suppressed)
    return kept, suppressed
