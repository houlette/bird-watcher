"""Find detections that keep reappearing at the same place, and call them scenery.

A visitor arrives, moves, and leaves. A lens flare, a wind spinner or a
bird-shaped rock is in the same pixels tomorrow, and YOLO fires on it every
time. That difference needs no labels to exploit, which is the point: the
scene mask depends on someone labeling, and stopped working the fortnight
that stopped.

Matching is on the box, not the crop. Crop pixels look like a promising
signal and are not: the YOLO box jitters by tens of pixels between
detections, so the same object is framed differently every time and a
perceptual hash moves with it. Hashing 1,280 real crops grouped almost
nothing across visits even at a loose threshold, while box overlap
separated the fixtures cleanly.

Thresholds were fitted against the May 2026 archive, the last stretch with
heavy hand-labeling: 7,516 detections, 2,405 corrected, 422 of them
confirmed as real birds.

    overlap  days  min hits   flagged   confirmed NAB   confirmed birds lost
    0.9      3     20         260       235             0
    0.9      3     1          321       254             2
    0.8      3     1          1613      622             22

All three conditions matter. Dropping to 0.8 overlap is what eats birds,
because a perch tolerates that much variation and a fixture does not, and
the minimum cluster size is what rescued the House Sparrow and the Rock
Pigeon that sat in small clusters of their own. Real birds do reuse a
perch. They do not reuse it twenty times at the same pixel across three
separate days.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from db.models import Correction, Detection, Species, Visit
from db.session import SessionLocal

log = logging.getLogger(__name__)

# See the table above before touching any of these three.
IOU_THRESHOLD = 0.90
MIN_DISTINCT_DAYS = 3
MIN_CLUSTER_SIZE = 20

# How far back a nightly run looks. Matches the scene mask's lookback so the
# two defences forget a moved feeder on the same schedule.
WINDOW_DAYS = 14

# Same escape hatch the scene mask gives a confident bird. On the May
# archive it protected nothing — all 14 high-confidence rows it spared were
# ones the user had already confirmed as NAB — so it costs about 5% of the
# recall for insurance against the case that data did not contain.
OVERRIDE_YOLO_CONFIDENCE = 0.65

# Boxes are bucketed by centre before comparison so this stays near-linear
# on an archive of tens of thousands of detections instead of comparing
# every pair. Anything overlapping at 0.9 shares a centre within a few
# pixels, so a coarse bucket with a one-cell halo cannot miss a match.
BUCKET_PX = 200

CORRECTION_SOURCE = "recurrence"

_cache_lock = threading.Lock()
_cached_boxes: list[tuple[int, int, int, int]] | None = None
_cached_at: datetime | None = None
REFRESH_INTERVAL_SECONDS = 60 * 60


@dataclass
class Fixture:
    """One recurring box and the detections that landed on it."""

    box: tuple[int, int, int, int]
    detection_ids: list[int] = field(default_factory=list)
    days: set = field(default_factory=set)

    @property
    def n(self) -> int:
        return len(self.detection_ids)


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _bucket(box) -> tuple[int, int]:
    x, y, w, h = box
    return ((x + w // 2) // BUCKET_PX, (y + h // 2) // BUCKET_PX)


def cluster(rows) -> list[Fixture]:
    """Group (detection_id, bbox, day) rows into recurring-box clusters.

    Greedy single-pass assignment against each cluster's seed box, which is
    what the threshold was fitted on. Bucketing only limits which clusters a
    row is compared against; it does not change the outcome.
    """
    by_bucket: dict[tuple[int, int], list[Fixture]] = defaultdict(list)
    out: list[Fixture] = []
    for det_id, box, day in rows:
        bx, by = _bucket(box)
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for f in by_bucket.get((bx + dx, by + dy), ()):
                    if iou(f.box, box) >= IOU_THRESHOLD:
                        hit = f
                        break
                if hit:
                    break
            if hit:
                break
        if hit is None:
            hit = Fixture(box=tuple(box))
            by_bucket[(bx, by)].append(hit)
            out.append(hit)
        hit.detection_ids.append(det_id)
        hit.days.add(day)
    return out


def qualifying(clusters) -> list[Fixture]:
    return [
        f for f in clusters
        if f.n >= MIN_CLUSTER_SIZE and len(f.days) >= MIN_DISTINCT_DAYS
    ]


def _rows_for_window(db, since: datetime, until: datetime | None):
    q = (
        db.query(Detection.id, Detection.bbox, Visit.started_at)
        .join(Visit, Detection.visit_id == Visit.id)
        .filter(Visit.started_at >= since)
    )
    if until is not None:
        q = q.filter(Visit.started_at < until)
    rows = []
    for det_id, bbox, started in q.order_by(Visit.started_at).all():
        if not bbox or len(bbox) < 4 or started is None:
            continue
        rows.append((det_id, tuple(bbox[:4]), started.date()))
    return rows


def find_fixtures(db, since: datetime, until: datetime | None = None) -> list[Fixture]:
    """Recurring boxes in a time window, strongest first."""
    fixtures = qualifying(cluster(_rows_for_window(db, since, until)))
    fixtures.sort(key=lambda f: -f.n)
    return fixtures


def get_fixture_boxes(force_refresh: bool = False) -> list[tuple[int, int, int, int]]:
    """Cached fixture boxes for live suppression, refreshed on TTL."""
    global _cached_boxes, _cached_at
    with _cache_lock:
        now = datetime.utcnow()
        stale = (
            _cached_boxes is None
            or _cached_at is None
            or (now - _cached_at).total_seconds() > REFRESH_INTERVAL_SECONDS
        )
        if force_refresh or stale:
            db = SessionLocal()
            try:
                since = now - timedelta(days=WINDOW_DAYS)
                _cached_boxes = [f.box for f in find_fixtures(db, since)]
                _cached_at = now
                log.info("Recurrence: %d fixture box(es) over the last %dd",
                         len(_cached_boxes), WINDOW_DAYS)
            except Exception:
                log.exception("Recurrence refresh failed; reusing prior cache")
                if _cached_boxes is None:
                    _cached_boxes = []
            finally:
                db.close()
        return _cached_boxes


def filter_detections(detections, boxes=None):
    """Drop detections sitting on a known fixture box.

    Mirrors scene_mask.filter_detections, including the confidence
    override, and returns `(kept, suppressed_count)`.
    """
    if boxes is None:
        boxes = get_fixture_boxes()
    if not boxes:
        return list(detections), 0
    kept, suppressed = [], 0
    for d in detections:
        if d.confidence >= OVERRIDE_YOLO_CONFIDENCE:
            kept.append(d)
            continue
        if any(iou(d.bbox, b) >= IOU_THRESHOLD for b in boxes):
            suppressed += 1
            continue
        kept.append(d)
    if suppressed:
        log.info("Recurrence filter suppressed %d detection(s)", suppressed)
    return kept, suppressed


def relabel_fixtures(db, since: datetime, until: datetime | None = None,
                     apply: bool = False) -> dict:
    """Mark detections on recurring boxes as Not a bird.

    Never touches a detection the user has already corrected, one already
    carrying a sentinel label, or one clearing the confidence override.
    Writes a Correction with source `recurrence` so every change is
    attributable and can be undone.
    """
    from db.models import NOT_A_BIRD_LABEL, SENTINEL_LABELS

    fixtures = find_fixtures(db, since, until)
    nab = db.query(Species).filter_by(common_name=NOT_A_BIRD_LABEL).one_or_none()
    if nab is None:
        return {"fixtures": len(fixtures), "relabeled": 0, "error": "no NAB species row"}

    sentinel_ids = {
        r[0] for r in db.query(Species.id)
        .filter(Species.common_name.in_(SENTINEL_LABELS)).all()
    }
    corrected = {r[0] for r in db.query(Correction.detection_id).distinct().all()}

    ids = [i for f in fixtures for i in f.detection_ids]
    box_of = {i: f.box for f in fixtures for i in f.detection_ids}
    stats = defaultdict(int)
    todo = []
    for det in db.query(Detection).filter(Detection.id.in_(ids)).all() if ids else []:
        if det.id in corrected:
            stats["already corrected"] += 1
        elif det.species_id in sentinel_ids:
            stats["already a sentinel"] += 1
        elif det.yolo_confidence is not None and det.yolo_confidence >= OVERRIDE_YOLO_CONFIDENCE:
            stats["clears the confidence override"] += 1
        else:
            todo.append(det)

    if apply:
        for det in todo:
            db.add(Correction(
                detection_id=det.id,
                correct_species_id=nab.id,
                source=CORRECTION_SOURCE,
                rationale=(
                    f"Recurring box {box_of[det.id]} seen on "
                    f"{len(next(f for f in fixtures if det.id in f.detection_ids).days)} "
                    f"separate days; scenery, not a visitor."
                ),
            ))
            det.species_id = nab.id
        db.commit()

    return {
        "fixtures": len(fixtures),
        "candidates": len(ids),
        "relabeled": len(todo) if apply else 0,
        "would_relabel": len(todo),
        "skipped": dict(stats),
    }
