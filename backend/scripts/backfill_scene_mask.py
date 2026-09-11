"""Apply the scene mask retroactively to detections it should have caught.

The mask suppresses detections before they are ever persisted, so while it
was dark (it builds its hot cells from a rolling window and that window ran
empty once labeling paused) the junk it would have dropped became rows
instead. Those rows are still in the feed. This relabels them.

Deliberately the opposite default from the other backfills in this
directory: this one reports and changes nothing unless you pass --apply,
because it rewrites species_id on rows you are looking at.

What it will not touch:
  - Detections you have already corrected. Your judgement wins, always.
  - Detections already labeled Not a bird or another sentinel.
  - Detections whose YOLO confidence clears OVERRIDE_YOLO_CONFIDENCE, which
    is the same escape hatch the live mask gives a confident bird sitting in
    a hot cell.
  - Detections with no recorded YOLO confidence, since the override cannot
    be evaluated for them and guessing in the destructive direction is the
    wrong default. The dry run counts these separately.

Every change writes a Correction row with source "scene-mask-backfill" and a
rationale naming the cell, so the pass is auditable and can be undone by
deleting corrections with that source and restoring species_id from
raw_predictions[0].

Usage:
    cd backend
    python scripts/backfill_scene_mask.py                  # dry run
    python scripts/backfill_scene_mask.py --apply
    python scripts/backfill_scene_mask.py --since 2026-08-20 --apply
    python scripts/backfill_scene_mask.py --include-unidentified
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func  # noqa: E402

from db.models import (  # noqa: E402
    NOT_A_BIRD_LABEL,
    SENTINEL_LABELS,
    Correction,
    Detection,
    Species,
)
from db.session import SessionLocal  # noqa: E402
from db.utils import utcnow  # noqa: E402
from pipeline import scene_mask  # noqa: E402
from pipeline.scene_mask import BACKFILL_SOURCE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_scene_mask")

COMMIT_BATCH = 200


def _default_since(db) -> datetime:
    """When the mask went dark: the last user NAB correction, plus the
    lookback window it takes to age out of the hot-cell calculation."""
    nab = db.query(Species).filter_by(common_name=NOT_A_BIRD_LABEL).one_or_none()
    if nab is None:
        return utcnow() - timedelta(days=30)
    last = (
        db.query(func.max(Correction.created_at))
        .filter(Correction.correct_species_id == nab.id)
        .filter((Correction.source.is_(None)) | (Correction.source != BACKFILL_SOURCE))
        .scalar()
    )
    if last is None:
        return utcnow() - timedelta(days=30)
    return last + timedelta(days=scene_mask.LOOKBACK_DAYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the changes; without it this only reports")
    ap.add_argument("--since", type=str, default=None,
                    help="YYYY-MM-DD; defaults to when the mask went dark")
    ap.add_argument("--include-unidentified", action="store_true",
                    help="also relabel rows the classifier never named")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    hot = scene_mask.get_hot_zones(force_refresh=True)
    if not hot:
        log.error("No hot cells, so there is nothing this pass could suppress. Stopping.")
        return

    db = SessionLocal()
    try:
        since = (
            datetime.strptime(args.since, "%Y-%m-%d")
            if args.since
            else _default_since(db)
        )
        log.info("Hot cells: %d. Window: detections created since %s.",
                 len(hot), since.strftime("%Y-%m-%d %H:%M"))

        nab = db.query(Species).filter_by(common_name=NOT_A_BIRD_LABEL).one_or_none()
        if nab is None:
            log.error("No '%s' species row; cannot relabel.", NOT_A_BIRD_LABEL)
            return

        corrected_ids = {
            row[0] for row in db.query(Correction.detection_id).distinct().all()
        }
        sentinel_ids = {
            row[0] for row in db.query(Species.id)
            .filter(Species.common_name.in_(SENTINEL_LABELS)).all()
        }

        q = db.query(Detection).filter(Detection.created_at >= since)
        rows = q.order_by(Detection.id).all()

        hits: list[Detection] = []
        by_cell: Counter = Counter()
        by_species: Counter = Counter()
        skipped = defaultdict(int)

        for d in rows:
            if d.id in corrected_ids:
                skipped["already corrected by you"] += 1
                continue
            if d.species_id in sentinel_ids:
                skipped["already a sentinel"] += 1
                continue
            if d.species_id is None and not args.include_unidentified:
                skipped["unidentified (use --include-unidentified)"] += 1
                continue
            if not d.bbox or len(d.bbox) < 4:
                skipped["no usable bbox"] += 1
                continue
            cell = scene_mask._bbox_to_cell(d.bbox)
            if cell not in hot:
                continue
            if d.yolo_confidence is None:
                skipped["in a hot cell but no YOLO confidence recorded"] += 1
                continue
            if d.yolo_confidence >= scene_mask.OVERRIDE_YOLO_CONFIDENCE:
                skipped["in a hot cell but clears the confidence override"] += 1
                continue
            hits.append(d)
            by_cell[cell] += 1
            sp = db.get(Species, d.species_id) if d.species_id else None
            by_species[sp.common_name if sp else "Unidentified"] += 1
            if args.limit and len(hits) >= args.limit:
                break

        log.info("=" * 66)
        log.info("%d detection(s) would be relabeled %s", len(hits), NOT_A_BIRD_LABEL)
        for cell, n in by_cell.most_common():
            log.info("    cell %-10s %4d", str(cell), n)
        log.info("  by current label:")
        for name, n in by_species.most_common(12):
            log.info("    %-28s %4d", name, n)
        log.info("  left alone:")
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1]):
            log.info("    %-46s %4d", reason, n)
        log.info("=" * 66)

        if not args.apply:
            log.info("Dry run. Re-run with --apply to write these changes.")
            return

        for i, d in enumerate(hits, 1):
            cell = scene_mask._bbox_to_cell(d.bbox)
            db.add(Correction(
                detection_id=d.id,
                correct_species_id=nab.id,
                source=BACKFILL_SOURCE,
                rationale=(
                    f"Scene mask backfill: cell {cell} is a known false-positive "
                    f"region; YOLO confidence {d.yolo_confidence:.2f} is below the "
                    f"{scene_mask.OVERRIDE_YOLO_CONFIDENCE} override."
                ),
            ))
            d.species_id = nab.id
            if i % COMMIT_BATCH == 0:
                db.commit()
                log.info("  committed %d/%d", i, len(hits))
        db.commit()
        log.info("Done. Relabeled %d detection(s).", len(hits))
    finally:
        db.close()


if __name__ == "__main__":
    main()
