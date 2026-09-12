"""Sweep the whole archive for recurring boxes and relabel them Not a bird.

The nightly job in pipeline.worker looks at the last fortnight. This is the
one-off that walks everything already recorded, which is where the bulk of
the cruft is: thousands of crops of the same handful of fixed objects, each
one currently claiming to be a bird in the feed.

Windowed rather than global on purpose. A box that recurs across three days
in August is a fixture in August; the same coordinates in September might be
a perch, because the feeder moves and the light changes. Each window is
judged on its own, which matches how the nightly job will behave from here.

Dry run by default, like scripts/backfill_scene_mask.py and unlike the older
backfills, because it rewrites species_id on rows you are looking at. Every
change writes a Correction with source "recurrence", so the whole pass can
be undone by deleting those.

Usage:
    cd backend
    python scripts/backfill_recurrence.py                     # dry run, all time
    python scripts/backfill_recurrence.py --since 2026-08-01
    python scripts/backfill_recurrence.py --apply
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func  # noqa: E402

from db.models import Visit  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline import recurrence  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_recurrence")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the changes; without it this only reports")
    ap.add_argument("--since", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--until", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--window-days", type=int, default=recurrence.WINDOW_DAYS)
    args = ap.parse_args()

    db = SessionLocal()
    try:
        first = db.query(func.min(Visit.started_at)).scalar()
        last = db.query(func.max(Visit.started_at)).scalar()
        if first is None:
            log.info("No visits. Nothing to do.")
            return
        start = datetime.strptime(args.since, "%Y-%m-%d") if args.since else first
        end = datetime.strptime(args.until, "%Y-%m-%d") if args.until else last + timedelta(days=1)
        step = timedelta(days=args.window_days)
        log.info("Archive spans %s to %s. Sweeping %s to %s in %d-day windows.%s",
                 first.date(), last.date(), start.date(), end.date(), args.window_days,
                 "" if args.apply else "  DRY RUN.")

        totals = Counter()
        skipped = Counter()
        cursor = start
        while cursor < end:
            stop = min(cursor + step, end)
            r = recurrence.relabel_fixtures(db, cursor, stop, apply=args.apply)
            n = r.get("relabeled") if args.apply else r.get("would_relabel", 0)
            if r.get("fixtures"):
                log.info("  %s → %s : %2d fixture(s), %4d candidate(s), %4d to relabel",
                         cursor.date(), stop.date(), r["fixtures"],
                         r.get("candidates", 0), n)
            totals["fixtures"] += r.get("fixtures", 0)
            totals["candidates"] += r.get("candidates", 0)
            totals["relabel"] += n
            for k, v in (r.get("skipped") or {}).items():
                skipped[k] += v
            cursor = stop

        log.info("=" * 66)
        log.info("%d fixture(s) across all windows", totals["fixtures"])
        log.info("%d detection(s) sat on one", totals["candidates"])
        log.info("%d %s relabeled Not a bird", totals["relabel"],
                 "were" if args.apply else "would be")
        for k, v in skipped.most_common():
            log.info("    left alone, %-42s %5d", k + ":", v)
        log.info("=" * 66)
        if not args.apply:
            log.info("Dry run. Re-run with --apply to write these changes.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
