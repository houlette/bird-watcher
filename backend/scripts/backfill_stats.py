"""Backfill PipelineStatsDaily for every UTC date that has at least one Visit.

Run-once-after-deploy script:

    cd backend && python scripts/backfill_stats.py

Safe to re-run — each date's row is upserted in place.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python scripts/backfill_stats.py` from the backend/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.session import SessionLocal, init_db  # noqa: E402
from pipeline.stats import dates_with_visits, upsert_daily_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_stats")


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        days = list(dates_with_visits(db))
        log.info("Found %d distinct visit-dates to backfill", len(days))
        for i, d in enumerate(days, 1):
            row = upsert_daily_stats(db, d)
            log.info(
                "[%d/%d] %s: %d clips, %d detections, %d corrections",
                i, len(days), d.isoformat(),
                row.clips_received, row.detections_total, row.detections_user_corrected,
            )
        log.info("Done")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
