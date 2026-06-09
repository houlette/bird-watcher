"""Populate crop_area_px / brightness / sharpness for old Detection rows.

After the schema migration in db/session.py adds the three crop-quality
columns, rows inserted before this commit have NULLs. This script loads
each affected crop, computes the three metrics via the same helper
pipeline/process.py uses at ingest, and writes them in batched commits.

Idempotent — skips rows that already have non-NULL values for all three
columns. Safe to re-run after partial completion.

Usage:
    cd backend
    python scripts/backfill_crop_quality.py            # all NULL rows
    python scripts/backfill_crop_quality.py --limit 100  # first 100
    python scripts/backfill_crop_quality.py --dry-run    # log what
                                                          # would update
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
from sqlalchemy import or_

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Detection  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline.process import _crop_quality  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_crop_quality")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMMIT_BATCH = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows processed (0 = all).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        log.info("Querying Detection rows missing quality metrics …")
        q = (
            db.query(Detection)
            .filter(or_(
                Detection.crop_area_px.is_(None),
                Detection.brightness.is_(None),
                Detection.sharpness.is_(None),
            ))
            .order_by(Detection.id.asc())
        )
        if args.limit:
            q = q.limit(args.limit)
        rows = q.all()
        log.info("Candidates: %d", len(rows))

        n_updated = 0
        n_missing_crop = 0
        n_unreadable = 0
        t0 = time.time()
        for i, det in enumerate(rows, 1):
            crop_path = DATA_DIR / det.crop_path
            if not crop_path.exists():
                n_missing_crop += 1
                continue
            img = cv2.imread(str(crop_path))
            if img is None or img.size == 0:
                n_unreadable += 1
                continue
            area_px, brightness, sharpness = _crop_quality(img, det.bbox or [])
            if not args.dry_run:
                det.crop_area_px = area_px
                det.brightness = brightness
                det.sharpness = sharpness
            n_updated += 1
            if i % COMMIT_BATCH == 0:
                if not args.dry_run:
                    db.commit()
                log.info("  %d/%d  updated=%d  missing=%d  unreadable=%d  "
                         "(elapsed %.0fs, %.1f /s)",
                         i, len(rows), n_updated, n_missing_crop, n_unreadable,
                         time.time() - t0, i / max(time.time() - t0, 1))
        if not args.dry_run:
            db.commit()
        log.info("=" * 60)
        log.info("Done in %.0fs.  updated=%d  missing_crop=%d  unreadable=%d",
                 time.time() - t0, n_updated, n_missing_crop, n_unreadable)
        if args.dry_run:
            log.info("DRY RUN — no commits.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
