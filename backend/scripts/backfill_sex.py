"""Backfill biological sex plumage tags on historical detections of dimorphic species.

Scans detections for Northern Cardinal, House Finch, Downy/Hairy Woodpecker,
and Rose-breasted Grosbeak where `sex IS NULL`, reads the crop image, and
populates `sex` with 'male', 'female', or NULL (when ambiguous).
"""
import argparse
import logging
from pathlib import Path
import sys
import time

# Ensure backend root is on sys.path for direct script execution
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import cv2
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from db.models import Detection, Species
from db.session import SessionLocal, init_db
from pipeline.dimorphism import DIMORPHIC_SPECIES, classify_sex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = BACKEND_DIR / "data"


def commit_updates(db: Session, updates: list[tuple[int, str]]) -> None:
    if not updates:
        return
    for attempt in range(10):
        try:
            for det_id, sex in updates:
                db.execute(
                    text("UPDATE detections SET sex = :sex WHERE id = :id"),
                    {"sex": sex, "id": det_id},
                )
            db.commit()
            return
        except OperationalError as e:
            db.rollback()
            if "locked" in str(e).lower() and attempt < 9:
                time.sleep(0.2 * (attempt + 1))
                continue
            raise


def backfill_sex(db: Session, dry_run: bool = False, batch_size: int = 100) -> dict[str, int]:
    species_names = list(DIMORPHIC_SPECIES)
    log.info("Querying unclassified detections for dimorphic species: %s", species_names)

    # Fetch rows as plain tuples so no open cursor or ORM session lock is held
    candidates = (
        db.query(Detection.id, Species.common_name, Detection.crop_path)
        .join(Species, Detection.species_id == Species.id)
        .filter(Species.common_name.in_(species_names))
        .filter(Detection.sex.is_(None))
        .order_by(Detection.id.desc())
        .all()
    )
    # Release read transaction
    db.commit()

    total_candidates = len(candidates)
    log.info("Found %d detections awaiting sex classification", total_candidates)

    stats = {"male": 0, "female": 0, "ambiguous": 0, "missing_crop": 0, "total": total_candidates}
    pending_updates: list[tuple[int, str]] = []
    processed = 0

    for det_id, sp_name, crop_path in candidates:
        processed += 1
        crop_file = DATA_DIR / crop_path if crop_path else None
        if not crop_file or not crop_file.exists():
            stats["missing_crop"] += 1
            continue

        img = cv2.imread(str(crop_file))
        if img is None:
            stats["missing_crop"] += 1
            continue

        res = classify_sex(img, sp_name)
        if res.sex:
            stats[res.sex] += 1
            if not dry_run:
                pending_updates.append((det_id, res.sex))
        else:
            stats["ambiguous"] += 1

        if len(pending_updates) >= batch_size:
            if not dry_run:
                commit_updates(db, pending_updates)
                pending_updates.clear()
            log.info("Processed %d / %d: %s", processed, total_candidates, stats)

    if not dry_run and pending_updates:
        commit_updates(db, pending_updates)
        pending_updates.clear()

    log.info("Backfill finished. Result: %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Backfill sex on dimorphic bird detections.")
    parser.add_argument("--dry-run", action="store_true", help="Calculate classifications without writing to DB")
    parser.add_argument("--batch-size", type=int, default=200, help="Commit batch size")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        backfill_sex(db, dry_run=args.dry_run, batch_size=args.batch_size)
    finally:
        db.close()


if __name__ == "__main__":
    main()
