"""Backfill biological sex plumage tags on historical detections of dimorphic species.

Scans detections for Northern Cardinal, House Finch, Downy/Hairy Woodpecker,
and Rose-breasted Grosbeak where `sex IS NULL`, reads the crop image, and
populates `sex` with 'male', 'female', or NULL (when ambiguous).
"""
import argparse
import logging
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from db.models import Detection, Species
from db.session import SessionLocal, init_db
from pipeline.dimorphism import DIMORPHIC_SPECIES, classify_sex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def backfill_sex(db: Session, dry_run: bool = False, batch_size: int = 200) -> dict[str, int]:
    species_names = list(DIMORPHIC_SPECIES)
    log.info("Querying unclassified detections for dimorphic species: %s", species_names)

    query = (
        db.query(Detection, Species.common_name)
        .join(Species, Detection.species_id == Species.id)
        .filter(Species.common_name.in_(species_names))
        .filter(Detection.sex.is_(None))
        .order_by(Detection.id.desc())
    )

    total_candidates = query.count()
    log.info("Found %d detections awaiting sex classification", total_candidates)

    stats = {"male": 0, "female": 0, "ambiguous": 0, "missing_crop": 0, "total": total_candidates}
    processed = 0

    for det, sp_name in query.yield_per(batch_size):
        crop_file = DATA_DIR / det.crop_path
        if not crop_file.exists():
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
                det.sex = res.sex
        else:
            stats["ambiguous"] += 1

        processed += 1
        if processed % batch_size == 0:
            if not dry_run:
                db.commit()
            log.info("Processed %d / %d: %s", processed, total_candidates, stats)

    if not dry_run:
        db.commit()

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
