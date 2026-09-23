"""Cleanse unreviewed non-regional detections by re-fusing with regional priors.

Finds detections of species not in the regional baseline that were not audio-confirmed,
are not sentinel labels, and have no human correction.
Re-runs Bayesian fusion on their stored raw_predictions using the regional allowlist
and 20x vagrant penalty:
  - If a regional species (e.g. Northern Cardinal, Mourning Dove) now wins with >= 0.35 conf:
    reassigns to the true local species.
  - If non-regional species still has >= 0.80 posterior probability: keeps it.
  - Otherwise: demotes to Unidentified (species_id=None, confidence=0.0) so the species
    counts and feeds are not polluted, while preserving the crop for manual labeling.

Usage:
    python scripts/reclassify_vagrants.py --dry-run
    python scripts/reclassify_vagrants.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import SENTINEL_LABELS, Correction, Detection, Species
from db.session import SessionLocal
from pipeline.fuse import fuse, is_regional_species

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reclassify_vagrants")


def get_or_create_species(db, common_name: str) -> Species:
    sp = db.query(Species).filter(Species.common_name == common_name).first()
    if sp is None:
        sp = Species(common_name=common_name, scientific_name="", is_rare=False)
        db.add(sp)
        db.flush()
    return sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print changes without saving to DB")
    parser.add_argument("--limit", type=int, default=None, help="Max detections to process")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # Find all detections with species assigned, not audio confirmed, and no user correction
        q = (
            db.query(Detection)
            .join(Species, Detection.species_id == Species.id)
            .outerjoin(Correction, Correction.detection_id == Detection.id)
            .filter(Correction.id.is_(None))
            .filter(Detection.species_id.isnot(None))
            .filter(Detection.audio_confirmed.is_(False))
        )
        all_unreviewed = q.all()
        log.info("Found %d unreviewed detections total", len(all_unreviewed))

        # Filter down to candidates whose current species is NOT regional and NOT a sentinel
        candidates = []
        for d in all_unreviewed:
            if d.species and d.species.common_name not in SENTINEL_LABELS and not is_regional_species(d.species.common_name):
                candidates.append(d)

        log.info("Found %d detections of non-regional species needing re-fusion", len(candidates))
        if args.limit:
            candidates = candidates[: args.limit]

        reassigned_count = Counter()
        demoted_count = Counter()
        kept_count = Counter()

        for d in candidates:
            sp_name = d.species.common_name if d.species else "Unknown"
            raw = d.raw_predictions
            when = d.visit.started_at if d.visit else d.created_at
            bbox = tuple(d.bbox) if d.bbox and len(d.bbox) == 4 else None

            candidate_preds: list[tuple[str, float]] = []
            if isinstance(raw, list) and raw:
                for item in raw:
                    if isinstance(item, dict) and "species" in item and "p" in item:
                        candidate_preds.append((item["species"], float(item["p"])))

            new_species_name: str | None = None
            new_conf: float = 0.0

            if candidate_preds:
                fused = fuse(candidate_preds, db=db, when=when, bbox=bbox)
                if fused:
                    top = fused[0]
                    reg = is_regional_species(top.species)
                    if reg and top.probability >= 0.35:
                        new_species_name = top.species
                        new_conf = top.probability
                    elif not reg and top.probability >= 0.80:
                        new_species_name = top.species
                        new_conf = top.probability
                    else:
                        new_species_name = None
                        new_conf = 0.0
            else:
                if (d.confidence or 0.0) >= 0.80:
                    new_species_name = sp_name
                    new_conf = d.confidence
                else:
                    new_species_name = None
                    new_conf = 0.0

            if new_species_name == sp_name:
                kept_count[sp_name] += 1
            elif new_species_name is not None and new_species_name != sp_name:
                reassigned_count[(sp_name, new_species_name)] += 1
                if not args.dry_run:
                    new_sp = get_or_create_species(db, new_species_name)
                    d.species_id = new_sp.id
                    d.confidence = new_conf
            else:
                demoted_count[sp_name] += 1
                if not args.dry_run:
                    d.species_id = None
                    d.confidence = 0.0

        if not args.dry_run:
            db.commit()
            log.info("Committed changes to DB.")

        log.info("=== SUMMARY ===")
        log.info("Reassigned to true local species:")
        for (old_sp, new_sp), count in reassigned_count.most_common():
            log.info("  %s -> %s: %d", old_sp, new_sp, count)

        log.info("Demoted to Unidentified:")
        total_demoted = sum(demoted_count.values())
        for sp, count in demoted_count.most_common():
            log.info("  %s: %d", sp, count)
        log.info("Total demoted: %d", total_demoted)

        log.info("Kept (conf >= 0.80 post-fusion):")
        total_kept = sum(kept_count.values())
        for sp, count in kept_count.most_common():
            log.info("  %s: %d", sp, count)
        log.info("Total kept: %d", total_kept)

    finally:
        db.close()


if __name__ == "__main__":
    main()
