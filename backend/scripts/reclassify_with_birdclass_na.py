"""Re-classify untouched detections with the freshly-deployed birdclass-na model.

The species classifier was swapped from denisjooo/EfficientNet to
houlette/birdclass-na (DINOv2-Base, 407-way NA-focused taxonomy) on
2026-06-09. The new model is ~+69 pp top-1 vs denisjooo on our test set
and beats birder-project on yard conditions by ~+39 pp. Old detections
labeled (or rejected) by denisjooo are likely improvable.

Scope — only touches detections the user/Claude haven't labeled:
  1. No Correction row exists, AND
  2. Either species_id IS NULL (was "Unidentified") OR confidence < THRESHOLD.

For each candidate, runs the FULL ingest-time classifier path:
  - classify_bird(crop)  → top-K species or [] if OTHER threshold met
  - binary_filter        → can override species → NAB

Effect:
  - new species accepted: update species_id, confidence, raw_predictions.
  - OTHER / NAB rejection: clear species_id, set confidence=0 (Unidentified).

Idempotent: re-running on the same data gives the same answers. NAB
overrides land as Unidentified, NOT as a "Not a bird" Correction row —
the binary filter alone isn't a user-strength signal.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from sqlalchemy import or_

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Correction, Detection, Species  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline.binary_filter import (  # noqa: E402
    is_enabled as binary_filter_enabled,
    nab_probability,
)
from pipeline.classify import classify_bird  # noqa: E402
from settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reclassify_with_birdclass_na")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOW_CONFIDENCE_DEFAULT = 0.5
COMMIT_BATCH = 100


def _resolve_species(db, common_name: str) -> int:
    """Get-or-create Species by common_name; mirror process.py:_resolve_species
    so newly-rescued species names land in the same canonical Species table
    the production pipeline writes to."""
    name = common_name.strip()
    sp = db.query(Species).filter(Species.common_name == name).one_or_none()
    if sp:
        return sp.id
    sp = Species(common_name=name, scientific_name="", is_rare=False)
    db.add(sp)
    db.flush()
    return sp.id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-conf", type=float, default=LOW_CONFIDENCE_DEFAULT,
                    help=f"Re-classify detections whose existing confidence is below "
                         f"this (default: {LOW_CONFIDENCE_DEFAULT}). Detections with "
                         f"confidence ≥ this are left alone (assumes old model was "
                         f"probably right when it was confident).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap candidates processed (0 = no cap).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run inference and report what WOULD change; don't commit.")
    ap.add_argument("--skip-binary-filter", action="store_true",
                    help="Don't apply the binary NAB filter on top of species classifier "
                         "(otherwise mimics full production semantics).")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        # ---- Candidates: no Correction, and (NULL species OR low confidence) ----
        log.info("Querying candidates …")
        candidates = (
            db.query(Detection)
            .outerjoin(Correction, Correction.detection_id == Detection.id)
            .filter(Correction.id.is_(None))
            .filter(or_(
                Detection.species_id.is_(None),
                Detection.confidence < args.low_conf,
            ))
            .order_by(Detection.id.asc())
            .all()
        )
        log.info("Total candidates: %d", len(candidates))
        if args.limit:
            candidates = candidates[: args.limit]
            log.info("Capped to first %d", len(candidates))

        n_unidentified_before = sum(1 for d in candidates if d.species_id is None)
        n_low_conf_before = len(candidates) - n_unidentified_before
        log.info("  was Unidentified: %d  was low-conf: %d",
                 n_unidentified_before, n_low_conf_before)
        log.info("Binary filter: %s",
                 "ENABLED" if (binary_filter_enabled() and not args.skip_binary_filter)
                 else "disabled (skipped)")

        # ---- Iterate ----
        n_accepted = 0      # got a species back
        n_rejected_other = 0  # species classifier returned []
        n_rejected_nab = 0    # binary filter overrode to NAB
        n_unchanged = 0       # no change (e.g. crop missing)
        n_missing_crop = 0
        species_count: Counter[str] = Counter()
        old_to_new: list[tuple[str, str]] = []  # for sample logging

        apply_binary = binary_filter_enabled() and not args.skip_binary_filter
        t0 = time.time()
        for i, det in enumerate(candidates, 1):
            crop_path = DATA_DIR / det.crop_path
            if not crop_path.exists():
                n_missing_crop += 1
                n_unchanged += 1
                continue
            img = cv2.imread(str(crop_path))
            if img is None or img.size == 0:
                n_missing_crop += 1
                n_unchanged += 1
                continue

            preds = classify_bird(img)
            if not preds:
                # Species classifier rejected (OTHER mass over threshold).
                if det.species_id is not None:
                    if not args.dry_run:
                        det.species_id = None
                        det.confidence = 0.0
                        det.raw_predictions = []
                    n_rejected_other += 1
                else:
                    n_unchanged += 1
                continue

            if apply_binary:
                nab_p = nab_probability(img)
                if nab_p is not None and nab_p >= settings.bird_binary_nab_threshold:
                    if det.species_id is not None:
                        if not args.dry_run:
                            det.species_id = None
                            det.confidence = 0.0
                            det.raw_predictions = []
                        n_rejected_nab += 1
                    else:
                        n_unchanged += 1
                    continue

            top = preds[0]
            new_species_id = _resolve_species(db, top.species) if not args.dry_run else None
            old_label = det.species.common_name if det.species else "(Unidentified)"
            old_conf = det.confidence

            if not args.dry_run:
                det.species_id = new_species_id
                det.confidence = top.probability
                det.raw_predictions = [
                    {"species": p.species, "p": p.probability, "raw": p.raw_label,
                     "audio": False}
                    for p in preds[:5]
                ]
            n_accepted += 1
            species_count[top.species] += 1
            if len(old_to_new) < 12:
                old_to_new.append((
                    f"#{det.id} {old_label} @ {old_conf:.2f}",
                    f"{top.species} @ {top.probability:.2f}",
                ))

            if not args.dry_run and i % COMMIT_BATCH == 0:
                db.commit()
                log.info("  %d/%d  accepted=%d  rejected_other=%d  rejected_nab=%d  "
                         "(elapsed %.0fs)",
                         i, len(candidates),
                         n_accepted, n_rejected_other, n_rejected_nab,
                         time.time() - t0)

        if not args.dry_run:
            db.commit()

        # ---- Summary ----
        elapsed = time.time() - t0
        log.info("=" * 60)
        log.info("Scanned %d candidates in %.0fs (%.1f /s)",
                 len(candidates), elapsed, len(candidates) / max(elapsed, 1))
        log.info("  re-classified to species: %d", n_accepted)
        log.info("  rejected (OTHER):         %d", n_rejected_other)
        log.info("  rejected (binary→NAB):    %d", n_rejected_nab)
        log.info("  no change:                %d  (of which missing crop: %d)",
                 n_unchanged, n_missing_crop)
        log.info("Top 15 species rescued:")
        for sp, n in species_count.most_common(15):
            log.info("  %-30s %d", sp, n)
        log.info("Sample changes (showing %d of %d):", len(old_to_new), n_accepted)
        for before, after in old_to_new:
            log.info("  %s  →  %s", before, after)
        if args.dry_run:
            log.info("DRY RUN — no commits. Re-run without --dry-run to apply.")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
