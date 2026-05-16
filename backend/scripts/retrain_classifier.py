"""Fine-tune the bird classifier from accumulated user corrections.

STATUS: stub. The active-learning UI in the PWA writes `Correction` rows
every time the user fixes a mis-ID'd detection. This script's job, once
implemented, is to consume those rows and produce a yard-personalized
classifier checkpoint that replaces the dennisjooo default.

DESIGN (deferred until ~500 corrections have accumulated):

  1. Pull every Correction row joined to its Detection + Species.
     Filter to corrections older than ~24h so users have time to undo.
  2. Reconstruct the training pair: (crop_path, correct_species_name).
     Skip rows whose crop file no longer exists (storage TTL eviction).
  3. Group by species and require a per-species minimum (e.g. ≥10
     corrections) so we don't fit on a single-example class.
  4. Load the dennisjooo EfficientNet-B2 weights and freeze every layer
     except the final classifier head.
  5. Map the corrected species names onto the model's existing class
     indices (the gpiosenka label space). Corrections targeting species
     not in that space are dropped for this version of the script — the
     real solution is a second classifier head over yard species only.
  6. Train for 5-10 epochs with a small LR (1e-4) and early stopping on
     a 10% validation split. Save the new head's state_dict to
     `backend/models/yard_finetune_<timestamp>.pt`.
  7. Update settings.bird_classifier_model to point at a local path that
     classify.py knows how to load — i.e. add a 'load from disk' branch
     in classify._load() that grafts a custom head onto the base model.

Until that happens, this script prints what *would* be used so users can
inspect their correction history before committing to the work.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrain")


def main() -> int:
    from db.models import Correction, Detection, Species
    from db.session import SessionLocal, init_db

    init_db()
    db = SessionLocal()
    try:
        rows = (
            db.query(Correction, Detection, Species)
            .join(Detection, Correction.detection_id == Detection.id)
            .join(Species, Correction.correct_species_id == Species.id)
            .all()
        )
        if not rows:
            log.info("No corrections recorded yet. Use the PWA's 'Wrong species?' button to teach the system.")
            return 0

        per_species = Counter([s.common_name for _, _, s in rows])
        log.info("=" * 60)
        log.info("Correction summary (%d total)", len(rows))
        log.info("=" * 60)
        for name, n in per_species.most_common():
            log.info("  %-30s %d corrections", name, n)
        log.info("")
        log.info("Fine-tuning not yet implemented. See module docstring for")
        log.info("the planned approach. Targeting ~500 corrections across ≥10")
        log.info("species before training is worth the effort.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
