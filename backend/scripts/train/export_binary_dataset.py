"""Export a binary bird/not-a-bird ImageFolder dataset.

Pivots away from species fine-tuning toward the high-value win:
suppressing YOLO's false positives at the yard. The species
classifier (`dennisjooo/Birds-Classifier-EfficientNetB2`) currently
does double-duty as species ID + NAB rejection, but with only 6-7
species above sample threshold the species half doesn't move much. A
dedicated bird/not-bird post-filter, trained on ALL labeled crops
(including LLM-HIGH and LLM-MEDIUM), should cut false positives at
roughly the same precision the current pipeline produces with much
less compute.

Why we include LLM-HIGH and LLM-MEDIUM by default here (vs the
species export, which is GOLD-only):
  - "Is this a bird?" is a much coarser-grained judgment than species
    ID. Claude's HIGH and MEDIUM bird/NAB calls are correct >95% of
    the time even when species is uncertain. The remaining label
    noise (~3-5%) is well below what a binary classifier shrugs off
    via training-time augmentation.
  - This balloons the dataset from ~2,700 samples (GOLD only) to
    ~7,000+ — enough that a small model can actually generalize.

Filters:
  - Crop file exists, ≥48×48 px, aspect ∈ [0.4, 2.5]. Wider tolerance
    than the species export since bird/not-bird is augmentation-robust.
  - Drops "Unknown bird" (uninformative as supervision).

Usage:
    cd backend
    python scripts/train/export_binary_dataset.py \\
        --out /tmp/birdwatcher/binary_dataset
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy import func

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import (  # noqa: E402
    NOT_A_BIRD_LABEL,
    Correction,
    Detection,
    Species,
    UNKNOWN_BIRD_LABEL,
)
from db.session import SessionLocal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_binary_dataset")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# All correction-source tiers — binary task tolerates LLM label noise.
ACCEPTED_SOURCES = (None, "user-confirmed", "llm-claude-confirmed",
                    "llm-claude", "llm-claude-medium")

# Minimum / maximum dimensions for a usable training crop.
MIN_DIM = 48
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5


def _crop_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            w, h = im.size
    except (UnidentifiedImageError, OSError):
        return False
    if w < MIN_DIM or h < MIN_DIM:
        return False
    aspect = w / h
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    db = SessionLocal()
    try:
        # Latest correction per detection (so a user override of a
        # Claude label wins).
        sub = (
            db.query(Correction.detection_id, func.max(Correction.id).label("latest"))
            .group_by(Correction.detection_id)
            .subquery()
        )
        rows = (
            db.query(Detection, Species.common_name, Correction.source)
            .join(sub, sub.c.detection_id == Detection.id)
            .join(Correction, Correction.id == sub.c.latest)
            .join(Species, Species.id == Correction.correct_species_id)
            .all()
        )

        # Bucket into bird/not_a_bird, drop Unknown bird.
        buckets: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        n_dropped_bad_crop = 0
        for det, name, source in rows:
            if name == UNKNOWN_BIRD_LABEL:
                continue
            if source not in ACCEPTED_SOURCES:
                continue
            crop_abs = DATA_DIR / det.crop_path
            if not _crop_ok(crop_abs):
                n_dropped_bad_crop += 1
                continue
            cls = "not_a_bird" if name == NOT_A_BIRD_LABEL else "bird"
            buckets[cls].append((det.id, crop_abs))

        log.info("Dropped %d crops (missing/tiny/extreme-aspect)", n_dropped_bad_crop)
        for cls, items in buckets.items():
            log.info("  %s: %d", cls, len(items))

        # Wipe and recreate output tree.
        out = args.out.resolve()
        if out.exists():
            log.info("Removing existing %s …", out)
            shutil.rmtree(out)
        (out / "train").mkdir(parents=True)
        (out / "val").mkdir(parents=True)

        total_train = 0
        total_val = 0
        for cls in ("bird", "not_a_bird"):
            samples = buckets[cls]
            random.shuffle(samples)
            n_val = max(1, int(len(samples) * args.val_frac))
            val, train = samples[:n_val], samples[n_val:]
            (out / "train" / cls).mkdir()
            (out / "val" / cls).mkdir()
            for det_id, crop_abs in train:
                dst = out / "train" / cls / f"det_{det_id}.jpg"
                if args.symlink:
                    dst.symlink_to(crop_abs)
                else:
                    shutil.copy(crop_abs, dst)
            for det_id, crop_abs in val:
                dst = out / "val" / cls / f"det_{det_id}.jpg"
                if args.symlink:
                    dst.symlink_to(crop_abs)
                else:
                    shutil.copy(crop_abs, dst)
            total_train += len(train)
            total_val += len(val)

        print()
        print(f"Wrote {total_train} train + {total_val} val crops to {out}")
        print(f"  bird:        {len(buckets['bird'])} total")
        print(f"  not_a_bird:  {len(buckets['not_a_bird'])} total")
        print()
        print("Next: python scripts/train/finetune_binary.py "
              f"--data {out} --out {out.parent / 'binary_filter'}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
