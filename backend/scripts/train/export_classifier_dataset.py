"""Export labeled crops as a fine-tuning-ready ImageFolder dataset.

The production classifier is `dennisjooo/Birds-Classifier-EfficientNetB2`
(see pipeline/classify.py:DEFAULT_MODEL). To fine-tune the same arch on
our yard-specific data we need a dataset on disk in the standard
ImageFolder layout:

    dataset/
      train/
        Mourning_Dove/det_12345.jpg
        Rock_Pigeon/det_12346.jpg
        Not_a_bird/det_12347.jpg
        ...
      val/
        ...

This script builds that, pulling crops via `Detection.crop_path` for
each non-sentinel-or-NAB Correction. NAB is kept as a separate class so
the fine-tuned model can learn to suppress false positives — the
biggest QoL win after the camera move.

Filters applied:
  - Only species with ≥ MIN_PER_CLASS labels (default 30). Below that
    a per-class softmax head overfits.
  - Crop file must exist on disk and be at least 64×64 px.
  - Crop aspect in [0.5, 2.0] — drops the worst of the pre-fix narrow
    strips so we don't teach the model "narrow strip = bird."
  - GOLD tier by default. Use --include-high to also include the
    auto-committed Claude-HIGH labels (adds volume, ~5-10% label noise).

Splits 90/10 train/val. The val split is sampled stratified per-species
so even rare classes have a few held-out examples.

Usage:
    cd backend
    python scripts/train/export_classifier_dataset.py \\
        --out /tmp/birdwatcher_dataset
    python scripts/train/export_classifier_dataset.py \\
        --out /tmp/birdwatcher_dataset \\
        --include-high --min-per-class 50
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
    SENTINEL_LABELS,
    Correction,
    Detection,
    Species,
    UNKNOWN_BIRD_LABEL,
)
from db.session import SessionLocal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("export_classifier_dataset")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Trust tiers. See pipeline/stats.py for the same taxonomy. Excluding
# "Unknown bird" everywhere — it's a label we use when *we* don't know,
# which is useless as classifier supervision (the model can't learn
# "not classifiable").
GOLD_SOURCES = (None, "user-confirmed", "llm-claude-confirmed")
HIGH_SOURCES = ("llm-claude",)


def _slug(name: str) -> str:
    """ImageFolder-friendly directory name."""
    return name.replace(" ", "_").replace("'", "").replace("/", "_")


def _crop_ok(path: Path) -> bool:
    """Filter unusable crops: missing file, too small, or extreme aspect."""
    if not path.exists():
        return False
    try:
        with Image.open(path) as im:
            w, h = im.size
    except (UnidentifiedImageError, OSError):
        return False
    if w < 64 or h < 64:
        return False
    aspect = w / h
    if aspect < 0.5 or aspect > 2.0:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory (will be wiped and recreated).")
    ap.add_argument("--include-high", action="store_true",
                    help="Include Claude-HIGH labels (more volume, some noise).")
    ap.add_argument("--min-per-class", type=int, default=30,
                    help="Skip species below this gold+included threshold.")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="Fraction held out per-class for validation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink crops instead of copying (saves disk).")
    args = ap.parse_args()

    random.seed(args.seed)
    sources = list(GOLD_SOURCES)
    if args.include_high:
        sources += list(HIGH_SOURCES)

    db = SessionLocal()
    try:
        # Latest correction per detection (so a user override of an LLM
        # label is the one we use). Pulls (Detection, species_name) only
        # for detections whose latest correction is in our source tiers.
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
            .filter(Correction.source.in_([s for s in sources if s is not None])
                    | Correction.source.is_(None))
            .all()
        )

        # Drop "Unknown bird" — uninformative. Keep "Not a bird" — that's
        # the most valuable negative class.
        rows = [r for r in rows if r[1] != UNKNOWN_BIRD_LABEL]

        # Bucket by species, filter, then split.
        by_species: dict[str, list[tuple[int, Path]]] = defaultdict(list)
        for det, name, _src in rows:
            crop_abs = DATA_DIR / det.crop_path
            if not _crop_ok(crop_abs):
                continue
            by_species[name].append((det.id, crop_abs))

        # Apply min-per-class filter.
        species_passing = {k: v for k, v in by_species.items() if len(v) >= args.min_per_class}
        dropped = [(k, len(v)) for k, v in by_species.items() if len(v) < args.min_per_class]
        log.info("Species passing min-per-class=%d: %d (%d dropped)",
                 args.min_per_class, len(species_passing), len(dropped))
        if dropped:
            log.info("Dropped (insufficient samples): %s",
                     ", ".join(f"{name}({n})" for name, n in sorted(dropped, key=lambda x: -x[1])[:10]))

        # Wipe and recreate the output tree.
        out = args.out.resolve()
        if out.exists():
            log.info("Removing existing %s …", out)
            shutil.rmtree(out)
        (out / "train").mkdir(parents=True)
        (out / "val").mkdir(parents=True)

        total_train = 0
        total_val = 0
        summary: list[tuple[str, int, int]] = []
        for name, samples in sorted(species_passing.items(), key=lambda kv: -len(kv[1])):
            random.shuffle(samples)
            n_val = max(1, int(len(samples) * args.val_frac))
            val, train = samples[:n_val], samples[n_val:]

            slug = _slug(name)
            (out / "train" / slug).mkdir()
            (out / "val" / slug).mkdir()
            for det_id, crop_abs in train:
                dst = out / "train" / slug / f"det_{det_id}.jpg"
                if args.symlink:
                    dst.symlink_to(crop_abs)
                else:
                    shutil.copy(crop_abs, dst)
            for det_id, crop_abs in val:
                dst = out / "val" / slug / f"det_{det_id}.jpg"
                if args.symlink:
                    dst.symlink_to(crop_abs)
                else:
                    shutil.copy(crop_abs, dst)
            total_train += len(train)
            total_val += len(val)
            summary.append((name, len(train), len(val)))

        # Pretty summary.
        print()
        print(f"Wrote {total_train} train + {total_val} val crops across "
              f"{len(species_passing)} classes to {out}")
        print()
        print(f"  {'species':30s} {'train':>7s} {'val':>5s}")
        for name, t, v in summary:
            print(f"  {name[:30]:30s} {t:>7d} {v:>5d}")
        print()
        print("Next: feed this into the standard HuggingFace image-classification fine-tune loop.")
        print("Suggested launch (run on a GPU box, ~2-4 min/epoch on RTX 3060 for this size):")
        print()
        print(f"    python -m transformers.examples.pytorch.image-classification.run_image_classification \\")
        print(f"        --model_name_or_path dennisjooo/Birds-Classifier-EfficientNetB2 \\")
        print(f"        --train_dir {out / 'train'} \\")
        print(f"        --validation_dir {out / 'val'} \\")
        print(f"        --output_dir {out / 'finetuned'} \\")
        print(f"        --do_train --do_eval \\")
        print(f"        --learning_rate 1e-4 --num_train_epochs 5 \\")
        print(f"        --per_device_train_batch_size 32 \\")
        print(f"        --remove_unused_columns False \\")
        print(f"        --ignore_mismatched_sizes")
        print()
        print("After training, swap the deployed classifier by setting")
        print("BIRD_CLASSIFIER_MODEL to the output dir in backend/.env.")
    finally:
        db.close()
    return 0


# Sentinels referenced only for documentation; flake8 noqa.
_ = (SENTINEL_LABELS, NOT_A_BIRD_LABEL)


if __name__ == "__main__":
    raise SystemExit(main())
