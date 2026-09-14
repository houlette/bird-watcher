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
  - Drops "Unknown bird" (uninformative as supervision) and "Poor
    quality" (db/models.py: the user is saying the crop is noise).
  - Refuses the held-out yardstick in scripts/train/heldout.py, every
    other detection from the same visits, and every detection captured on
    a capture day reserved for the yardstick, so no retrain can absorb the
    rows it will be judged on or their near-copies.

Train/val is split by group, not by detection. Crops from one visit share
the second, the light and usually the object, and 72.6% of labelled
crops had a sibling in the same visit, so a random split put near-copies
on both sides and flattered val accuracy. The default group is the local
capture day, which also keeps a recurring glint or leaf from one
afternoon out of both sides; `--group-by visit` is the weaker option.

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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import (  # noqa: E402
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    UNKNOWN_BIRD_LABEL,
    Detection,
    Visit,
)
from db.session import SessionLocal  # noqa: E402
from scripts.train import heldout  # noqa: E402
from settings import settings  # noqa: E402

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

CLASSES = ("bird", "not_a_bird")


@dataclass(frozen=True)
class Sample:
    detection_id: int
    crop: Path
    cls: str
    visit_id: int
    captured_at: datetime


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


def collect(
    db, yardstick: heldout.Yardstick, tz: ZoneInfo, data_dir: Path = DATA_DIR
) -> tuple[list[Sample], Counter]:
    """Every exportable sample, and a Counter of rows left out by reason."""
    labels = heldout.latest_labels(db)
    frozen, siblings = heldout.withheld_detection_ids(db, yardstick)
    dets = {
        det.id: (det, started_at)
        for det, started_at in db.query(Detection, Visit.started_at)
        .join(Visit, Visit.id == Detection.visit_id)
        .filter(Detection.id.in_(list(labels)))
    }
    samples: list[Sample] = []
    dropped: Counter[str] = Counter()
    for det_id, (source, _, label) in sorted(labels.items()):
        if label in (UNKNOWN_BIRD_LABEL, POOR_QUALITY_LABEL):
            dropped[f"label {label}"] += 1
            continue
        if source not in ACCEPTED_SOURCES:
            dropped[f"source {source}"] += 1
            continue
        if det_id in frozen:
            dropped["held out: yardstick row"] += 1
            continue
        if det_id in siblings:
            dropped["held out: same visit as a yardstick row"] += 1
            continue
        det, started_at = dets[det_id]
        if heldout.local_day(started_at, tz) in yardstick.heldout_days:
            dropped["held out: capture day reserved for the yardstick"] += 1
            continue
        crop = data_dir / det.crop_path
        if not _crop_ok(crop):
            dropped["crop missing, tiny or extreme aspect"] += 1
            continue
        cls = "not_a_bird" if label == NOT_A_BIRD_LABEL else "bird"
        samples.append(Sample(det_id, crop, cls, det.visit_id, started_at))
    return samples, dropped


def group_key(sample: Sample, by: str, tz: ZoneInfo):
    if by == "visit":
        return sample.visit_id
    return heldout.local_day(sample.captured_at, tz)


def split_by_group(samples: list[Sample], key, val_frac: float, rng: random.Random):
    """Walk the groups in seeded random order and put each in val if val
    still holds at most `val_frac` of each class with it added. No group
    lands on both sides.

    Both limits matter on this data. On 2026-09-13 one day, 2026-05-25, held
    28% of the exportable crops, and stopping at the first group to cross
    the target put 30.5% of crops in val. A single limit on the total
    instead filled val with the small bird-only days from June, giving 512
    birds and 85 junk crops against a train set that was half junk."""
    groups: dict = defaultdict(list)
    for s in samples:
        groups[key(s)].append(s)
    if len(groups) < 2:
        raise SystemExit(f"Only {len(groups)} group(s); cannot split train from val.")
    caps = {cls: val_frac * n for cls, n in Counter(s.cls for s in samples).items()}
    order = sorted(groups, key=str)
    rng.shuffle(order)
    val_keys = set()
    in_val: Counter[str] = Counter()
    for k in order:
        adds = Counter(s.cls for s in groups[k])
        if all(in_val[cls] + n <= caps[cls] for cls, n in adds.items()):
            val_keys.add(k)
            in_val += adds
    if not val_keys:
        val_keys.add(min(order, key=lambda k: len(groups[k])))
    train = [s for k in order if k not in val_keys for s in groups[k]]
    val = [s for k in order if k in val_keys for s in groups[k]]
    return train, val, len(groups), len(val_keys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--group-by", choices=("day", "visit"), default="day",
                    help="Unit that train and val may not share (default: local capture day).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true")
    args = ap.parse_args()

    yardstick = heldout.load_yardstick()
    tz = ZoneInfo(settings.camera_timezone)
    db = SessionLocal()
    try:
        samples, dropped = collect(db, yardstick, tz)
    finally:
        db.close()

    train, val, n_groups, n_val_groups = split_by_group(
        samples, lambda s: group_key(s, args.group_by, tz), args.val_frac, random.Random(args.seed)
    )

    # Wipe and recreate output tree.
    out = args.out.resolve()
    if out.exists():
        log.info("Removing existing %s …", out)
        shutil.rmtree(out)
    for split_name, split in (("train", train), ("val", val)):
        for cls in CLASSES:
            (out / split_name / cls).mkdir(parents=True)
        for s in split:
            dst = out / split_name / s.cls / f"det_{s.detection_id}.jpg"
            if args.symlink:
                dst.symlink_to(s.crop)
            else:
                shutil.copy(s.crop, dst)

    key_desc = f"local capture day ({settings.camera_timezone})" if args.group_by == "day" else "visit_id"
    print()
    print(f"Wrote {len(train)} train + {len(val)} val crops to {out}")
    print(f"Split grouped by {key_desc}: {n_groups} groups, {n_val_groups} in val "
          f"({len(val) / max(len(samples), 1):.1%} of crops)")
    print()
    print(f"  {'':<6} {'bird':>7} {'not_a_bird':>11} {'total':>7}  bird share")
    for split_name, split in (("train", train), ("val", val)):
        counts = Counter(s.cls for s in split)
        total = len(split)
        print(f"  {split_name:<6} {counts['bird']:>7} {counts['not_a_bird']:>11} {total:>7}  "
              f"{counts['bird'] / max(total, 1):.1%}")
    recent = Counter(s.cls for s in samples if heldout.local_day(s.captured_at, tz) >= heldout.DAY_SPLIT_FROM)
    print(f"  captured on or after {heldout.DAY_SPLIT_FROM}: {recent['bird']} bird, "
          f"{recent['not_a_bird']} not_a_bird")
    print()
    held = sum(n for reason, n in dropped.items() if reason.startswith("held out"))
    print(f"Withheld {held} labelled crops for the held-out yardstick frozen {yardstick.frozen_at}:")
    print(f"  {dropped['held out: yardstick row']} yardstick rows, "
          f"{dropped['held out: same visit as a yardstick row']} from the same visits, "
          f"{dropped['held out: capture day reserved for the yardstick']} from "
          f"{len(yardstick.heldout_days)} reserved capture days")
    print("Other rows left out:")
    for reason, n in sorted(dropped.items()):
        if not reason.startswith("held out"):
            print(f"  {n:>6}  {reason}")
    print()
    print("Next: python scripts/train/finetune_binary.py "
          f"--data {out} --out {out.parent / 'binary_filter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
