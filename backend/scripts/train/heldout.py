"""The permanent held-out yardstick for the binary bird/not-a-bird filter.

Before this existed, the eval scored every corrected detection and the
export trained on a subset of the same rows, so about 96% of the yardstick
was memorised. This module fixes a set of detections that the export will
never emit, so a retrain and the model it replaces can be scored on rows
neither of them saw.

Three cohorts, frozen by detection id in heldout_yardstick.json:

  hard_birds  The user corrected the row to a real species after the
              deployed model was trained. Selected towards hard crops,
              since the classifier had the species wrong first.
  easy_birds  Haikubox heard the classifier's species and nobody has ever
              corrected the row. Independent of the user's labelling, and
              selected towards easy crops.
  junk        The user corrected the row to Not a bird after the deployed
              model was trained, on a row the filter had passed. These are
              the deployed model's misses by construction, so its catch
              rate on them is a floor, not an estimate.

On 2026-09-13 the deployed model killed about a third of the hard birds at
the live 0.75 threshold and about 2.5% of the easy ones. The two cohorts
are selected in opposite directions, so neither is a false-kill rate for
birds in general. Quote both, and use the yardstick for paired
old-versus-new comparison.

A candidate row is left out of every cohort if any crop from its visit is
in the deployed model's training set: siblings in one visit share the
light, the second and usually the object. For the same reason the export
withholds whole visits, not only the frozen ids.

Rows captured on or after DAY_SPLIT_FROM are split by local capture day,
because the first freeze held out every junk label since the filter went
live and left a retrain nothing recent to learn from. Days holding an easy
bird always go to the yardstick, since easy birds carry no label and could
never be trained on. The other days go to the yardstick or to training so
that each side holds about half of the hard birds and half of the junk,
and the export withholds every row captured on a yardstick day. Rows
captured earlier stay held out by visit, as before: those days were the
deployed model's training period.

The ids are frozen rather than re-derived on each run for two reasons.
Two models evaluated weeks apart must be scored on the same rows, and
labels the user adds after the freeze must stay available for training
instead of being swallowed by a cohort whose rule keeps matching.

Freezing is a one-off and refuses to overwrite:
    docker compose exec api python scripts/train/heldout.py freeze \\
        --trained-dataset /app/data/binary_dataset --out /tmp/heldout_yardstick.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import (  # noqa: E402
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    SENTINEL_LABELS,
    Correction,
    Detection,
    Species,
    Visit,
)

FROZEN_PATH = Path(__file__).with_name("heldout_yardstick.json")
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# The deployed model's weights were written 2026-06-07 16:18 UTC from a
# dataset exported 2026-06-06 21:56 UTC, so no correction made after this
# moment can be in its training set.
DEPLOYED_MODEL_TRAINED_AT = datetime(2026, 6, 7)
# First local capture day that is split between the yardstick and training.
DAY_SPLIT_FROM = "2026-06-07"
SPLIT_SEED = 0

HARD_BIRDS = "hard_birds"
EASY_BIRDS = "easy_birds"
JUNK = "junk"
COHORTS = (HARD_BIRDS, EASY_BIRDS, JUNK)
TRUTH = {HARD_BIRDS: "bird", EASY_BIRDS: "bird", JUNK: "nab"}

# Sources that are the user's own judgement. Used to notice a frozen row
# the user has relabelled since the freeze.
USER_SOURCES = (None, "user-confirmed")


@dataclass(frozen=True)
class Yardstick:
    frozen_at: str
    cohorts: dict[str, tuple[int, ...]]
    # Local capture days on or after DAY_SPLIT_FROM that belong to the
    # yardstick; the export withholds every row captured on them.
    heldout_days: frozenset[str] = field(default_factory=frozenset)

    @property
    def detection_ids(self) -> set[int]:
        return {i for ids in self.cohorts.values() for i in ids}


def local_day(captured_at: datetime, tz: ZoneInfo) -> str:
    """ISO date at the camera for a naive UTC capture time."""
    return captured_at.replace(tzinfo=timezone.utc).astimezone(tz).date().isoformat()


def load_yardstick(path: Path = FROZEN_PATH) -> Yardstick:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The export will not run without it, because "
            "it could then emit held-out rows into a training set."
        )
    raw = json.loads(path.read_text())
    return Yardstick(
        frozen_at=raw["frozen_at"],
        cohorts={c: tuple(raw["cohorts"][c]) for c in COHORTS},
        heldout_days=frozenset(raw["heldout_days"]),
    )


def resolve_crop(crop_path: str | None, data_dir: Path = DATA_DIR) -> Path | None:
    if not crop_path:
        return None
    p = Path(crop_path)
    if p.is_absolute():
        return p if p.exists() else None
    p = data_dir / crop_path
    return p if p.exists() else None


def _source_filter(sources):
    named = [s for s in sources if s is not None]
    clauses = [Correction.source.in_(named)] if named else []
    if None in sources:
        clauses.append(Correction.source.is_(None))
    return or_(*clauses)


def latest_labels(db, detection_ids=None, sources=None) -> dict[int, tuple[str | None, datetime, str]]:
    """{detection_id: (source, created_at, label)} for each detection's
    latest correction, optionally only among `sources`. A detection can
    carry several corrections, and only the newest is its label."""
    sub = db.query(Correction.detection_id, func.max(Correction.id).label("latest"))
    if sources is not None:
        sub = sub.filter(_source_filter(sources))
    if detection_ids is not None:
        sub = sub.filter(Correction.detection_id.in_(list(detection_ids)))
    sub = sub.group_by(Correction.detection_id).subquery()
    rows = (
        db.query(Correction.detection_id, Correction.source, Correction.created_at, Species.common_name)
        .join(sub, Correction.id == sub.c.latest)
        .join(Species, Species.id == Correction.correct_species_id)
        .all()
    )
    return {det_id: (source, created_at, label) for det_id, source, created_at, label in rows}


def trained_detection_ids(dataset_dir: Path) -> set[int]:
    """Detection ids in an ImageFolder export, read from its file names."""
    ids = set()
    for p in dataset_dir.glob("*/*/det_*.jpg"):
        m = re.fullmatch(r"det_(\d+)\.jpg", p.name)
        if m:
            ids.add(int(m.group(1)))
    return ids


def split_days(counts: dict[str, Counter], forced: set[str], rng: random.Random) -> set[str]:
    """Choose the capture days that go to the yardstick, so that it holds
    about half of each cohort counted in `counts`. `forced` days go to the
    yardstick whatever they hold.

    Days are placed largest first, each on whichever side leaves the two
    halves closer, with the seeded generator breaking ties. Placing them in
    calendar or random order instead would let a late burst, such as the
    205 junk labels captured 11 to 16 August, land on one side whole."""
    totals: Counter[str] = Counter()
    for c in counts.values():
        totals.update(c)
    keys = [k for k in sorted(totals) if totals[k]]

    def gap(held: Counter, train: Counter) -> float:
        return sum(((held[k] - train[k]) / totals[k]) ** 2 for k in keys)

    chosen = set(forced)
    held: Counter[str] = Counter()
    train: Counter[str] = Counter()
    for d in forced:
        held.update(counts.get(d, Counter()))
    rest = sorted(d for d in counts if d not in forced)
    rng.shuffle(rest)
    rest.sort(key=lambda d: -sum(counts[d][k] / totals[k] for k in keys))
    for d in rest:
        to_held = gap(held + counts[d], train)
        to_train = gap(held, train + counts[d])
        if to_held < to_train or (to_held == to_train and rng.random() < 0.5):
            held.update(counts[d])
            chosen.add(d)
        else:
            train.update(counts[d])
    return chosen


def select_cohorts(
    db, trained_ids: set[int], tz: ZoneInfo, data_dir: Path = DATA_DIR, seed: int = SPLIT_SEED
) -> tuple[dict[str, list[int]], dict, list[str]]:
    """Apply the cohort rules to the database as it is now.

    Returns the cohorts, how many rows per cohort matched the rule but were
    left out and why, and the capture days reserved for the yardstick."""
    candidates: dict[str, list[int]] = {c: [] for c in COHORTS}
    for det_id, (source, created_at, label) in latest_labels(db).items():
        if source is not None or created_at <= DEPLOYED_MODEL_TRAINED_AT:
            continue
        if label == NOT_A_BIRD_LABEL:
            candidates[JUNK].append(det_id)
        elif label not in SENTINEL_LABELS:
            candidates[HARD_BIRDS].append(det_id)
    corrected = db.query(Correction.id).filter(Correction.detection_id == Detection.id).exists()
    candidates[EASY_BIRDS] = [
        i for (i,) in db.query(Detection.id).filter(Detection.audio_confirmed == 1).filter(~corrected)
    ]

    trained_visits = {
        v for (v,) in db.query(Detection.visit_id).filter(Detection.id.in_(list(trained_ids))).distinct()
    }
    kept: dict[str, list[tuple[int, str]]] = {}
    why: dict[str, Counter] = {}
    for cohort, ids in candidates.items():
        dets = (
            db.query(Detection, Visit.started_at)
            .join(Visit, Visit.id == Detection.visit_id)
            .filter(Detection.id.in_(ids))
            .all()
        )
        kept[cohort], why[cohort] = [], Counter()
        for det, started_at in dets:
            if cohort == JUNK and det.nab_override_p is not None:
                why[cohort]["filter_fired"] += 1
            elif det.visit_id in trained_visits:
                why[cohort]["visit_in_deployed_training"] += 1
            elif resolve_crop(det.crop_path, data_dir) is None:
                why[cohort]["crop_missing"] += 1
            else:
                kept[cohort].append((det.id, local_day(started_at, tz)))

    per_day: dict[str, Counter] = defaultdict(Counter)
    forced = set()
    for cohort, rows in kept.items():
        for _, day in rows:
            if day < DAY_SPLIT_FROM:
                continue
            if cohort == EASY_BIRDS:
                forced.add(day)
            else:
                per_day[day][cohort] += 1
    heldout_days = split_days(per_day, forced, random.Random(seed))

    cohorts: dict[str, list[int]] = {}
    for cohort, rows in kept.items():
        ids = []
        for det_id, day in rows:
            if day >= DAY_SPLIT_FROM and day not in heldout_days:
                why[cohort]["capture_day_trainable"] += 1
            else:
                ids.append(det_id)
        cohorts[cohort] = sorted(ids)
    return cohorts, {c: dict(w) for c, w in why.items()}, sorted(heldout_days)


def withheld_detection_ids(db, yardstick: Yardstick) -> tuple[set[int], set[int]]:
    """(frozen ids, other detections that share a visit with a frozen id).
    The export refuses both."""
    frozen = yardstick.detection_ids
    visits = {v for (v,) in db.query(Detection.visit_id).filter(Detection.id.in_(list(frozen))).distinct()}
    siblings = {i for (i,) in db.query(Detection.id).filter(Detection.visit_id.in_(list(visits)))}
    return frozen, siblings - frozen


def _truth_now(label: str | None) -> str | None:
    if label == NOT_A_BIRD_LABEL:
        return "nab"
    if label == POOR_QUALITY_LABEL:
        return None
    return "bird"


def cohort_rows(db, yardstick: Yardstick | None = None, data_dir: Path = DATA_DIR):
    """The rows to score, as {cohort: [(detection_id, crop_file)]}, plus a
    Counter of frozen rows that cannot be scored any more.

    A frozen row drops out if it has been deleted, if its crop has gone, or
    if the user has since relabelled it into the other class or as Poor
    quality. The drop counts are part of the result and belong in any
    report that quotes these rows."""
    yardstick = yardstick or load_yardstick()
    ids = yardstick.detection_ids
    user = latest_labels(db, detection_ids=ids, sources=USER_SOURCES)
    crops = dict(db.query(Detection.id, Detection.crop_path).filter(Detection.id.in_(list(ids))).all())
    rows: dict[str, list[tuple[int, Path]]] = {c: [] for c in COHORTS}
    skips: Counter[str] = Counter()
    for cohort in COHORTS:
        for det_id in yardstick.cohorts[cohort]:
            if det_id not in crops:
                skips[f"{cohort}:row_deleted"] += 1
                continue
            if det_id in user and _truth_now(user[det_id][2]) != TRUTH[cohort]:
                skips[f"{cohort}:relabelled"] += 1
                continue
            crop = resolve_crop(crops[det_id], data_dir)
            if crop is None:
                skips[f"{cohort}:crop_missing"] += 1
                continue
            rows[cohort].append((det_id, crop))
    return rows, skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    freeze = sub.add_parser("freeze", help="Select the cohorts from the live database and write them out.")
    freeze.add_argument("--trained-dataset", type=Path, required=True,
                        help="The ImageFolder export the deployed model was trained on.")
    freeze.add_argument("--out", type=Path, default=FROZEN_PATH)
    freeze.add_argument("--force", action="store_true", help="Overwrite an existing yardstick.")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"{args.out} exists. Refreezing changes the rows every past result was "
              "measured on; pass --force if that is the intent.", file=sys.stderr)
        return 1
    trained = trained_detection_ids(args.trained_dataset)
    if not trained:
        print(f"No det_<id>.jpg files under {args.trained_dataset}", file=sys.stderr)
        return 1

    from db.session import SessionLocal
    from settings import settings

    db = SessionLocal()
    try:
        cohorts, excluded, heldout_days = select_cohorts(db, trained, ZoneInfo(settings.camera_timezone))
    finally:
        db.close()

    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deployed_model_trained_at": DEPLOYED_MODEL_TRAINED_AT.date().isoformat(),
        "trained_dataset_rows": len(trained),
        "day_split_from": DAY_SPLIT_FROM,
        "camera_timezone": settings.camera_timezone,
        "split_seed": SPLIT_SEED,
        "counts": {c: len(ids) for c, ids in cohorts.items()},
        "excluded": excluded,
        "heldout_days": heldout_days,
        "cohorts": cohorts,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"Wrote {args.out}")
    print(f"  {len(heldout_days)} capture days from {DAY_SPLIT_FROM} reserved for the yardstick")
    for c in COHORTS:
        print(f"  {c:<11} {len(cohorts[c]):>5}  left out: {excluded[c] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
