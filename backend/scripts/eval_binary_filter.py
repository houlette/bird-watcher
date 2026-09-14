"""Evaluate the binary bird/not-a-bird filter on the held-out yardstick.

Every number this prints is measured on the three frozen cohorts in
scripts/train/heldout.py, which the training export refuses to emit. The
previous version scored every corrected detection, about 96% of which the
deployed model had trained on; it also counted a detection once per
correction rather than once, and took labels written by the pipeline's own
recurrence and scene-mask filters as ground truth.

Three cautions travel with every result:

  - The two bird cohorts are selected in opposite directions, so the kill
    rate on each is not the kill rate on birds in general. They are
    printed side by side, and neither should be quoted alone.
  - The junk cohort is the deployed model's own misses, so the deployed
    model's catch rate there is a floor by construction. A retrain's catch
    rate on it means something only next to the deployed model's.
  - Crops are the polished JPEGs on disk. Production scores the fused raw
    crop, and at 0.75 the two disagree on about half of the rows the user
    later called junk (see the nab_p_* columns on Detection). Rates here do
    not transfer to live behaviour until training, eval and serving use
    one representation.

Paired comparison of two models: run once per model and compare the
per-detection `scores` in the JSON output.

Usage:
    docker compose exec api python scripts/eval_binary_filter.py
    docker compose exec api python scripts/eval_binary_filter.py --limit 50
    docker compose exec -e BIRD_BINARY_FILTER_MODEL=/app/data/models/candidate \\
        api python scripts/eval_binary_filter.py --json > candidate.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.session import SessionLocal, init_db  # noqa: E402
from pipeline import binary_filter  # noqa: E402
from scripts.train import heldout  # noqa: E402
from settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_binary_filter")

MEASURED_ON = (
    "held-out yardstick: no row, and no other crop from its visit, is in the "
    "deployed model's training set or in any export made since the freeze"
)
CROPS = "polished JPEG on disk; production scores the fused raw crop, so these rates do not transfer to live behaviour"

COHORT_NOTES = {
    heldout.HARD_BIRDS: "killed; selected towards hard crops",
    heldout.EASY_BIRDS: "killed; selected towards easy crops",
    heldout.JUNK: "caught; the deployed model's misses, a floor for it",
}


def _score(rows: dict[str, list[tuple[int, Path]]]) -> tuple[dict[str, dict[int, float]], Counter]:
    scores: dict[str, dict[int, float]] = {c: {} for c in rows}
    skips: Counter[str] = Counter()
    binary_filter.warmup()
    total = sum(len(r) for r in rows.values())
    done = 0
    t0 = time.time()
    for cohort, items in rows.items():
        for det_id, crop in items:
            done += 1
            img = cv2.imread(str(crop))
            if img is None or img.size == 0:
                skips[f"{cohort}:unreadable"] += 1
                continue
            p_nab = binary_filter.nab_probability(img)
            if p_nab is None:
                skips[f"{cohort}:filter_returned_none"] += 1
                continue
            scores[cohort][det_id] = float(p_nab)
            if done % 250 == 0:
                log.info("Scored %d / %d  (%.1f/s)", done, total, done / max(time.time() - t0, 1e-6))
    log.info("Done scoring %d crops in %.1fs", sum(len(s) for s in scores.values()), time.time() - t0)
    return scores, skips


def _flagged(values: np.ndarray, threshold: float) -> float | None:
    return float((values >= threshold).mean()) if len(values) else None


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """ROC AUC by the rank-sum identity — no sklearn dependency.
    AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg).
    """
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    # Average ranks for ties so AUC is well-defined under repeated scores.
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    rank = 1
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2
        ranks[order[i:j + 1]] = avg_rank
        rank += (j - i) + 1
        i = j + 1
    sum_pos_ranks = float(ranks[y_true == 1].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _auc_vs_junk(birds: np.ndarray, junk: np.ndarray) -> float | None:
    y_true = np.concatenate([np.zeros(len(birds)), np.ones(len(junk))]).astype(np.int64)
    return _roc_auc(y_true, np.concatenate([birds, junk]))


def summarise(scores: dict[str, dict[int, float]], threshold: float) -> dict:
    arrays = {c: np.array(list(s.values()), dtype=np.float64) for c, s in scores.items()}
    cohorts = {}
    for c, a in arrays.items():
        cohorts[c] = {
            "n": int(len(a)),
            "flagged_at_threshold": int((a >= threshold).sum()),
            "flagged_share": _flagged(a, threshold),
            "median_p_nab": float(np.median(a)) if len(a) else None,
        }
    sweep = [
        {"threshold": round(float(t), 2), **{c: _flagged(a, t) for c, a in arrays.items()}}
        for t in np.arange(0.05, 1.0, 0.05)
    ]
    return {
        "cohorts": cohorts,
        "roc_auc": {
            "hard_birds_vs_junk": _auc_vs_junk(arrays[heldout.HARD_BIRDS], arrays[heldout.JUNK]),
            "easy_birds_vs_junk": _auc_vs_junk(arrays[heldout.EASY_BIRDS], arrays[heldout.JUNK]),
        },
        "sweep": sweep,
    }


def _pct(x: float | None) -> str:
    return "   —  " if x is None else f"{x * 100:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Score a random sample of at most this many rows per cohort (0 = all).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for the --limit sample.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON, with per-detection scores, instead of the table.")
    args = ap.parse_args()

    init_db()
    if not binary_filter.is_enabled():
        log.error(
            "Binary filter is disabled (BIRD_BINARY_FILTER_MODEL unset or "
            "model missing). Configure it and re-run."
        )
        return 2

    yardstick = heldout.load_yardstick()
    db = SessionLocal()
    try:
        rows, skips = heldout.cohort_rows(db, yardstick)
    finally:
        db.close()

    if args.limit:
        rng = random.Random(args.seed)
        rows = {
            c: sorted(rng.sample(items, args.limit)) if len(items) > args.limit else items
            for c, items in rows.items()
        }
    log.info("Rows to score: %s", {c: len(r) for c, r in rows.items()})

    scores, score_skips = _score(rows)
    skips.update(score_skips)
    threshold = settings.bird_binary_nab_threshold
    summary = summarise(scores, threshold)

    result = {
        "model": settings.bird_binary_filter_model,
        "operating_threshold": threshold,
        "measured_on": MEASURED_ON,
        "yardstick_frozen_at": yardstick.frozen_at,
        "crops": CROPS,
        "limit_per_cohort": args.limit or None,
        "limit_seed": args.seed if args.limit else None,
        **summary,
        "skips": dict(skips),
        "scores": {c: {str(k): v for k, v in s.items()} for c, s in scores.items()},
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print()
    print("Binary bird/not-a-bird filter")
    print(f"  model:        {result['model']}")
    print(f"  threshold:    {threshold} (production)")
    print(f"  measured on:  {MEASURED_ON}")
    print(f"                (frozen {yardstick.frozen_at})")
    print(f"  crops:        {CROPS}")
    if args.limit:
        print(f"  sample:       at most {args.limit} per cohort, seed {args.seed}; not the full yardstick")
    if skips:
        print(f"  not scored:   {dict(sorted(skips.items()))}")
    print()
    print(f"  {'cohort':<11} {'n':>5}  {'flagged at ' + str(threshold):>16}  {'median P(NAB)':>13}")
    for c in heldout.COHORTS:
        b = summary["cohorts"][c]
        median = "   —" if b["median_p_nab"] is None else f"{b['median_p_nab']:.3f}"
        print(f"  {c:<11} {b['n']:>5}  {_pct(b['flagged_share'])} ({b['flagged_at_threshold']:>4})  "
              f"{median:>13}   {COHORT_NOTES[c]}")
    auc = summary["roc_auc"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    print()
    print(f"  ROC AUC: hard birds vs junk {fmt(auc['hard_birds_vs_junk'])}, "
          f"easy birds vs junk {fmt(auc['easy_birds_vs_junk'])}")
    print()
    print("Share of each cohort with P(NAB) at or above the threshold:")
    print(f"  {'threshold':>9}  {'hard_birds':>10}  {'easy_birds':>10}  {'junk':>7}")
    for row in summary["sweep"]:
        print(f"  {row['threshold']:>9.2f}  {_pct(row[heldout.HARD_BIRDS]):>10}  "
              f"{_pct(row[heldout.EASY_BIRDS]):>10}  {_pct(row[heldout.JUNK]):>7}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
