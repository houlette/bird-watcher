"""A/B-style eval of the binary bird/NAB post-filter against the locked
Corrections set.

Independent of the species classifier (see benchmark_classifiers.py for
that A/B). For every Detection with a Correction we have a binary
ground truth:

    Corrected to NOT_A_BIRD_LABEL   → ground-truth NAB (positive — we
                                       WANT the filter to fire)
    Corrected to a real species      → ground-truth BIRD (negative — we
                                       want the filter to NOT fire)
    Corrected to UNKNOWN_BIRD_LABEL  → ambiguous, dropped

Sentinel "Unknown bird" rows are positive confirmations that something
IS a bird, so they're treated as negatives just like real species.

For every eligible crop the script runs `binary_filter.nab_probability`
(same code path production uses), then sweeps thresholds to report:

  - Precision / recall / F1 at every 0.05 step
  - ROC AUC + PR AUC
  - The operating-point row (settings.bird_binary_nab_threshold) and
    what production confusion looks like at that threshold
  - Per-correction-source breakdown so we can see whether the filter's
    performance differs on user-labeled vs LLM-labeled negatives

Designed to be re-runnable after every binary-filter retrain — the
Corrections set is the frozen yardstick. Same harness style as
benchmark_classifiers.py (--limit, --json, idempotent).

Usage:
    docker compose exec api python scripts/eval_binary_filter.py
    docker compose exec api python scripts/eval_binary_filter.py --limit 100
    docker compose exec api python scripts/eval_binary_filter.py --json > eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import (  # noqa: E402
    NOT_A_BIRD_LABEL,
    SENTINEL_LABELS,
    UNKNOWN_BIRD_LABEL,
    Correction,
    Detection,
    Species,
)
from db.session import SessionLocal, init_db  # noqa: E402
from pipeline import binary_filter  # noqa: E402
from settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_binary_filter")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_samples(db, limit: int) -> list[tuple[int, str, str | None, str]]:
    """Return [(detection_id, crop_path, correction_source, ground_truth)].

    ground_truth ∈ {"nab", "bird"}. UNKNOWN_BIRD is treated as BIRD —
    the user explicitly confirmed it as a bird, just without a species
    commitment. That's still a valid negative for the filter.
    """
    rows = (
        db.query(Detection.id, Detection.crop_path, Correction.source, Species.common_name)
        .join(Correction, Correction.detection_id == Detection.id)
        .join(Species, Correction.correct_species_id == Species.id)
        .order_by(Detection.id.asc())
        .all()
    )
    out = []
    for det_id, crop_path, source, sp_name in rows:
        if not crop_path:
            continue
        if sp_name == NOT_A_BIRD_LABEL:
            gt = "nab"
        elif sp_name == UNKNOWN_BIRD_LABEL or sp_name not in SENTINEL_LABELS:
            gt = "bird"
        else:
            continue
        out.append((det_id, crop_path, source, gt))
        if limit and len(out) >= limit:
            break
    return out


def _resolve_crop(crop_path: str) -> Path | None:
    """Crop paths in DB may be absolute or relative; map both to the
    actual file on disk under data/."""
    p = Path(crop_path)
    if p.exists():
        return p
    # Stored values look like "crops/<visit>/<track>.jpg" or similar —
    # join under DATA_DIR.
    rel = DATA_DIR / crop_path.lstrip("/")
    if rel.exists():
        return rel
    # One more fallback: just the filename under data/crops/.
    fallback = DATA_DIR / "crops" / p.name
    if fallback.exists():
        return fallback
    return None


def _score_samples(samples: list[tuple[int, str, str | None, str]]):
    """Run nab_probability on each crop. Returns parallel arrays of
    (y_true, y_score, source) and a counter of skip reasons."""
    y_true: list[int] = []
    y_score: list[float] = []
    sources: list[str | None] = []
    skips: Counter[str] = Counter()

    if not binary_filter.is_enabled():
        log.error("BIRD_BINARY_FILTER_MODEL not set — nothing to evaluate.")
        return y_true, y_score, sources, skips

    binary_filter.warmup()
    t0 = time.time()
    for i, (det_id, crop_path, source, gt) in enumerate(samples, start=1):
        path = _resolve_crop(crop_path)
        if path is None:
            skips["missing_file"] += 1
            continue
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            skips["unreadable"] += 1
            continue
        p_nab = binary_filter.nab_probability(img)
        if p_nab is None:
            skips["filter_returned_none"] += 1
            continue
        y_true.append(1 if gt == "nab" else 0)
        y_score.append(float(p_nab))
        sources.append(source)
        if i % 250 == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            log.info("Scored %d / %d  (%.1f/s)", i, len(samples), rate)

    log.info("Done scoring %d crops in %.1fs", len(y_score), time.time() - t0)
    return y_true, y_score, sources, skips


def _confusion(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """Confusion matrix at the given threshold (positive = NAB)."""
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else None
    )
    # The "bird-kept rate" is the user-facing thing: of real birds, what
    # fraction does the filter NOT throw away? Same as 1 - FPR.
    bird_kept = tn / (tn + fp) if (tn + fp) else None
    return {
        "threshold": round(threshold, 3),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bird_kept_rate": bird_kept,
    }


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


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Trapezoidal PR AUC over the sorted score grid."""
    if (y_true == 1).sum() == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / max(int(y_sorted.sum()), 1)
    # Prepend (recall=0, precision=precisions[0]) so the curve starts on
    # the y-axis.
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0]], precisions])
    return float(np.trapz(precisions, recalls))


def _per_source_breakdown(
    y_true: np.ndarray, y_score: np.ndarray, sources: list[str | None], threshold: float
) -> dict:
    """Confusion matrix at the operating threshold, sliced by Correction.source.
    NULL source is the user-via-UI default; "llm-claude*" is LLM-generated.
    Helps spot whether the filter overfits to LLM labels."""
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # [tp,fp,tn,fn]
    for yt, ys, src in zip(y_true, y_score, sources):
        bucket = "user" if src is None else (
            "llm" if src.startswith("llm-claude") else src
        )
        b = buckets[bucket]
        pred = 1 if ys >= threshold else 0
        if pred == 1 and yt == 1: b[0] += 1
        elif pred == 1 and yt == 0: b[1] += 1
        elif pred == 0 and yt == 0: b[2] += 1
        else: b[3] += 1

    out = {}
    for src, (tp, fp, tn, fn) in buckets.items():
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        out[src] = {
            "n": tp + fp + tn + fn,
            "n_nab": tp + fn, "n_bird": tn + fp,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision,
            "recall": recall,
            "bird_kept_rate": tn / (tn + fp) if (tn + fp) else None,
        }
    return out


def _format_row(r: dict) -> str:
    def f(x, pct=False):
        if x is None: return "  —  "
        return f"{x * 100:5.1f}%" if pct else f"{x:5.3f}"
    return (
        f"  τ={r['threshold']:.2f}  "
        f"P={f(r['precision'], True)}  "
        f"R={f(r['recall'], True)}  "
        f"F1={f(r['f1'])}  "
        f"birds-kept={f(r['bird_kept_rate'], True)}  "
        f"[tp={r['tp']:>4} fp={r['fp']:>4} tn={r['tn']:>4} fn={r['fn']:>4}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap eligible samples (0 = all). Useful for smoke tests.")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the formatted table.")
    args = ap.parse_args()

    init_db()
    if not binary_filter.is_enabled():
        log.error(
            "Binary filter is disabled (BIRD_BINARY_FILTER_MODEL unset or "
            "model missing). Configure it and re-run."
        )
        return 2

    db = SessionLocal()
    try:
        samples = _load_samples(db, args.limit)
    finally:
        db.close()

    n_nab = sum(1 for _, _, _, gt in samples if gt == "nab")
    n_bird = len(samples) - n_nab
    log.info("Eligible samples: %d  (NAB=%d, bird=%d)", len(samples), n_nab, n_bird)
    if not samples:
        log.error("No samples found — does the DB have Corrections?")
        return 1

    y_true_list, y_score_list, sources, skips = _score_samples(samples)
    y_true = np.array(y_true_list, dtype=np.int64)
    y_score = np.array(y_score_list, dtype=np.float64)

    if len(y_true) == 0:
        log.error("No crops scored — every sample skipped. Skips: %s", dict(skips))
        return 1

    sweep = [_confusion(y_true, y_score, t) for t in np.arange(0.05, 1.0, 0.05)]
    operating = _confusion(y_true, y_score, settings.bird_binary_nab_threshold)
    # Argmax-F1 best threshold over the sweep + the production tau.
    candidates = sweep + [operating]
    best = max(
        (c for c in candidates if c["f1"] is not None),
        key=lambda c: c["f1"],
        default=None,
    )
    roc_auc = _roc_auc(y_true, y_score)
    pr_auc = _pr_auc(y_true, y_score)
    per_source = _per_source_breakdown(
        y_true, y_score, sources, settings.bird_binary_nab_threshold
    )

    result = {
        "model": settings.bird_binary_filter_model,
        "operating_threshold": settings.bird_binary_nab_threshold,
        "n_total": int(len(y_true)),
        "n_nab": int((y_true == 1).sum()),
        "n_bird": int((y_true == 0).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "operating_point": operating,
        "best_f1_point": best,
        "sweep": sweep,
        "per_source": per_source,
        "skips": dict(skips),
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print()
    print(f"Binary bird/NAB filter eval")
    print(f"  model:     {result['model']}")
    print(f"  samples:   {result['n_total']:,}  (NAB={result['n_nab']:,}  bird={result['n_bird']:,})")
    print(f"  ROC AUC:   {roc_auc:.3f}" if roc_auc is not None else "  ROC AUC:   —")
    print(f"  PR  AUC:   {pr_auc:.3f}" if pr_auc is not None else "  PR  AUC:   —")
    if skips:
        print(f"  skipped:   {dict(skips)}")
    print()
    print("Threshold sweep (positive = NAB):")
    for row in sweep:
        print(_format_row(row))
    print()
    print(f"Operating point  (production τ = {result['operating_threshold']}):")
    print(_format_row(operating))
    if best is not None:
        print(f"\nBest-F1 point in sweep:")
        print(_format_row(best))
    print()
    print(f"Per Correction.source @ τ = {result['operating_threshold']}:")
    for src, b in per_source.items():
        precision = "  —  " if b["precision"] is None else f"{b['precision']*100:5.1f}%"
        recall = "  —  " if b["recall"] is None else f"{b['recall']*100:5.1f}%"
        kept = "  —  " if b["bird_kept_rate"] is None else f"{b['bird_kept_rate']*100:5.1f}%"
        print(
            f"  {src:<10} n={b['n']:>5}  (nab={b['n_nab']:>4} bird={b['n_bird']:>4})  "
            f"P={precision}  R={recall}  birds-kept={kept}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
