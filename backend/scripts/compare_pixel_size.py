"""Compare per-species bbox sizes before and after the camera move.

The camera was repositioned from ~12-14 ft away to ~9 ft on 2026-06-01
(~22:00 UTC). The geometric prediction was a 1.5× linear pixel gain
(2.25× area). This script measures the actual lift in pixels-per-bird,
stratified by species so we can distinguish "birds are bigger now"
from "species mix shifted toward bigger birds."

Usage:
    cd backend
    python scripts/compare_pixel_size.py                          # default cutoff
    python scripts/compare_pixel_size.py --cutoff 2026-06-01T22:00
    python scripts/compare_pixel_size.py --min-after 30           # only species
                                                                  # with ≥30
                                                                  # post-move
                                                                  # detections

Reports median bbox diagonal, median max(w,h), and median area for the
full population and per-species. Per-species rows only appear when both
sides have at least `--min-each` samples (default 10) — fewer than that
is too noisy to be worth reporting.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Correction, Detection, Species, Visit  # noqa: E402
from db.session import SessionLocal  # noqa: E402

# Default cutoff matches the actual move time inferred from
# data/calibration/heatmap_bg.jpg + size_priors.json mtimes (Jun 1
# 22:41 UTC). Override with --cutoff if the move was at a different
# time.
DEFAULT_CUTOFF = "2026-06-01T22:00"


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


def _stats(rows: list[tuple[float, float, float]]) -> tuple[int, float, float, float]:
    """rows = [(diag, max_wh, area), ...] → (n, med_diag, med_max_wh, med_area)."""
    if not rows:
        return (0, 0.0, 0.0, 0.0)
    return (
        len(rows),
        _median([r[0] for r in rows]),
        _median([r[1] for r in rows]),
        _median([r[2] for r in rows]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                    help="ISO timestamp; detections at/after this go in the 'after' bucket.")
    ap.add_argument("--since", default="2026-05-01",
                    help="Only consider detections from this date forward (limits 'before').")
    ap.add_argument("--min-each", type=int, default=10,
                    help="Min samples per side for a species to appear in the table.")
    ap.add_argument("--min-after", type=int, default=0,
                    help="Optionally require N post-move samples (default 0 = use --min-each).")
    ap.add_argument("--corrected-only", action="store_true",
                    help="Limit to user/LLM-corrected detections (ground truth only).")
    args = ap.parse_args()

    cutoff = datetime.fromisoformat(args.cutoff)
    since = datetime.fromisoformat(args.since)
    min_after = max(args.min_after, args.min_each)

    db = SessionLocal()
    try:
        # Resolve species name: prefer the latest Correction, fall back
        # to the classifier's Species.common_name. This matches the
        # "what label do we believe?" rule used elsewhere.
        q = (
            db.query(
                Detection.bbox,
                Visit.started_at,
                Correction.id.label("correction_id"),
                Species.common_name.label("classifier_name"),
            )
            .join(Visit, Visit.id == Detection.visit_id)
            .outerjoin(Species, Species.id == Detection.species_id)
            .outerjoin(Correction, Correction.detection_id == Detection.id)
            .filter(Visit.started_at >= since)
        )
        rows = q.all()

        # Pull correction labels in a second pass (Correction.species_id
        # joins to Species too; doing it inline above produces a
        # cartesian on multiple-correction detections, rare but messy).
        correction_names: dict[int, str] = {}
        if rows:
            corrected = (
                db.query(Correction.id, Species.common_name)
                .join(Species, Species.id == Correction.correct_species_id)
                .all()
            )
            correction_names = {cid: name for cid, name in corrected}

        # buckets[species]["before"|"after"] = [(diag, max_wh, area)]
        per_species: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(
            lambda: {"before": [], "after": []}
        )
        overall: dict[str, list[tuple[float, float, float]]] = {"before": [], "after": []}

        for bbox, started_at, correction_id, classifier_name in rows:
            if not bbox or len(bbox) < 4:
                continue
            label = correction_names.get(correction_id) or classifier_name
            if args.corrected_only and not correction_id:
                continue
            if not label:
                label = "(unlabeled)"
            w, h = float(bbox[2]), float(bbox[3])
            triple = (math.hypot(w, h), max(w, h), w * h)
            bucket = "after" if started_at >= cutoff else "before"
            per_species[label][bucket].append(triple)
            overall[bucket].append(triple)

        # --- overall ---
        n_b, med_d_b, med_m_b, med_a_b = _stats(overall["before"])
        n_a, med_d_a, med_m_a, med_a_a = _stats(overall["after"])
        print(f"Cutoff: {cutoff.isoformat()}  Since: {since.date()}  "
              f"Corrected-only: {args.corrected_only}\n")
        print(f"{'OVERALL':30s} | before n={n_b:>5d}  med_diag={med_d_b:6.1f}  "
              f"med_max(w,h)={med_m_b:6.1f}  med_area={med_a_b:8.0f}")
        print(f"{'':30s} |  after n={n_a:>5d}  med_diag={med_d_a:6.1f}  "
              f"med_max(w,h)={med_m_a:6.1f}  med_area={med_a_a:8.0f}")
        if n_b and n_a:
            print(f"{'':30s} |  lift                  "
                  f"diag×{med_d_a / med_d_b:.2f}  "
                  f"max(w,h)×{med_m_a / med_m_b:.2f}  "
                  f"area×{med_a_a / med_a_b:.2f}")
        print()

        # --- per species ---
        # Sort by post-move N descending so the most-confident species
        # rows are at the top.
        eligible = [
            (sp, buckets) for sp, buckets in per_species.items()
            if len(buckets["before"]) >= args.min_each
            and len(buckets["after"]) >= min_after
        ]
        eligible.sort(key=lambda kv: -len(kv[1]["after"]))

        if not eligible:
            print(f"No species had ≥{args.min_each} before and ≥{min_after} after. "
                  "Try a smaller --min-each or wait for more post-move data.")
            return 0

        header = (
            f"{'species':30s} | {'n_b':>5s} {'diag_b':>7s} {'mwh_b':>6s} | "
            f"{'n_a':>5s} {'diag_a':>7s} {'mwh_a':>6s} | {'×diag':>6s} {'×area':>6s}"
        )
        print(header)
        print("-" * len(header))
        for sp, buckets in eligible:
            n_b, d_b, m_b, a_b = _stats(buckets["before"])
            n_a, d_a, m_a, a_a = _stats(buckets["after"])
            lift_d = d_a / d_b if d_b else 0.0
            lift_a = a_a / a_b if a_b else 0.0
            print(f"{sp[:30]:30s} | {n_b:>5d} {d_b:>7.1f} {m_b:>6.1f} | "
                  f"{n_a:>5d} {d_a:>7.1f} {m_a:>6.1f} | "
                  f"{lift_d:>6.2f} {lift_a:>6.2f}")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
