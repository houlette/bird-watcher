"""Head-to-head: single-best bbox vs track-median bbox as size proxy.

Runs after Detection.track_bboxes has been populated for enough labeled
detections (production code in pipeline/process.py started populating
it in commit XXX). For each labeled real-species detection that has a
non-empty track_bboxes list, compute:

  - single_best:  max(w, h) of detection.bbox (the historical proxy)
  - track_median: median(max(w_i, h_i)) over all track frames

Then re-evaluate both proxies with the same Spearman ρ + pair-AUC
methodology as compare_proxies.py.

The question we want answered:
  Does smoothing across the track meaningfully shrink within-species
  variance and bump borderline-pair AUCs (e.g., Robin vs Mourning Dove)
  past the 0.75 "useful prior" threshold?

Decision criterion: track_median wins iff it improves the Spearman ρ
AND improves at least one previously-borderline pair from < 0.75 to ≥ 0.75.

Run this only after at least 20 labeled detections per species of
interest have track_bboxes populated. Earlier runs return a meaningless
comparison dominated by the existing (single-best) data.
"""
from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import SENTINEL_LABELS, Correction, Detection, Species  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402

from scripts.depth.species_sizes import expected_length_cm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare_track_smoothed")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_PER_SPECIES = 20


def _max_wh(bbox: list) -> float:
    _, _, w, h = bbox
    return float(max(w, h))


def _evaluate(name: str, by_species: dict[str, list[float]]) -> dict:
    big = {s: vs for s, vs in by_species.items() if len(vs) >= MIN_PER_SPECIES}
    medians = {s: float(np.median(vs)) for s, vs in big.items()}
    pts = [(s, expected_length_cm(s), m) for s, m in medians.items()
           if expected_length_cm(s) is not None]
    rho = None
    if len(pts) >= 3:
        rho, _ = spearmanr([p[1] for p in pts], [p[2] for p in pts])

    spp = sorted(big.keys())
    auc_pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(spp):
        for b in spp[i + 1:]:
            u, _ = mannwhitneyu(big[a], big[b], alternative="two-sided")
            auc = u / (len(big[a]) * len(big[b]))
            auc_pairs.append((a, b, max(auc, 1.0 - auc)))
    auc_pairs.sort(key=lambda r: -r[2])

    return {
        "name": name,
        "rho": rho,
        "medians": medians,
        "auc_pairs": auc_pairs,
        "n_pairs_075": sum(1 for _, _, a in auc_pairs if a >= 0.75),
    }


def main() -> int:
    init_db()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        rows = (
            db.query(Detection, Species.common_name)
            .join(Correction, Correction.detection_id == Detection.id)
            .join(Species, Correction.correct_species_id == Species.id)
            .filter(~Species.common_name.in_(SENTINEL_LABELS))
            .filter(Detection.track_bboxes.isnot(None))
            .all()
        )
        log.info("Labeled detections with track_bboxes populated: %d", len(rows))

        if not rows:
            log.warning(
                "No detections have track_bboxes yet — run after the "
                "instrumentation has been live long enough for new "
                "labels to accumulate (~1 week)."
            )
            return 1

        single_by: dict[str, list[float]] = defaultdict(list)
        median_by: dict[str, list[float]] = defaultdict(list)
        track_lens: list[int] = []
        for d, name in rows:
            if not d.bbox or len(d.bbox) != 4:
                continue
            single_by[name].append(_max_wh(d.bbox))
            tb = d.track_bboxes or []
            if not tb:
                continue
            track_lens.append(len(tb))
            per_frame = [_max_wh(b) for b in tb if b and len(b) == 4]
            if not per_frame:
                continue
            median_by[name].append(float(np.median(per_frame)))

        if track_lens:
            log.info(
                "Track lengths — mean %.1f, median %d, max %d frames",
                float(np.mean(track_lens)), int(np.median(track_lens)), max(track_lens),
            )

        results = [
            _evaluate("single_best_max_wh", single_by),
            _evaluate("track_median_max_wh", median_by),
        ]

        out = RESULTS_DIR / "track_smoothed_comparison.md"
        lines = ["# Track-smoothed bbox vs single-best comparison\n"]
        lines.append("| Proxy | Spearman ρ | Pairs ≥ 0.75 AUC |")
        lines.append("|---|---:|---:|")
        for r in results:
            rho = f"{r['rho']:.3f}" if r["rho"] is not None else "—"
            lines.append(f"| `{r['name']}` | {rho} | {r['n_pairs_075']} |")
        for r in results:
            lines.append(f"\n## `{r['name']}` — pair AUCs\n")
            lines.append("| A | B | AUC | Δ Cornell (cm) |")
            lines.append("|---|---|---:|---:|")
            for a, b, auc in r["auc_pairs"]:
                la, lb = expected_length_cm(a), expected_length_cm(b)
                delta = f"{abs(la - lb):.1f}" if (la is not None and lb is not None) else "—"
                star = " ⭐" if auc >= 0.75 else ""
                lines.append(f"| {a} | {b} | {auc:.3f}{star} | {delta} |")
        out.write_text("\n".join(lines))
        log.info("Wrote %s", out)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
