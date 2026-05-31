"""Compare candidate size proxies head-to-head.

After the first validation pass we learned monocular depth doesn't
help (Spearman ρ went UP from 0.40 to 0.80 when we dropped the depth
correction). This script compares the four most-promising depth-free
variants on the same labeled-detection set and writes a head-to-head
report.

Variants:
  1. bbox_diag              baseline — sqrt(w²+h²)
  2. max_wh                 longest bbox axis; aspect-ratio robust
  3. sqrt_area              sqrt(w·h); pose-noise robust
  4. y_corrected_diag       bbox_diag normalized by a fitted
                            "bbox-vs-y" curve, where y is the foot
                            position in the frame. Captures the
                            camera-look-down perspective without
                            invoking metric depth at all.

For each variant we report:
  - Spearman ρ between species-median proxy and Cornell length
  - All pair AUCs (size-only separability)
  - The species-medians table

Output → `backend/scripts/depth/results/proxies_comparison.md`.
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
log = logging.getLogger("compare_proxies")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_PER_SPECIES = 20

# Y-correction is fitted on a single well-sampled species. Mourning Dove
# (n=164) is the densest sample across the full y range; we assume all
# Doves are the same true size, so any bbox_diag vs y trend within
# Doves is pure perspective. We use a degree-2 polynomial — linear is
# too rigid; degree-3 starts to overfit on a single species.
Y_FIT_SPECIES = "Mourning Dove"
Y_FIT_DEGREE = 2


# ─── Proxies ──────────────────────────────────────────────────────────────


def bbox_diag(b: tuple) -> float:
    _, _, w, h = b
    return float(np.sqrt(w * w + h * h))


def max_wh(b: tuple) -> float:
    _, _, w, h = b
    return float(max(w, h))


def sqrt_area(b: tuple) -> float:
    _, _, w, h = b
    return float(np.sqrt(w * h))


# ─── Y-correction fit (perspective-normalized bbox_diag) ──────────────────


def _fit_y_correction(rows: list[tuple[Detection, str]]) -> callable:
    """Fit bbox_diag(y) on the densest-sampled species (assumed-constant
    true size); return a callable `y_correction(y) → scale_factor` such
    that multiplying a raw bbox_diag by it normalizes for perspective."""
    pts: list[tuple[float, float]] = []
    for d, n in rows:
        if n != Y_FIT_SPECIES or not d.bbox or len(d.bbox) != 4:
            continue
        bx, by, bw, bh = d.bbox
        foot_y = by + bh
        pts.append((foot_y, bbox_diag(d.bbox)))
    if len(pts) < 30:
        log.warning(
            "Only %d %s samples; falling back to identity correction",
            len(pts), Y_FIT_SPECIES,
        )
        return lambda y: 1.0
    ys = np.array([p[0] for p in pts])
    diags = np.array([p[1] for p in pts])
    coef = np.polyfit(ys, diags, Y_FIT_DEGREE)
    median_diag = float(np.median(diags))

    def correction(y: float) -> float:
        # Multiply each detection's bbox_diag by `median_diag /
        # predicted_diag_at_y`. A bird high in the frame (small y,
        # therefore small predicted_diag) gets boosted; a bird at the
        # well-sampled feeder y gets ~1.0.
        predicted = float(np.polyval(coef, y))
        if predicted <= 0:
            return 1.0
        return median_diag / predicted

    log.info(
        "Fitted y-correction on %d %s samples: coef=%s, ref_diag=%.1f",
        len(pts), Y_FIT_SPECIES, coef.tolist(), median_diag,
    )
    return correction


def y_corrected_diag(b: tuple, correction: callable) -> float:
    _, by, _, bh = b
    foot_y = by + bh
    return bbox_diag(b) * correction(foot_y)


# ─── Per-variant evaluation ──────────────────────────────────────────────


def _evaluate(name: str, by_species: dict[str, list[float]]) -> dict:
    """Spearman ρ + sorted pair AUCs for a single proxy."""
    big = {s: vs for s, vs in by_species.items() if len(vs) >= MIN_PER_SPECIES}
    medians = {s: float(np.median(vs)) for s, vs in big.items()}

    # Spearman vs Cornell
    pts = [(s, expected_length_cm(s), m) for s, m in medians.items()
           if expected_length_cm(s) is not None]
    rho = None
    if len(pts) >= 3:
        rho, _ = spearmanr([p[1] for p in pts], [p[2] for p in pts])

    # Pair AUCs
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


# ─── Report ──────────────────────────────────────────────────────────────


def _write_report(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "proxies_comparison.md"
    lines = []
    lines.append("# Size-proxy variant comparison\n")
    lines.append(
        "Head-to-head on the same labeled-detection set. All four "
        "variants are depth-free.\n"
    )

    # Headline table
    lines.append("## Verdict table\n")
    lines.append("| Proxy | Spearman ρ vs Cornell | Pairs ≥ 0.75 AUC |")
    lines.append("|---|---:|---:|")
    for r in results:
        rho = f"{r['rho']:.3f}" if r["rho"] is not None else "—"
        lines.append(f"| `{r['name']}` | {rho} | {r['n_pairs_075']} |")

    # Per-species medians
    lines.append("\n## Per-species medians by proxy\n")
    all_species = sorted(
        {s for r in results for s in r["medians"]},
        key=lambda s: expected_length_cm(s) or 0.0,
    )
    header = ["Species (Cornell)"] + [f"`{r['name']}`" for r in results]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for sp in all_species:
        cl = expected_length_cm(sp)
        row = [f"{sp} ({cl:.0f} cm)" if cl else sp]
        for r in results:
            m = r["medians"].get(sp)
            row.append(f"{m:.1f}" if m else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Per-variant pair tables
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
    return out


# ─── Driver ──────────────────────────────────────────────────────────────


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        rows = (
            db.query(Detection, Species.common_name)
            .join(Correction, Correction.detection_id == Detection.id)
            .join(Species, Correction.correct_species_id == Species.id)
            .filter(~Species.common_name.in_(SENTINEL_LABELS))
            .all()
        )
        log.info("Found %d labeled real-species detections", len(rows))

        # Fit the y-correction once, on the densest species.
        y_correction = _fit_y_correction(rows)

        proxies = [
            ("bbox_diag", lambda b: bbox_diag(tuple(b))),
            ("max_wh", lambda b: max_wh(tuple(b))),
            ("sqrt_area", lambda b: sqrt_area(tuple(b))),
            ("y_corrected_diag", lambda b: y_corrected_diag(tuple(b), y_correction)),
        ]

        results = []
        for name, fn in proxies:
            by: dict[str, list[float]] = defaultdict(list)
            for d, n in rows:
                if not d.bbox or len(d.bbox) != 4:
                    continue
                by[n].append(fn(d.bbox))
            results.append(_evaluate(name, by))

        _write_report(results)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
