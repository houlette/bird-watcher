"""Does the depth-derived size proxy actually distinguish species?

Run after `build_depth_map.py`. Joins every Detection with a
real-species Correction to the precomputed depth map, computes a
scale-free size proxy, and tests whether species separate by size.

Decision criteria (codified in REPORT.md):

  1. Spearman ρ ≥ 0.6 between species-median proxy and Cornell-published
     mean length — "the proxy gets ordering broadly right."
  2. At least 3 confusable pairs achieve size-only AUC ≥ 0.75 — "the
     proxy rescues real mis-classifications."

Outputs (all under backend/scripts/depth/results/):
  - REPORT.md            top-line verdict + tables
  - size_by_species.png  violin plot of proxy by species
  - rank_check.png       proxy vs Cornell scatter
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow `python scripts/depth/validate_size_signal.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import SENTINEL_LABELS, Correction, Detection, Species, Visit  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402

from scripts.depth.size_from_bbox import size_proxy  # noqa: E402
from scripts.depth.species_sizes import expected_length_cm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("validate_size_signal")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEPTH_NPZ = DATA_DIR / "calibration" / "depth_map.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Per-species minimums for the various analyses. Below these the
# numbers are too noisy to mean anything.
MIN_PER_SPECIES_FOR_VIOLIN = 10
MIN_PER_SPECIES_FOR_PAIRWISE = 20
MIN_PER_SPECIES_FOR_RANK = 20


def _load_depth() -> np.ndarray:
    if not DEPTH_NPZ.exists():
        raise SystemExit(
            f"Depth map missing: {DEPTH_NPZ}\n"
            "Run `python scripts/depth/build_depth_map.py` first."
        )
    with np.load(DEPTH_NPZ) as data:
        return data["depth"]


def _gather_labeled_detections(db) -> list[tuple[Detection, str]]:
    """Every (Detection, corrected_species_name) where the user said
    'real species' (not NAB / Unknown). Pattern lifted from
    backend/pipeline/stats.py's correction queries."""
    rows = (
        db.query(Detection, Species.common_name)
        .join(Correction, Correction.detection_id == Detection.id)
        .join(Species, Correction.correct_species_id == Species.id)
        .filter(~Species.common_name.in_(SENTINEL_LABELS))
        .all()
    )
    return [(d, name) for d, name in rows]


def _compute_proxies(
    detections: list[tuple[Detection, str]], depth: np.ndarray
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Return (proxy values per species, rejection counts per status)."""
    by_species: dict[str, list[float]] = defaultdict(list)
    rejected: dict[str, int] = defaultdict(int)
    for det, name in detections:
        if not det.bbox or len(det.bbox) != 4:
            rejected["bad_bbox"] += 1
            continue
        proxy, status = size_proxy(tuple(det.bbox), depth)
        if proxy is None:
            rejected[status] += 1
            continue
        by_species[name].append(proxy)
    return by_species, dict(rejected)


# ─── Plots ────────────────────────────────────────────────────────────────


def _save_violin_plot(by_species: dict[str, list[float]], out: Path) -> None:
    """Violin plot of proxy by species, ordered by Cornell length when
    known. Species without a Cornell entry are appended at the end."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = [(s, vs) for s, vs in by_species.items() if len(vs) >= MIN_PER_SPECIES_FOR_VIOLIN]
    # Stable order: known-length species ascending by length, then unknown alphabetical.
    with_len = [(s, vs, expected_length_cm(s)) for s, vs in eligible]
    known = sorted([t for t in with_len if t[2] is not None], key=lambda t: t[2])
    unknown = sorted([t for t in with_len if t[2] is None], key=lambda t: t[0])
    ordered = known + unknown
    if not ordered:
        log.warning("No species pass MIN_PER_SPECIES_FOR_VIOLIN=%d", MIN_PER_SPECIES_FOR_VIOLIN)
        return

    fig, ax = plt.subplots(figsize=(max(8.0, len(ordered) * 0.45), 6.0))
    data = [t[1] for t in ordered]
    labels = [f"{t[0]}\n({t[2]:.0f} cm)" if t[2] else t[0] for t in ordered]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("#2f5d3d")
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("size_proxy  (bbox diagonal × foot depth, m·px)")
    ax.set_title(
        "Depth-derived size proxy by labeled species\n"
        f"Ordered left→right by Cornell-published length where known. n≥{MIN_PER_SPECIES_FOR_VIOLIN}."
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.info("Wrote %s", out)


def _save_rank_check(species_medians: dict[str, float], out: Path) -> tuple[float | None, list[tuple[str, float, float]]]:
    """Scatter of (Cornell length, proxy median) per species. Returns
    (Spearman ρ, ranked list) for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    pts: list[tuple[str, float, float]] = []
    for sp, med in species_medians.items():
        cl = expected_length_cm(sp)
        if cl is None:
            continue
        pts.append((sp, cl, med))
    if len(pts) < 3:
        log.warning("Need ≥3 species with Cornell entries; got %d", len(pts))
        return None, []

    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    rho, _ = spearmanr(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=40, color="#2f5d3d", alpha=0.8)
    for sp, cl, med in pts:
        ax.annotate(sp, (cl, med), fontsize=7, alpha=0.75, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Cornell-published length (cm)")
    ax.set_ylabel("Median size_proxy (m·px)")
    ax.set_title(f"Proxy vs. real size — Spearman ρ = {rho:.3f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    log.info("Wrote %s", out)
    return float(rho), pts


# ─── Pairwise separability ────────────────────────────────────────────────


def _pairwise_auc(a: list[float], b: list[float]) -> float:
    """Probability that a random sample from `a` exceeds one from `b`,
    i.e. AUC of "is in a?" classifier using size alone. Mann-Whitney U
    rescaled. Symmetric around 0.5; we report max(auc, 1-auc) so the
    direction of separation doesn't matter for the threshold check."""
    from scipy.stats import mannwhitneyu

    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b))
    return max(auc, 1.0 - auc)


def _compute_pairwise_table(by_species: dict[str, list[float]]) -> list[dict]:
    eligible = sorted(
        s for s, vs in by_species.items() if len(vs) >= MIN_PER_SPECIES_FOR_PAIRWISE
    )
    out: list[dict] = []
    for i, s1 in enumerate(eligible):
        for s2 in eligible[i + 1:]:
            auc = _pairwise_auc(by_species[s1], by_species[s2])
            l1, l2 = expected_length_cm(s1), expected_length_cm(s2)
            delta = abs(l1 - l2) if (l1 is not None and l2 is not None) else None
            out.append(
                {
                    "a": s1,
                    "b": s2,
                    "n_a": len(by_species[s1]),
                    "n_b": len(by_species[s2]),
                    "median_a": float(np.median(by_species[s1])),
                    "median_b": float(np.median(by_species[s2])),
                    "auc": float(auc),
                    "real_delta_cm": delta,
                }
            )
    out.sort(key=lambda r: -r["auc"])
    return out


# ─── Report ──────────────────────────────────────────────────────────────


def _write_report(
    by_species: dict[str, list[float]],
    rejected: dict[str, int],
    spearman_rho: float | None,
    rank_pts: list[tuple[str, float, float]],
    pairwise: list[dict],
    depth_meta: dict,
) -> Path:
    out = RESULTS_DIR / "REPORT.md"
    total_dets = sum(len(v) for v in by_species.values()) + sum(rejected.values())

    # Decision criteria
    rho_pass = spearman_rho is not None and spearman_rho >= 0.6
    n_high_auc = sum(1 for p in pairwise if p["auc"] >= 0.75)
    auc_pass = n_high_auc >= 3
    verdict = (
        "**SHIP**: build a follow-up plan for pipeline integration."
        if (rho_pass and auc_pass)
        else (
            "**WEAK**: directional signal present, but separability isn't there yet — revisit with more data."
            if (rho_pass and not auc_pass)
            else "**STOP**: depth-derived size doesn't separate species. Shelve the idea or invest in true LiDAR."
        )
    )

    lines = []
    lines.append("# Depth-prior validation report\n")
    lines.append(f"_Generated by `scripts/depth/validate_size_signal.py`_\n")
    lines.append(f"\n## Verdict: {verdict}\n")
    lines.append("\n### Decision criteria")
    lines.append(f"- Spearman ρ (proxy vs Cornell): **{spearman_rho:.3f}**  → "
                 f"{'✓ pass' if rho_pass else '✗ fail'} (threshold ≥ 0.60)" if spearman_rho is not None
                 else "- Spearman ρ: insufficient species with Cornell lengths.")
    lines.append(f"- Pairs with AUC ≥ 0.75: **{n_high_auc}**  → "
                 f"{'✓ pass' if auc_pass else '✗ fail'} (threshold ≥ 3)\n")

    # Data summary
    lines.append("\n## Dataset\n")
    lines.append(f"- Labeled real-species detections evaluated: **{total_dets}**")
    lines.append(f"- Got a size proxy: {sum(len(v) for v in by_species.values())}")
    if rejected:
        rej_summary = ", ".join(f"{k}={v}" for k, v in sorted(rejected.items(), key=lambda kv: -kv[1]))
        lines.append(f"- Rejected by reason: {rej_summary}")
    lines.append(f"- Species with ≥ {MIN_PER_SPECIES_FOR_VIOLIN} proxies: "
                 f"{sum(1 for v in by_species.values() if len(v) >= MIN_PER_SPECIES_FOR_VIOLIN)}")
    lines.append(f"\n### Depth map\n")
    lines.append(f"- Model: `{depth_meta.get('model', '?')}`")
    lines.append(f"- Source frame: `{depth_meta.get('source_frame', '?')}`")
    lines.append(f"- Image size: {depth_meta.get('image_size')}")
    lines.append(f"- Depth stats (m): {depth_meta.get('depth_stats')}")

    # Charts
    lines.append("\n## Charts\n")
    lines.append("![Per-species size proxy distribution](size_by_species.png)\n")
    lines.append("![Proxy vs. Cornell length](rank_check.png)\n")

    # Rank table
    if rank_pts:
        lines.append("\n## Per-species medians vs. Cornell\n")
        lines.append("| Species | Cornell (cm) | n | Median proxy | Implied scale |")
        lines.append("|---|---:|---:|---:|---:|")
        sorted_pts = sorted(rank_pts, key=lambda p: p[1])
        for sp, cl, med in sorted_pts:
            n = len(by_species[sp])
            implied = med / cl
            lines.append(f"| {sp} | {cl:.1f} | {n} | {med:.2f} | {implied:.3f} |")

    # Pairwise
    if pairwise:
        lines.append("\n## Pairwise separability (size-only AUC)\n")
        lines.append(f"Sorted descending. Pairs requiring n ≥ {MIN_PER_SPECIES_FOR_PAIRWISE} per species.\n")
        lines.append("| A | B | n_A | n_B | median_A | median_B | size-AUC | Δ real (cm) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for r in pairwise[:40]:
            delta = f"{r['real_delta_cm']:.1f}" if r["real_delta_cm"] is not None else "—"
            star = " ⭐" if r["auc"] >= 0.75 else ""
            lines.append(
                f"| {r['a']} | {r['b']} | {r['n_a']} | {r['n_b']} | "
                f"{r['median_a']:.2f} | {r['median_b']:.2f} | "
                f"{r['auc']:.3f}{star} | {delta} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(
        "- A pair with **AUC ≥ 0.75** means a size-only classifier could "
        "tell these two species apart 75 %+ of the time. That's the "
        "regime where a size prior in fuse.py would actually rescue "
        "mis-classifications.\n"
    )
    lines.append(
        "- A **Spearman ρ ≥ 0.6** means the proxy's species ranking "
        "roughly matches reality — necessary but not sufficient for "
        "shipping. Without ρ ≥ 0.6 even the relative-size signal is "
        "broken.\n"
    )

    out.write_text("\n".join(lines))
    log.info("Wrote %s", out)
    return out


def main() -> int:
    init_db()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    depth = _load_depth()
    log.info("Loaded depth map %dx%d", depth.shape[1], depth.shape[0])

    depth_meta: dict = {}
    depth_json = DATA_DIR / "calibration" / "depth_map.json"
    if depth_json.exists():
        depth_meta = json.loads(depth_json.read_text())

    db = SessionLocal()
    try:
        dets = _gather_labeled_detections(db)
        log.info("Found %d labeled real-species detections", len(dets))

        by_species, rejected = _compute_proxies(dets, depth)
        log.info(
            "Got proxies for %d detections across %d species; rejections: %s",
            sum(len(v) for v in by_species.values()),
            len(by_species),
            rejected,
        )

        _save_violin_plot(by_species, RESULTS_DIR / "size_by_species.png")
        species_medians = {s: float(np.median(v)) for s, v in by_species.items()
                           if len(v) >= MIN_PER_SPECIES_FOR_RANK}
        spearman_rho, rank_pts = _save_rank_check(species_medians, RESULTS_DIR / "rank_check.png")
        pairwise = _compute_pairwise_table(by_species)
        _write_report(by_species, rejected, spearman_rho, rank_pts, pairwise, depth_meta)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
