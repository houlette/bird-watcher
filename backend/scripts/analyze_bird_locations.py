"""Where in the camera frame do detections happen?

Renders heatmaps of bbox centers over a representative background
frame so the user can decide whether moving the camera or feeder
would meaningfully change the data distribution.

Three views, saved as PNGs under backend/scripts/depth/results/:

  1. location_heatmap.png       — density of real-bird detections
  2. small_birds_heatmap.png    — density of detections whose bbox
                                  is smaller than the median (i.e.,
                                  the resolution-limited ones)
  3. size_by_region.png         — median bbox_diag in a coarse grid,
                                  shown as a contour map. Cool spots
                                  mean small birds; warm spots mean
                                  big birds.

Also prints summary stats: median bbox_diag overall and per quadrant,
quadrant counts, and the centroid of all real-bird detections (which
is where, on average, the camera "looks" at the bird community).

Usage:
    cd backend
    python scripts/analyze_bird_locations.py
    open scripts/depth/results/location_heatmap.png
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import SENTINEL_LABELS, Correction, Detection, Species  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_bird_locations")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Default output dir is under data/ (bind-mounted) so the Stats page can
# serve the PNGs via the existing /media/ static mount. Override via
# main(output_dir=...) for ad-hoc local runs.
DEFAULT_RESULTS_DIR = DATA_DIR / "heatmaps"

# Image dimensions (4K Reolink full frame, after tile reassembly).
W, H = 3840, 2160

# Background frame for the overlay — same one the depth-map experiment
# used (a clean morning capture). If absent, we render on a black canvas.
BG_FRAME_REL = "clips/upload/2026/05/29/Birdfeeder_00_20260529102542.jpg"


def _load_real_bird_detections() -> list[tuple[int, int, int, int]]:
    """Return [(cx, cy, bbox_diag, source_class)] for every detection
    that's either (a) classifier-labeled to a non-sentinel species OR
    (b) user/llm-confirmed to a non-sentinel species. NAB-corrected
    detections are excluded so the heatmap shows where ACTUAL birds
    appear, not where YOLO false-positives cluster (the scene mask
    already handles those).

    source_class: 0 = user / classifier-confirmed; 1 = llm-claude /
    llm-claude-medium / llm-claude-confirmed; 2 = no Correction yet.
    Not used in the current heatmaps but persisted in case we want to
    slice later.
    """
    db = SessionLocal()
    try:
        # Join Detection to Species (via species_id) and outer-join to
        # Correction; filter out detections whose ASSIGNED species is a
        # sentinel (NAB / Unknown). Note: the Detection.species_id is
        # overwritten by Corrections, so this filter already reflects
        # both classifier outputs and user/LLM revisions.
        rows = (
            db.query(Detection, Species.common_name, Correction.source)
            .join(Species, Detection.species_id == Species.id)
            .outerjoin(Correction, Correction.detection_id == Detection.id)
            .filter(~Species.common_name.in_(SENTINEL_LABELS))
            .all()
        )
        out: list[tuple[int, int, int, int]] = []
        for det, sp_name, corr_source in rows:
            if not det.bbox or len(det.bbox) != 4:
                continue
            x, y, w, h = det.bbox
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            diag = int(round(float(np.hypot(w, h))))
            if corr_source is None:
                cls = 2
            elif corr_source.startswith("llm-claude"):
                cls = 1
            else:
                cls = 0
            out.append((cx, cy, diag, cls))
        return out
    finally:
        db.close()


def _bg_image() -> np.ndarray:
    bg_path = DATA_DIR / BG_FRAME_REL
    if bg_path.exists():
        img = cv2.imread(str(bg_path))
        if img is not None:
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            return img
        log.warning("Could not decode bg frame %s; falling back to black", bg_path)
    else:
        log.warning("Bg frame missing at %s; falling back to black", bg_path)
    return np.zeros((H, W, 3), dtype=np.uint8)


def _render_heatmap(
    points: list[tuple[int, int]],
    title: str,
    out_path: Path,
    bins: int = 96,
    cmap_name: str = "hot",
) -> None:
    """KDE-flavored heatmap by 2D-histogramming bbox centers and
    Gaussian-smoothing. Overlaid on the background frame at 60% alpha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    log.info("Rendering %s with %d points", out_path.name, len(points))
    if not points:
        log.warning("No points; skipping %s", out_path.name)
        return

    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)

    # bins×bins histogram across the full frame, then Gaussian smoothed
    # to read as continuous density. A 96-bin grid on 3840×2160 gives
    # ~40 px / cell, finer than the 100-px scene-mask grid.
    hist, _, _ = np.histogram2d(
        xs, ys,
        bins=[bins, bins * H // W],
        range=[[0, W], [0, H]],
    )
    smooth = gaussian_filter(hist.T, sigma=2.0)
    # Normalize to [0, 1]; saturate the top end so a single hotspot
    # doesn't crush the dynamic range.
    p_low, p_high = np.percentile(smooth, [5, 99])
    if p_high <= p_low:
        return
    norm = np.clip((smooth - p_low) / (p_high - p_low), 0.0, 1.0)

    bg = _bg_image()
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.imshow(bg_rgb, extent=(0, W, H, 0), aspect="auto")  # invert y
    ax.imshow(
        norm, extent=(0, W, H, 0), aspect="auto",
        cmap=cmap_name, alpha=0.55, interpolation="bilinear",
    )
    # 50% and 90% density contours
    cs = ax.contour(
        norm,
        extent=(0, W, H, 0),
        levels=[0.25, 0.5, 0.75],
        colors=["white"],
        alpha=0.7,
        linewidths=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt={0.25: "25%", 0.5: "50%", 0.75: "75%"})
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _render_size_grid(
    samples: list[tuple[int, int, int]],
    out_path: Path,
    cell_px: int = 240,
) -> None:
    """For each cell_px × cell_px tile, plot the median bbox_diag of
    detections falling into it. Reveals the spatial resolution-bottleneck:
    cool cells = small birds (the ones we struggle to ID)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = W // cell_px
    rows = H // cell_px
    sums = np.zeros((rows, cols), dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.int32)
    for cx, cy, diag in samples:
        c = min(cols - 1, cx // cell_px)
        r = min(rows - 1, cy // cell_px)
        sums[r, c] += diag
        counts[r, c] += 1
    medians = np.full((rows, cols), np.nan, dtype=np.float64)
    mask = counts >= 5  # need a minimum sample for a stable median
    medians[mask] = sums[mask] / counts[mask]

    bg = _bg_image()
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=120)
    ax.imshow(bg_rgb, extent=(0, W, H, 0), aspect="auto")
    im = ax.imshow(
        medians, extent=(0, W, H, 0), aspect="auto",
        cmap="RdYlGn", alpha=0.6, interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("median bbox diagonal (px)", fontsize=9)
    # Print the count in each cell as a small label so the user can
    # tell cells that are warm because of one big bird from cells with
    # lots of samples.
    for r in range(rows):
        for c in range(cols):
            if counts[r, c] >= 5:
                ax.text(
                    c * cell_px + cell_px / 2,
                    r * cell_px + cell_px / 2,
                    str(int(counts[r, c])),
                    color="black", fontsize=7, ha="center", va="center",
                )
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Median bbox diagonal by region (cells need ≥5 detections)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _print_summary(points: list[tuple[int, int, int, int]]) -> None:
    if not points:
        log.info("No data.")
        return
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    diags = np.array([p[2] for p in points], dtype=np.float32)
    print()
    print(f"Total real-bird detections: {len(points):,}")
    print(f"Bbox diagonal:")
    print(f"  median = {np.median(diags):.0f} px")
    print(f"  p25    = {np.percentile(diags, 25):.0f} px")
    print(f"  p75    = {np.percentile(diags, 75):.0f} px")
    print(f"Centroid (where the camera 'looks' on average):")
    print(f"  x = {np.mean(xs):.0f} / {W}  (frame fraction {np.mean(xs)/W*100:.0f}%)")
    print(f"  y = {np.mean(ys):.0f} / {H}  (frame fraction {np.mean(ys)/H*100:.0f}%)")
    print()
    print("Quadrant counts (top/bottom × left/right):")
    counts: Counter[str] = Counter()
    for x, y, _, _ in points:
        v = "top" if y < H / 2 else "bottom"
        h = "left" if x < W / 2 else "right"
        counts[f"{v}-{h}"] += 1
    for q in ("top-left", "top-right", "bottom-left", "bottom-right"):
        n = counts[q]
        print(f"  {q:<14} {n:>5}  ({n / len(points) * 100:>5.1f}%)")
    print()
    print("Per-quadrant median bbox diagonal:")
    for q in ("top-left", "top-right", "bottom-left", "bottom-right"):
        v_lo, v_hi = (0, H / 2) if q.startswith("top") else (H / 2, H)
        h_lo, h_hi = (0, W / 2) if q.endswith("left") else (W / 2, W)
        in_q = [d for x, y, d, _ in points if v_lo <= y < v_hi and h_lo <= x < h_hi]
        if in_q:
            print(f"  {q:<14} {np.median(in_q):>6.0f} px  (n={len(in_q)})")
        else:
            print(f"  {q:<14}      —     (no samples)")


def main(output_dir: Path | None = None) -> int:
    """Render the three heatmaps. Returns 0 on success (idempotent).
    Safe to call from the worker's nightly cron — initializes the DB,
    overwrites existing PNGs.
    """
    init_db()
    out_dir = output_dir or DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    points = _load_real_bird_detections()
    log.info("Loaded %d real-bird detections", len(points))

    if not points:
        log.warning("No real-bird detections found; nothing to render.")
        return 0

    centers = [(p[0], p[1]) for p in points]
    diags = [p[2] for p in points]
    median_diag = float(np.median(diags))

    _render_heatmap(
        centers,
        f"Real-bird detection density ({len(points):,} detections)",
        out_dir / "location_heatmap.png",
    )

    small_centers = [(p[0], p[1]) for p in points if p[2] < median_diag]
    _render_heatmap(
        small_centers,
        f"SMALL birds (bbox_diag < median = {median_diag:.0f} px) — "
        f"{len(small_centers):,} detections — the resolution-bottleneck zones",
        out_dir / "small_birds_heatmap.png",
        cmap_name="cool",
    )

    _render_size_grid(
        [(p[0], p[1], p[2]) for p in points],
        out_dir / "size_by_region.png",
    )

    _print_summary(points)
    log.info("Done. Output under %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
