"""Fit per-species log-normal size priors from user corrections.

Pipeline:
  1. Pull every Detection with a real-species Correction.
  2. Drop detections whose bbox aspect ratio (w/h) is outside
     [ASPECT_RATIO_MIN, ASPECT_RATIO_MAX] — wings-extended frames,
     partial-tail clips, off-frame artifacts. These contaminate the
     per-species fit and inflate σ_log.
  3. Fit a per-region Dove anchor curve: discretize foot_y into
     N_PERCH_BINS bins, compute Dove (anchor species) median max_wh
     per bin. For any bin with ≥ MIN_PER_BIN_FOR_PERCH Dove samples,
     emit a scale factor = global_dove_median / bin_dove_median. This
     normalizes for camera perspective WITHOUT depending on any
     monocular-depth model — Doves are the "ruler at every distance"
     because we have lots of them at every part of the feeder.
  4. Apply the perch scaling to each detection's max_wh BEFORE fitting
     per-species log-normals. This shrinks within-species variance
     (good for pair AUCs) without changing the mean (the species
     ordering is preserved).
  5. Write fits + perch_scales to `data/calibration/size_priors.json`.

Why log-normal: bird-bbox pixels are strictly positive and the
multiplicative noise model (camera depth × pose × YOLO bbox jitter)
implies log-normal more naturally than Gaussian. Combining log-normal
likelihoods is also clean — log space turns the multiplicative prior
into an additive shift.

Why `max(w, h)` and not bbox_diag: head-to-head comparison in
`compare_proxies.py` showed `max_wh` wins every songbird-vs-songbird
pair (Robin/Cardinal/Dove discrimination). User explicitly opted for
songbird discrimination over pigeon/dove discrimination.

Usage:
    cd backend
    python scripts/depth/calibrate_size_priors.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow `python scripts/depth/calibrate_size_priors.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import SENTINEL_LABELS, Correction, Detection, Species  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate_size_priors")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "calibration" / "size_priors.json"

# Below this label count the fit is too noisy to use. Species with
# fewer labels get NO prior (size_multiplier returns 1.0 — uninformative).
# 20 was the threshold we used in the validation analysis where pair AUCs
# were stable; using the same number here keeps the prior consistent
# with what we already measured.
MIN_LABELS = 20

# Cap log_std so a species with very uniform labels (small std) doesn't
# produce a knife-edge prior that nukes its own posterior for any
# even-slightly-different observation. 0.10 corresponds to ±10 % size
# variation as the "really tight" floor — anything tighter is almost
# certainly bbox-noise underestimation, not a real biological signal.
MIN_LOG_STD = 0.10

# Pose-gating bounds on bbox aspect ratio (w / h). 0.4 catches narrow
# slits (head-on or partial-occlusion clipping); 2.5 catches very-wide
# wings-extended frames. The bird literature would call this "extreme
# postures" — they're real birds but the bbox doesn't reflect body size.
ASPECT_RATIO_MIN = 0.40
ASPECT_RATIO_MAX = 2.50

# Perch perspective calibration. Discretize foot_y into N_PERCH_BINS
# equal-pixel bins across the 4K frame height; per-bin Dove medians
# define a scale factor we apply to ALL species in that bin.
# ANCHOR_SPECIES must be the densest-sampled species across the bbox
# foot-y range. MIN_PER_BIN_FOR_PERCH gates which bins emit a scale;
# bins below the threshold fall back to scale=1.0 (no correction).
N_PERCH_BINS = 5
ANCHOR_SPECIES = "Mourning Dove"
MIN_PER_BIN_FOR_PERCH = 50

# Image height — must match Detection.bbox's reference frame. The
# Reolink RLC-811WA shoots at 4K (2160 rows); detect.py converts tile
# bboxes back to full-frame coords before storage.
IMAGE_HEIGHT_PX = 2160


def _max_wh(bbox: list) -> float:
    _, _, w, h = bbox
    return float(max(w, h))


def _aspect_ratio_ok(bbox: list) -> bool:
    """True if the bbox is within ASPECT_RATIO_MIN/MAX bounds."""
    _, _, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    ratio = w / h
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX


def _foot_y(bbox: list) -> float:
    _, by, _, bh = bbox
    return float(by + bh)


def _bin_index(foot_y: float, n_bins: int = N_PERCH_BINS, height: int = IMAGE_HEIGHT_PX) -> int:
    """Map a foot_y pixel coord to a discrete bin index in [0, n_bins)."""
    bin_h = height / n_bins
    idx = int(foot_y // bin_h)
    return max(0, min(n_bins - 1, idx))


def _build_perch_scales(rows_aspect_ok: list[tuple[list, str]]) -> list[dict]:
    """Compute per-y-bin perspective scale factors anchored on the
    dense-sampled anchor species. Returns a list of bin descriptors:
        [{"y_min": ..., "y_max": ..., "n": ..., "scale": ...}, ...]
    where `scale` is the multiplier to apply to a raw max_wh observed
    in that bin to "rescale" it to the anchor's global median context.
    Bins below MIN_PER_BIN_FOR_PERCH samples get scale=1.0 (no-op).
    """
    by_bin: list[list[float]] = [[] for _ in range(N_PERCH_BINS)]
    for bbox, name in rows_aspect_ok:
        if name != ANCHOR_SPECIES:
            continue
        by_bin[_bin_index(_foot_y(bbox))].append(_max_wh(bbox))

    all_anchor = [v for vs in by_bin for v in vs]
    if not all_anchor:
        log.warning("No %s samples — emitting no perch scales", ANCHOR_SPECIES)
        return []
    global_median = float(np.median(all_anchor))
    log.info(
        "Anchor (%s) global median max_wh = %.1f px across %d samples",
        ANCHOR_SPECIES, global_median, len(all_anchor),
    )

    bin_h = IMAGE_HEIGHT_PX / N_PERCH_BINS
    out: list[dict] = []
    for i, vs in enumerate(by_bin):
        y_min = int(i * bin_h)
        y_max = int((i + 1) * bin_h)
        if len(vs) >= MIN_PER_BIN_FOR_PERCH:
            bin_median = float(np.median(vs))
            scale = global_median / bin_median if bin_median > 0 else 1.0
        else:
            bin_median = float(np.median(vs)) if vs else None
            scale = 1.0  # not enough data → no-op
        out.append({
            "y_min": y_min,
            "y_max": y_max,
            "n": len(vs),
            "bin_median": bin_median,
            "scale": scale,
        })
        log.info(
            "  perch bin [y∈[%d,%d)]: n=%d, bin_median=%s, scale=%.3f",
            y_min, y_max, len(vs),
            f"{bin_median:.1f}" if bin_median is not None else "—",
            scale,
        )
    return out


def _scale_for_bin(perch_scales: list[dict], foot_y: float) -> float:
    """Look up the scale factor for a foot_y. Falls back to 1.0 if no
    scales are configured."""
    if not perch_scales:
        return 1.0
    idx = _bin_index(foot_y)
    return float(perch_scales[idx]["scale"])


def main() -> int:
    init_db()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    db = SessionLocal()
    try:
        rows = (
            db.query(Detection.bbox, Species.common_name)
            .join(Correction, Correction.detection_id == Detection.id)
            .join(Species, Correction.correct_species_id == Species.id)
            .filter(~Species.common_name.in_(SENTINEL_LABELS))
            # Family labels ("Sparrow", "Warbler") collapse multiple species
            # of varying real sizes; fitting one log-normal across all of
            # them just inflates σ and pollutes the species fits we DO
            # care about. Skip them at calibration time.
            .filter((Species.is_family.is_(None)) | (Species.is_family == 0))
            .all()
        )
        log.info("Found %d real-species labels", len(rows))

        # Step 1: aspect-ratio gating. Drop wings-extended / clipped frames.
        rows_aspect_ok: list[tuple[list, str]] = []
        dropped_aspect = 0
        for bbox, name in rows:
            if not bbox or len(bbox) != 4:
                continue
            if _aspect_ratio_ok(bbox):
                rows_aspect_ok.append((bbox, name))
            else:
                dropped_aspect += 1
        log.info(
            "Aspect-ratio gate (∈[%.2f, %.2f]): kept %d, dropped %d (%.1f%%)",
            ASPECT_RATIO_MIN, ASPECT_RATIO_MAX,
            len(rows_aspect_ok), dropped_aspect,
            100.0 * dropped_aspect / max(1, len(rows)),
        )

        # Step 2: build per-perch scale table from the anchor species.
        perch_scales = _build_perch_scales(rows_aspect_ok)

        # Step 3: fit per-species log-normals on PERCH-CORRECTED max_wh
        # values. Each detection contributes max_wh × scale[its_y_bin].
        # This shrinks σ_log without shifting the relative ordering.
        by_species: dict[str, list[float]] = {}
        for bbox, name in rows_aspect_ok:
            raw = _max_wh(bbox)
            if raw <= 0:
                continue
            scale = _scale_for_bin(perch_scales, _foot_y(bbox))
            by_species.setdefault(name, []).append(raw * scale)

        fits: dict[str, dict] = {}
        for sp, vals in by_species.items():
            if len(vals) < MIN_LABELS:
                continue
            log_vals = [math.log(v) for v in vals]
            n = len(log_vals)
            mu = sum(log_vals) / n
            var = sum((x - mu) ** 2 for x in log_vals) / max(1, n - 1)
            sigma = max(MIN_LOG_STD, math.sqrt(var))
            fits[sp] = {
                "n": n,
                "log_mean": mu,
                "log_std": sigma,
                # For the report / sanity-eyeballing — these are the
                # geometric mean and approximate range, in pixels.
                "geometric_mean_px": math.exp(mu),
                "approx_p05_px": math.exp(mu - 1.645 * sigma),
                "approx_p95_px": math.exp(mu + 1.645 * sigma),
            }
            log.info(
                "Fitted %s: n=%d, μ_log=%.3f, σ_log=%.3f, gm=%.1f px [%.1f, %.1f]",
                sp, n, mu, sigma, math.exp(mu),
                fits[sp]["approx_p05_px"], fits[sp]["approx_p95_px"],
            )

        if not fits:
            log.warning("No species met MIN_LABELS=%d; writing empty priors", MIN_LABELS)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "proxy": "max_wh",
            "min_labels": MIN_LABELS,
            "min_log_std": MIN_LOG_STD,
            "aspect_ratio_bounds": [ASPECT_RATIO_MIN, ASPECT_RATIO_MAX],
            "perch": {
                "image_height_px": IMAGE_HEIGHT_PX,
                "n_bins": N_PERCH_BINS,
                "anchor_species": ANCHOR_SPECIES,
                "min_per_bin": MIN_PER_BIN_FOR_PERCH,
                "scales": perch_scales,
            },
            "species": dict(sorted(fits.items())),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
        log.info("Wrote %d species priors → %s", len(fits), OUTPUT_PATH)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
