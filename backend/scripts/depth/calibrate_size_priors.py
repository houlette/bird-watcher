"""Fit per-species log-normal size priors from user corrections.

Joins Detection ↔ Correction for real-species labels, takes the
`max(w, h)` of each detection's bbox, and fits a log-normal
distribution per species (with ≥ MIN_LABELS samples). Writes the
result to `data/calibration/size_priors.json` in the same shape as
`yard_priors.json` so a future loader (`pipeline.size_prior`) can
mtime-cache it the same way.

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


def _max_wh(bbox: list) -> float:
    _, _, w, h = bbox
    return float(max(w, h))


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
            .all()
        )
        log.info("Found %d real-species labels", len(rows))

        # species_name → list[max_wh]
        by_species: dict[str, list[float]] = {}
        for bbox, name in rows:
            if not bbox or len(bbox) != 4:
                continue
            val = _max_wh(bbox)
            if val <= 0:
                continue
            by_species.setdefault(name, []).append(val)

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
            "species": dict(sorted(fits.items())),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
        log.info("Wrote %d species priors → %s", len(fits), OUTPUT_PATH)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
