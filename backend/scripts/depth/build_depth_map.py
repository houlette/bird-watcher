"""Build a static metric depth map of the bird-feeder scene.

Runs Depth Anything V2 Metric (outdoor) on a single clean background
frame and persists the result to `data/calibration/depth_map.npz` +
sidecar `depth_map.json`. A future production loader can mtime-cache
the JSON the way `pipeline/calibration.py` does today.

Usage:
    python scripts/depth/build_depth_map.py            # auto-pick a frame
    python scripts/depth/build_depth_map.py --frame path/to/bg.jpg

Notes:
  - Auto-pick walks `data/clips/upload/` looking for a sunrise-hour JPG
    (less likely to contain a bird). For best results, pick a frame
    manually from a quiet dawn.
  - First run downloads the model weights (~25 MB for Small,
    ~100 MB for Base). Cached by transformers thereafter.
  - The diagnostic overlay PNG is written to the same calibration dir;
    open it to sanity-check whether the depth map makes sense (closer
    things should appear hotter).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/depth/build_depth_map.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_depth_map")


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
CAL_DIR = DATA_DIR / "calibration"
DEPTH_NPZ = CAL_DIR / "depth_map.npz"
DEPTH_JSON = CAL_DIR / "depth_map.json"
DEPTH_OVERLAY = CAL_DIR / "depth_map_overlay.png"

# Model choice: -Small is ~25 MB and fast on CPU; the larger variants
# don't help much for the static-background use case where we only care
# about scene structure, not bird-scale detail. If you find the depth
# map looks coarse near the feeder edge, swap to -Base.
MODEL_NAME = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def _auto_pick_frame() -> Path | None:
    """Walk the clips directory for a recent dawn JPG."""
    candidates: list[Path] = []
    for f in CLIPS_DIR.rglob("*.jpg"):
        # Filenames look like Birdfeeder_00_YYYYMMDDHHMMSS.jpg.
        name = f.name
        if len(name) < 20:
            continue
        try:
            hour = int(name[-10:-8])
        except ValueError:
            continue
        # 5–7 UTC = roughly 1–3 AM EDT — too early. We want
        # 10–12 UTC = 6–8 AM EDT, dawn-ish, with light but minimal
        # bird activity.
        if 10 <= hour <= 12:
            candidates.append(f)
    if not candidates:
        return None
    # Most recent file — most-recent yard state is most useful.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"Could not read image: {path}")
    return bgr


def _run_depth(rgb: np.ndarray) -> np.ndarray:
    """Run Depth Anything V2 Metric Outdoor on an RGB uint8 image,
    returning depth in meters as a float32 array shaped like the input."""
    # Import here so the script's CLI / autopick logic doesn't pay for
    # the multi-second torch / transformers import unless it's about
    # to use them.
    from transformers import pipeline as hf_pipeline

    log.info("Loading depth model: %s", MODEL_NAME)
    estimator = hf_pipeline("depth-estimation", model=MODEL_NAME)

    # transformers' depth pipeline expects a PIL image.
    from PIL import Image
    pil = Image.fromarray(rgb)

    log.info("Running depth inference on %d×%d image…", rgb.shape[1], rgb.shape[0])
    out = estimator(pil)
    # The transformers depth pipeline returns BOTH `predicted_depth`
    # (float32 metric meters for the Metric-Outdoor model) AND `depth`
    # (a PIL "L" mode visualization rescaled to 0–255). Always prefer
    # `predicted_depth` — the PIL image has lost the metric scale.
    if "predicted_depth" in out:
        depth = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    elif "depth" in out and getattr(out["depth"], "mode", None) == "I;16":
        # Fallback: some older pipeline versions return only an "I;16"
        # PIL image whose raw pixel value IS the metric depth in mm.
        depth = np.array(out["depth"], dtype=np.float32) / 1000.0
    else:
        raise SystemExit(f"Unexpected depth pipeline output keys: {list(out.keys())}")

    # Some pipeline versions return the depth at the model's native
    # resolution, not the input's. Resample if needed.
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(
            depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return depth


def _save_overlay(bgr: np.ndarray, depth: np.ndarray, path: Path) -> None:
    """Render the depth map as a colored overlay on the source image
    for eyeball validation."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        log.warning("Depth map has no finite values; skipping overlay")
        return
    lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
    if hi <= lo:
        return
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(bgr, 0.45, heat, 0.55, 0.0)
    cv2.imwrite(str(path), overlay)
    log.info("Wrote diagnostic overlay → %s", path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame",
        type=Path,
        help="Path to a clean background frame (default: auto-pick a "
        "dawn JPG from data/clips/upload/).",
    )
    args = parser.parse_args()

    CAL_DIR.mkdir(parents=True, exist_ok=True)

    frame_path: Path | None = args.frame
    if frame_path is None:
        frame_path = _auto_pick_frame()
        if frame_path is None:
            log.error(
                "No dawn JPGs found under %s; pass --frame manually.", CLIPS_DIR
            )
            return 1
        log.info("Auto-picked background frame: %s", frame_path)

    bgr = _load_image(frame_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth = _run_depth(rgb)

    finite = depth[np.isfinite(depth)]
    stats = {
        "min_m": float(np.min(finite)) if finite.size else None,
        "max_m": float(np.max(finite)) if finite.size else None,
        "median_m": float(np.median(finite)) if finite.size else None,
        "mean_m": float(np.mean(finite)) if finite.size else None,
    }

    log.info("Depth stats (m): %s", stats)

    # The .npz holds the actual depth array (too big for JSON).
    np.savez_compressed(DEPTH_NPZ, depth=depth.astype(np.float32))
    log.info("Wrote depth array → %s (%.1f KB)", DEPTH_NPZ, DEPTH_NPZ.stat().st_size / 1024)

    # JSON sidecar matches the shape of yard_priors.json (see
    # pipeline/calibration.py) so a future loader is a copy-paste.
    sidecar = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_frame": str(frame_path.relative_to(DATA_DIR)),
        "model": MODEL_NAME,
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "depth_npz_path": str(DEPTH_NPZ.relative_to(DATA_DIR)),
        "depth_stats": stats,
    }
    DEPTH_JSON.write_text(json.dumps(sidecar, indent=2))
    log.info("Wrote sidecar → %s", DEPTH_JSON)

    _save_overlay(bgr, depth, DEPTH_OVERLAY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
