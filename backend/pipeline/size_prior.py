"""Per-species log-normal size prior over `max(w, h)`.

Loaded lazily from `data/calibration/size_priors.json` (built by
`scripts/depth/calibrate_size_priors.py`). Exposed to `fuse.py` as
`size_multiplier(species, bbox) → float in [SIZE_FLOOR, 1.0]`.

Three corrections applied per observation, in order:
  1. Aspect-ratio gate — if w/h is outside the calibrated bounds (the
     bird is wings-extended, partially clipped, or otherwise pose-
     contaminated), return 1.0 immediately. The size prior says
     nothing about pose outliers.
  2. Perch scaling — multiply max_wh by the precomputed scale for
     this bbox's foot-y bin. Birds higher in the frame are at greater
     depth; the anchor species (Mourning Dove) is used as a "ruler at
     every height" to normalize for camera perspective without
     invoking metric depth.
  3. Log-normal likelihood — under the species's μ_log, σ_log fit:

         z = |log(scaled_max_wh) - μ_s| / σ_s
         mult = max(SIZE_FLOOR, exp(-z))

Properties:
  - Maxes at 1.0 when the perch-scaled observation matches the
    species's geometric mean.
  - Falls to ~0.37 at one log-σ away.
  - Floors at SIZE_FLOOR so a wildly-wrong observation doesn't zero out
    the species entirely. SIZE_FLOOR=0.33 is the symmetric inverse of
    `fuse.AUDIO_BOOST=3.0` — the prior nudges by a comparable
    magnitude in the opposite direction, never dominates.
  - Returns 1.0 (uninformative) when:
      a) the calibration file is missing or malformed,
      b) the species has no entry (fewer than MIN_LABELS at calibration
         time),
      c) the bbox aspect ratio is out of bounds, or
      d) the bbox is degenerate.

If the calibration file is missing, `fuse.py` callers see an identity
multiplier and the pipeline behaves exactly as before — same fallback
discipline as `pipeline.calibration` (yard priors).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

log = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent.parent / "data" / "calibration" / "size_priors.json"

# Floor for the size multiplier. Match magnitude of fuse.AUDIO_BOOST (3.0)
# so neither prior can solo-dominate posterior re-ranking. Higher floor =
# weaker prior; lower floor = stronger penalty for size mismatch.
SIZE_FLOOR = 0.33

# Defaults if the calibration JSON doesn't carry these (older format).
_DEFAULT_ASPECT_BOUNDS = (0.40, 2.50)
_DEFAULT_IMAGE_HEIGHT_PX = 2160


@dataclass(frozen=True)
class _SpeciesFit:
    n: int
    log_mean: float
    log_std: float


@dataclass(frozen=True)
class _CalibrationCache:
    species: dict[str, _SpeciesFit]
    aspect_bounds: tuple[float, float]
    # Sorted-by-y-min list of (y_min, y_max, scale) for the perch
    # scaling. Empty list means "no perch correction" — every observation
    # gets scale=1.0 regardless of foot_y.
    perch_bins: list[tuple[float, float, float]]
    image_height_px: int


_lock = Lock()
_cache: _CalibrationCache | None = None
_loaded_mtime: float | None = None


def _parse_perch_bins(perch_payload: dict | None) -> list[tuple[float, float, float]]:
    if not perch_payload:
        return []
    scales = perch_payload.get("scales") or []
    out: list[tuple[float, float, float]] = []
    for entry in scales:
        try:
            out.append((
                float(entry["y_min"]),
                float(entry["y_max"]),
                float(entry["scale"]),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda t: t[0])
    return out


def _load_priors() -> _CalibrationCache | None:
    """Load + cache size priors. Returns None if missing/malformed."""
    global _cache, _loaded_mtime

    if not CALIBRATION_PATH.exists():
        return None

    mtime = CALIBRATION_PATH.stat().st_mtime
    if _cache is not None and _loaded_mtime == mtime:
        return _cache

    with _lock:
        # Double-check inside the lock — a second thread may have loaded.
        if _cache is not None and _loaded_mtime == mtime:
            return _cache
        try:
            payload = json.loads(CALIBRATION_PATH.read_text())
            species_dict = payload.get("species", {})
            fits: dict[str, _SpeciesFit] = {}
            for name, entry in species_dict.items():
                fits[name] = _SpeciesFit(
                    n=int(entry["n"]),
                    log_mean=float(entry["log_mean"]),
                    log_std=float(entry["log_std"]),
                )
            ab = payload.get("aspect_ratio_bounds") or _DEFAULT_ASPECT_BOUNDS
            aspect_bounds = (float(ab[0]), float(ab[1]))
            perch_payload = payload.get("perch") or {}
            perch_bins = _parse_perch_bins(perch_payload)
            image_height = int(perch_payload.get("image_height_px", _DEFAULT_IMAGE_HEIGHT_PX))

            _cache = _CalibrationCache(
                species=fits,
                aspect_bounds=aspect_bounds,
                perch_bins=perch_bins,
                image_height_px=image_height,
            )
            _loaded_mtime = mtime
            log.info(
                "Loaded size priors: %d species, %d perch bins, aspect bounds %s, from %s",
                len(fits), len(perch_bins), aspect_bounds,
                payload.get("generated_at", "?"),
            )
            return _cache
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            log.exception("Failed to load size priors from %s", CALIBRATION_PATH)
            _cache = None
            _loaded_mtime = mtime  # don't re-attempt until file changes
            return None


def reset_cache_for_tests() -> None:
    """Test hook — drop the cached priors so tests can re-monkeypatch
    CALIBRATION_PATH and re-load. Mirrors pipeline.calibration's pattern."""
    global _cache, _loaded_mtime
    with _lock:
        _cache = None
        _loaded_mtime = None


def _scale_for_foot_y(cache: _CalibrationCache, foot_y: float) -> float:
    """Look up the perch scale factor for a foot_y pixel coord.
    Falls back to 1.0 if the y is outside every bin or no bins exist."""
    for y_min, y_max, scale in cache.perch_bins:
        if y_min <= foot_y < y_max:
            return scale
    return 1.0


def size_multiplier(species: str, bbox: tuple | list | None) -> float:
    """Multiplicative prior for `species` given observed bbox `[x,y,w,h]`.

    Applies aspect-ratio gating + perch scaling internally; returns 1.0
    (no effect) when:
      - the calibration file is missing or malformed,
      - the species has no entry,
      - the bbox is degenerate, or
      - the bbox aspect ratio is outside the calibrated bounds.
    """
    if bbox is None or len(bbox) != 4:
        return 1.0
    _, by, w, h = bbox
    if w <= 0 or h <= 0:
        return 1.0
    cache = _load_priors()
    if cache is None:
        return 1.0
    fit = cache.species.get(species)
    if fit is None or fit.log_std <= 0:
        return 1.0

    # Aspect-ratio gate — pose outliers get no prior at all.
    aspect = w / h
    if aspect < cache.aspect_bounds[0] or aspect > cache.aspect_bounds[1]:
        return 1.0

    # Perch scaling.
    foot_y = by + h
    scale = _scale_for_foot_y(cache, foot_y)
    scaled_max_wh = max(w, h) * scale
    if scaled_max_wh <= 0:
        return 1.0

    z = abs((math.log(scaled_max_wh) - fit.log_mean) / fit.log_std)
    return max(SIZE_FLOOR, math.exp(-z))
