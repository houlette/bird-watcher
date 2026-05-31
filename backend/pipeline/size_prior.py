"""Per-species log-normal size prior over `max(w, h)`.

Loaded lazily from `data/calibration/size_priors.json` (built by
`scripts/depth/calibrate_size_priors.py`). Exposed to `fuse.py` as
`size_multiplier(species, max_wh) → float in [SIZE_FLOOR, 1.0]`.

The multiplier is the (truncated) log-normal density at the observed
size, evaluated under the species's own fit:

    z = |log(max_wh) - μ_s| / σ_s
    mult = max(SIZE_FLOOR, exp(-z))

Properties:
  - Maxes at 1.0 when max_wh is exactly the species's geometric mean.
  - Falls to ~0.37 at one log-σ away (typical size for the species).
  - Floors at SIZE_FLOOR so a wildly-wrong observation doesn't zero out
    the species entirely. SIZE_FLOOR=0.33 is the symmetric inverse of
    `fuse.AUDIO_BOOST=3.0` — the prior nudges by a comparable
    magnitude in the opposite direction, never dominates.
  - Returns 1.0 (uninformative) when:
      a) the calibration file is missing or malformed,
      b) the species has no entry (fewer than MIN_LABELS at calibration
         time), or
      c) max_wh is non-positive (degenerate bbox).

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


@dataclass(frozen=True)
class _SpeciesFit:
    n: int
    log_mean: float
    log_std: float


_lock = Lock()
_cache: dict[str, _SpeciesFit] | None = None
_loaded_mtime: float | None = None


def _load_priors() -> dict[str, _SpeciesFit] | None:
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
            species = payload.get("species", {})
            fits: dict[str, _SpeciesFit] = {}
            for name, entry in species.items():
                fits[name] = _SpeciesFit(
                    n=int(entry["n"]),
                    log_mean=float(entry["log_mean"]),
                    log_std=float(entry["log_std"]),
                )
            _cache = fits
            _loaded_mtime = mtime
            log.info(
                "Loaded size priors: %d species from %s",
                len(fits), payload.get("generated_at", "?"),
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


def size_multiplier(species: str, max_wh: float) -> float:
    """Multiplicative prior for `species` given observed bbox `max(w,h)`.

    Returns 1.0 (no effect) when there's no calibrated fit for the
    species or the input is degenerate.
    """
    if max_wh is None or max_wh <= 0:
        return 1.0
    priors = _load_priors()
    if priors is None:
        return 1.0
    fit = priors.get(species)
    if fit is None or fit.log_std <= 0:
        return 1.0
    z = abs((math.log(max_wh) - fit.log_mean) / fit.log_std)
    return max(SIZE_FLOOR, math.exp(-z))
