"""Yard-specific calibration: per-species presence and seasonal pattern.

Reads `data/calibration/yard_priors.json` (built by
`scripts/calibrate_from_haikubox.py`) and exposes two pieces of derived
knowledge to the rest of the pipeline:

  - `get_allowlist()`: which species are known to actually visit *this* yard.
    Replaces the hand-coded NA backyard list in classify.py with one derived
    from real Haikubox audio detections.
  - `get_monthly_multiplier(species, month)`: how much more (or less) likely
    this species is in this month relative to its yard-wide average. Replaces
    the hand-coded seasonal table in fuse.py with one derived from real
    monthly distribution data.

If the calibration file is missing or malformed, both functions return
sentinel values that tell the callers to use their hard-coded fallbacks.
That keeps the pipeline functional on day one (before the user has run the
calibration script) and trivially testable.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)

CALIBRATION_PATH = Path(__file__).parent.parent / "data" / "calibration" / "yard_priors.json"

# Don't allow-list a species the Haikubox has only barely heard — most likely
# a misidentification or extreme vagrant. This threshold can be relaxed once
# we have confidence in BirdNET's signal at this site.
MIN_DETECTIONS_FOR_ALLOWLIST = 5

# Cap the seasonal multiplier so a very-seasonal species doesn't completely
# swamp the visual classifier. We want the prior to nudge close calls, not
# override clear visual evidence.
MAX_SEASONAL_MULTIPLIER = 4.0
MIN_SEASONAL_MULTIPLIER = 0.1


_lock = Lock()
_cache: dict[str, Any] | None = None
_loaded_mtime: float | None = None


def _load_calibration() -> dict[str, Any] | None:
    """Load + cache the calibration file. Returns None if missing/malformed."""
    global _cache, _loaded_mtime

    if not CALIBRATION_PATH.exists():
        return None

    mtime = CALIBRATION_PATH.stat().st_mtime
    if _cache is not None and _loaded_mtime == mtime:
        return _cache

    with _lock:
        # Re-check after acquiring the lock.
        if _cache is not None and _loaded_mtime == mtime:
            return _cache
        try:
            with CALIBRATION_PATH.open() as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Calibration file at %s unreadable: %s", CALIBRATION_PATH, exc)
            return None

        if not isinstance(data, dict) or "species" not in data:
            log.warning("Calibration file at %s has unexpected shape", CALIBRATION_PATH)
            return None

        _cache = data
        _loaded_mtime = mtime
        n = len(data.get("species", {}))
        log.info("Loaded yard calibration: %d species from %s", n, data.get("generated_at"))
        return _cache


def is_calibrated() -> bool:
    """True when a yard_priors.json is present and parseable."""
    return _load_calibration() is not None


def get_allowlist() -> set[str] | None:
    """Set of species names with enough Haikubox detections to count as
    'known to visit this yard', or None to signal 'use hand-coded fallback'."""
    data = _load_calibration()
    if data is None:
        return None
    species = data.get("species", {})
    return {
        name
        for name, info in species.items()
        if isinstance(info, dict) and info.get("total", 0) >= MIN_DETECTIONS_FOR_ALLOWLIST
    }


def get_monthly_multiplier(species: str, month: int) -> float | None:
    """Seasonal prior multiplier for `species` in `month` (1-12).

    Returns None when the calibration file is absent — tells callers to fall
    back to the hand-coded _SEASONAL_PRIORS table. Returns 1.0 for species the
    Haikubox knows about but has no per-month signal for. Returns a clamped
    multiplier derived from the species' monthly distribution otherwise:

        multiplier = (this month's fraction) / (1/12)

    so a species that's present in March 50% of the time gets a 6× multiplier
    (clamped to MAX_SEASONAL_MULTIPLIER), while one with 0% gets the floor.
    """
    data = _load_calibration()
    if data is None:
        return None

    species_data = data.get("species", {}).get(species)
    if not isinstance(species_data, dict):
        # Species not in calibration → no signal, treat as neutral (1.0) so the
        # classifier's visual probability passes through unchanged.
        return 1.0

    monthly = species_data.get("monthly_pct")
    if not isinstance(monthly, dict):
        return 1.0

    fraction = monthly.get(str(month))
    if fraction is None:
        return 1.0
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        return 1.0

    multiplier = fraction * 12.0  # 1/12 = neutral
    return max(MIN_SEASONAL_MULTIPLIER, min(MAX_SEASONAL_MULTIPLIER, multiplier))


def reset_cache_for_tests() -> None:
    """Test helper — invalidate the in-memory cache after the test mutates the file."""
    global _cache, _loaded_mtime
    with _lock:
        _cache = None
        _loaded_mtime = None
