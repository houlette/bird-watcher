"""Bayesian fusion of visual evidence with audio + seasonal priors.

Given the classifier's top-K predictions for one visit, we want a posterior
over species that takes into account:

  - Audio: did the Haikubox hear this species recently?
  - Seasonal: how likely is this species at the feeder in this month?

We treat all three signals as independent likelihoods and multiply:

    P_post(species) ∝ P_visual(species) · P_audio(species) · P_seasonal(species)

then renormalize over the candidate set. We deliberately renormalize only over
the visual top-K rather than over all species; species outside the visual
top-K are extremely unlikely given the photo, and including them would let
strong audio/season priors override clear visual evidence for an unrelated
common bird that simply happened to be heard.

`audio_confirmed` is True when the chosen top-1 had at least one matching
Haikubox detection in the correlation window — useful for UI badges.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from sqlalchemy.orm import Session

from db.models import HaikuboxDetection
from db.utils import utcnow
from pipeline import calibration, size_prior
from settings import settings

log = logging.getLogger(__name__)

# Multiplicative boost to a species' probability if the Haikubox heard it
# within the correlation window. Tuned empirically: too high and audio
# overwhelms a clear visual; too low and we get no benefit. 3× lets a
# 0.30 visual prob beat a 0.55 visual prob when audio backs it.
AUDIO_BOOST = 3.0

# Floor for the audio prior so unheard species aren't zeroed out (which
# would break renormalization in the all-heard-none-matched edge case).
AUDIO_FLOOR = 1.0


@dataclass
class FusedPrediction:
    species: str
    probability: float
    audio_confirmed: bool
    seasonal_boost: float  # 1.0 means no effect
    size_mult: float = 1.0  # 1.0 means no effect (no size calibration / unknown species)


def _audio_species_set(db: Session, when: datetime) -> set[str]:
    """Common names heard by the Haikubox within the correlation window.

    Spans [when - lookback, when + lookahead]. Covers pre-visit approach, the
    duration of the video clip (~20s), immediate post-visit departure, and
    minor clock drift between camera NTP and Haikubox cloud clock.
    """
    window_start = when - timedelta(seconds=settings.audio_correlation_window_seconds)
    window_end = when + timedelta(seconds=settings.audio_correlation_lookahead_seconds)
    rows = (
        db.query(HaikuboxDetection.species_common_name)
        .filter(HaikuboxDetection.detected_at >= window_start)
        .filter(HaikuboxDetection.detected_at <= window_end)
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


# Seasonal priors: a rough monthly multiplier for common eastern-NA backyard
# species. 1.0 = no effect, <1.0 = less likely this month, >1.0 = more likely.
# This is intentionally coarse — Phase 6 active learning will personalize it
# to the user's actual yard, but the hand-tuned table catches the obvious
# wins (e.g., juncos in winter, hummingbirds in summer) on day one.
#
# Months are 1-indexed. Keys must match the classifier's label strings; only
# the species we feel confident about are listed. Unlisted species get 1.0.
# We use a floor of 0.02 instead of 0.0 so Cromwell's rule holds: overwhelming visual
# or audio evidence for an unexpected vagrant / overwintering bird is never zeroed out.
_SEASONAL_PRIORS: dict[str, dict[int, float]] = {
    "Dark-eyed Junco": {
        1: 2.0, 2: 2.0, 3: 1.5, 4: 0.5, 5: 0.1, 6: 0.1,
        7: 0.1, 8: 0.1, 9: 0.5, 10: 1.5, 11: 2.0, 12: 2.0,
    },
    "Ruby-throated Hummingbird": {
        1: 0.02, 2: 0.02, 3: 0.1, 4: 1.0, 5: 2.0, 6: 2.0,
        7: 2.0, 8: 2.0, 9: 1.5, 10: 0.3, 11: 0.02, 12: 0.02,
    },
    "American Goldfinch": {
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5, 6: 1.5,
        7: 1.5, 8: 1.5, 9: 1.2, 10: 1.0, 11: 1.0, 12: 1.0,
    },
    "White-throated Sparrow": {
        1: 1.5, 2: 1.5, 3: 1.3, 4: 1.0, 5: 0.3, 6: 0.1,
        7: 0.1, 8: 0.1, 9: 0.5, 10: 1.3, 11: 1.5, 12: 1.5,
    },
    "Baltimore Oriole": {
        1: 0.02, 2: 0.02, 3: 0.1, 4: 1.0, 5: 2.0, 6: 1.5,
        7: 1.0, 8: 0.5, 9: 0.2, 10: 0.02, 11: 0.02, 12: 0.02,
    },
}


def _seasonal_multiplier(species: str, month: int) -> float:
    """Pick the seasonal prior multiplier for `species` in `month`.

    Prefer the yard-calibrated monthly distribution (built from real Haikubox
    detections). Fall back to the hand-coded eastern-NA table when no
    calibration is present.
    """
    calibrated = calibration.get_monthly_multiplier(species, month)
    if calibrated is not None:
        return calibrated
    return _SEASONAL_PRIORS.get(species, {}).get(month, 1.0)


# Generic family / category labels that represent valid regional classifications
# even if not specified down to species.
COMMON_FAMILY_LABELS: frozenset[str] = frozenset({
    "sparrow",
    "woodpecker",
    "hawk",
    "finch",
    "warbler",
    "thrush",
    "gull",
    "duck",
    "swallow",
    "wren",
    "dove",
    "pigeon",
    "blackbird",
    "flycatcher",
    "sandpiper",
    "owl",
    "goose",
    "heron",
    "falcon",
    "titmouse",
    "chickadee",
    "nuthatch",
    "oriole",
    "tanager",
    "bunting",
    "grosbeak",
    "vireo",
    "kinglet",
    "unknown bird",
})

_regional_species_normalized: set[str] | None = None


def get_regional_species() -> set[str]:
    """Union of yard-calibrated species (from Haikubox >=5 detections) and
    curated Eastern-NA backyard species, excluding southwestern lookalikes.
    All strings normalized via _hyphen_insensitive.
    """
    global _regional_species_normalized
    if _regional_species_normalized is None:
        from pipeline.classify import NA_BACKYARD_ALLOWLIST, _hyphen_insensitive
        # Base curated eastern NA backyard species, excluding southwestern lookalikes like Pyrrhuloxia
        curated = {
            _hyphen_insensitive(s)
            for s in NA_BACKYARD_ALLOWLIST
            if _hyphen_insensitive(s) != "pyrrhuloxia"
        }
        curated.update(COMMON_FAMILY_LABELS)
        _regional_species_normalized = curated

    res = set(_regional_species_normalized)
    calibrated = calibration.get_allowlist()
    if calibrated:
        from pipeline.classify import _hyphen_insensitive
        for s in calibrated:
            res.add(_hyphen_insensitive(s))
    return res


def is_regional_species(species: str | None) -> bool:
    """True if species is in the yard calibration or curated Eastern-NA baseline."""
    if not species:
        return False
    from pipeline.classify import _hyphen_insensitive
    norm = _hyphen_insensitive(species)
    return norm in get_regional_species() or norm in COMMON_FAMILY_LABELS


# Alias for backward compatibility
_is_known_or_curated_species = is_regional_species


def fuse(
    predictions: Iterable[tuple[str, float]],
    *,
    db: Session,
    when: datetime | None = None,
    bbox: tuple[float, float, float, float] | list[float] | None = None,
) -> list[FusedPrediction]:
    """Re-rank visual predictions by combining with audio + seasonal priors.

    `predictions` is the top-K from the classifier as (species, probability).
    `bbox`, if given, enables the per-species size prior (max(w, h) under
    the log-normal fit from `data/calibration/size_priors.json`). When
    `bbox` is None or the calibration file is absent, the size prior is
    a no-op (1.0 multiplier) — preserving pre-prior behavior.

    Returns a list of FusedPrediction sorted by posterior probability desc.
    """
    when = when or utcnow()
    audio_heard = _audio_species_set(db, when) if (settings.haikubox_serial and settings.haikubox_api_key) else set()
    month = when.month

    # `pipeline.size_prior.size_multiplier()` handles aspect-ratio gating
    # and perch scaling internally; we only need to forward the bbox.
    bbox_for_prior: tuple | None = tuple(bbox) if bbox is not None and len(bbox) == 4 else None

    scored: list[FusedPrediction] = []
    for species, p_visual in predictions:
        audio_mult = AUDIO_BOOST if species in audio_heard else AUDIO_FLOOR
        seasonal_mult = _seasonal_multiplier(species, month)
        size_mult = size_prior.size_multiplier(species, bbox_for_prior) if bbox_for_prior else 1.0
        # Exotic / vagrant downweight: species not in the regional baseline or yard calibration
        # (e.g. Inca Dove, White-winged Dove, Harris's Sparrow, Phainopepla) receive a 20x penalty (0.05)
        # unless audio-confirmed by Haikubox.
        vagrant_mult = 1.0 if (species in audio_heard or is_regional_species(species)) else 0.05
        posterior = p_visual * audio_mult * seasonal_mult * size_mult * vagrant_mult
        scored.append(
            FusedPrediction(
                species=species,
                probability=posterior,
                audio_confirmed=species in audio_heard,
                seasonal_boost=seasonal_mult,
                size_mult=size_mult,
            )
        )

    total = sum(s.probability for s in scored)
    if total > 0:
        for s in scored:
            s.probability /= total
    elif scored:
        # Fallback to uniform distribution if all candidates collapsed to zero probability
        uniform = 1.0 / len(scored)
        for s in scored:
            s.probability = uniform
    scored.sort(key=lambda s: s.probability, reverse=True)
    return scored
