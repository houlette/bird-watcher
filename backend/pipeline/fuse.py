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
from pipeline import calibration
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


def _audio_species_set(db: Session, window_end: datetime) -> set[str]:
    """Common names heard by the Haikubox within the correlation window."""
    window_start = window_end - timedelta(seconds=settings.audio_correlation_window_seconds)
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
_SEASONAL_PRIORS: dict[str, dict[int, float]] = {
    "Dark-eyed Junco": {
        1: 2.0, 2: 2.0, 3: 1.5, 4: 0.5, 5: 0.1, 6: 0.1,
        7: 0.1, 8: 0.1, 9: 0.5, 10: 1.5, 11: 2.0, 12: 2.0,
    },
    "Ruby-throated Hummingbird": {
        1: 0.0, 2: 0.0, 3: 0.1, 4: 1.0, 5: 2.0, 6: 2.0,
        7: 2.0, 8: 2.0, 9: 1.5, 10: 0.3, 11: 0.0, 12: 0.0,
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
        1: 0.0, 2: 0.0, 3: 0.1, 4: 1.0, 5: 2.0, 6: 1.5,
        7: 1.0, 8: 0.5, 9: 0.2, 10: 0.0, 11: 0.0, 12: 0.0,
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


def fuse(
    predictions: Iterable[tuple[str, float]],
    *,
    db: Session,
    when: datetime | None = None,
) -> list[FusedPrediction]:
    """Re-rank visual predictions by combining with audio + seasonal priors.

    `predictions` is the top-K from the classifier as (species, probability).
    Returns a list of FusedPrediction sorted by posterior probability desc.
    """
    when = when or utcnow()
    audio_heard = _audio_species_set(db, when) if (settings.haikubox_serial and settings.haikubox_api_key) else set()
    month = when.month

    scored: list[FusedPrediction] = []
    for species, p_visual in predictions:
        audio_mult = AUDIO_BOOST if species in audio_heard else AUDIO_FLOOR
        seasonal_mult = _seasonal_multiplier(species, month)
        posterior = p_visual * audio_mult * seasonal_mult
        scored.append(
            FusedPrediction(
                species=species,
                probability=posterior,
                audio_confirmed=species in audio_heard,
                seasonal_boost=seasonal_mult,
            )
        )

    total = sum(s.probability for s in scored)
    if total > 0:
        for s in scored:
            s.probability /= total
    scored.sort(key=lambda s: s.probability, reverse=True)
    return scored
