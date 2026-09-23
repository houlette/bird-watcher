"""Plumage / sex dimorphism classifier for common feeder species.

Many frequent backyard visitors exhibit distinct sexual dimorphism in their
plumage (e.g. crimson male Northern Cardinal vs. olive-buff female, rosy-crowned
male House Finch vs. streaked brown female, red-naped male Downy Woodpecker vs.
black/white female).

This module uses deterministic HSV and Lab color space analysis to classify sex
from cropped bird images without requiring extra heavy neural network weights.
When lighting or angle makes sex ambiguous, it cleanly returns None rather than
guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

DIMORPHIC_SPECIES: frozenset[str] = frozenset({
    "Northern Cardinal",
    "House Finch",
    "House Sparrow",
    "Downy Woodpecker",
    "Hairy Woodpecker",
    "Rose-breasted Grosbeak",
})


@dataclass
class SexResult:
    sex: str | None  # "male" | "female" | None
    confidence: float = 0.0
    method: str = "none"
    details: dict[str, Any] = field(default_factory=dict)


def is_dimorphic_species(species_name: str | None) -> bool:
    """True if species has distinctive sex plumage differences supported here."""
    if not species_name:
        return False
    from pipeline.classify import _hyphen_insensitive
    norm = _hyphen_insensitive(species_name)
    return any(_hyphen_insensitive(s) == norm for s in DIMORPHIC_SPECIES)


def _classify_cardinal(img: np.ndarray) -> SexResult:
    """Northern Cardinal: Male is scarlet/crimson throughout; female is olive-buff/tan.

    Male has high red pixel fraction and elevated Lab a* values. Female is
    predominantly buff/fawn with only minor red accents on the crest or wings.
    """
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return SexResult(sex=None, confidence=0.0, method="cardinal_too_small")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Crimson red mask in HSV
    m1 = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([13, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 65, 45]), np.array([180, 255, 255]))
    red_mask = m1 | m2
    red_frac = float(np.count_nonzero(red_mask) / (h * w))

    # Olive-buff / tan mask (warm yellowish-brown typical of female body plumage)
    buff_mask = cv2.inRange(hsv, np.array([13, 30, 40]), np.array([38, 150, 200]))
    buff_frac = float(np.count_nonzero(buff_mask) / (h * w))

    a_ch = lab[:, :, 1]
    a_p75 = float(np.percentile(a_ch, 75))

    details = {
        "red_frac": round(red_frac, 4),
        "buff_frac": round(buff_frac, 4),
        "a_p75": round(a_p75, 1),
    }

    # Decisive Male: abundant red or strong red-to-buff dominance or high chromatic a*
    if red_frac >= 0.14 or (red_frac >= 0.08 and red_frac > buff_frac) or a_p75 >= 136.5:
        conf = min(0.65 + red_frac * 2.0, 0.98)
        return SexResult(sex="male", confidence=round(conf, 2), method="cardinal_plumage", details=details)

    # Decisive Female: notable buff/tan body coverage with minimal red
    if buff_frac >= 0.12 and red_frac < 0.08:
        conf = min(0.65 + buff_frac * 1.5, 0.95)
        return SexResult(sex="female", confidence=round(conf, 2), method="cardinal_plumage", details=details)

    if buff_frac > 2.0 * red_frac and red_frac < 0.07:
        return SexResult(sex="female", confidence=0.80, method="cardinal_plumage", details=details)

    return SexResult(sex=None, confidence=0.0, method="cardinal_ambiguous", details=details)


def _classify_finch(img: np.ndarray) -> SexResult:
    """House Finch: Male has rosy-red forehead, brow, bib, and rump. Female is streaked grayish-brown."""
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return SexResult(sex=None, confidence=0.0, method="finch_too_small")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red in HSV (saturation >= 50, value >= 45)
    m1 = cv2.inRange(hsv, np.array([0, 50, 45]), np.array([14, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([166, 50, 45]), np.array([180, 255, 255]))
    red_mask = m1 | m2
    total_red = int(np.count_nonzero(red_mask))
    red_frac = float(total_red / (h * w))

    # Upper 70% where the red head and bib reside
    h_split = max(int(h * 0.70), 1)
    upper_red = red_mask[:h_split, :]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(upper_red)
    areas = stats[1:, 4] if num_labels > 1 else []
    max_cluster = int(max(areas)) if len(areas) > 0 else 0

    details = {
        "red_frac": round(red_frac, 4),
        "total_red": total_red,
        "max_cluster": max_cluster,
    }

    # Male: significant clustered red bib or cap
    if max_cluster >= 120 or red_frac >= 0.025:
        conf = min(0.70 + red_frac * 3.0, 0.98)
        return SexResult(sex="male", confidence=round(conf, 2), method="finch_red_bib", details=details)

    # Female: total absence or tiny speck of red in an otherwise streaked brown bird
    if max_cluster < 60 and red_frac < 0.010:
        conf = 0.90 if max_cluster < 20 else 0.75
        return SexResult(sex="female", confidence=round(conf, 2), method="finch_brown_streaked", details=details)

    return SexResult(sex=None, confidence=0.0, method="finch_ambiguous", details=details)


def _classify_woodpecker(img: np.ndarray) -> SexResult:
    """Downy / Hairy Woodpecker: Male has bright red occipital/nape patch; female is black and white."""
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return SexResult(sex=None, confidence=0.0, method="woodpecker_too_small")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 65, 55]), np.array([14, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([166, 65, 55]), np.array([180, 255, 255]))
    red_mask = m1 | m2

    # Woodpecker red patch is on the upper half of the head
    h_split = max(int(h * 0.55), 1)
    upper_red = red_mask[:h_split, :]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(upper_red)
    areas = stats[1:, 4] if num_labels > 1 else []
    max_cluster = int(max(areas)) if len(areas) > 0 else 0
    total_upper_red = int(np.count_nonzero(upper_red))

    details = {
        "total_upper_red": total_upper_red,
        "max_cluster": max_cluster,
    }

    # Male: concentrated red nape patch
    if max_cluster >= 15:
        conf = min(0.75 + (max_cluster / 200.0), 0.98)
        return SexResult(sex="male", confidence=round(conf, 2), method="woodpecker_red_nape", details=details)

    # Female: clean black and white head with no red patch
    if max_cluster <= 3 and total_upper_red <= 5:
        return SexResult(sex="female", confidence=0.88, method="woodpecker_plain_head", details=details)

    return SexResult(sex=None, confidence=0.0, method="woodpecker_ambiguous", details=details)


def _classify_grosbeak(img: np.ndarray) -> SexResult:
    """Rose-breasted Grosbeak: Male has vivid ruby-red triangular breast patch; female is streaked brown."""
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return SexResult(sex=None, confidence=0.0, method="grosbeak_too_small")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 65, 50]), np.array([13, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([167, 65, 50]), np.array([180, 255, 255]))
    red_mask = m1 | m2
    total_red = int(np.count_nonzero(red_mask))
    red_frac = float(total_red / (h * w))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask)
    areas = stats[1:, 4] if num_labels > 1 else []
    max_cluster = int(max(areas)) if len(areas) > 0 else 0

    details = {
        "red_frac": round(red_frac, 4),
        "total_red": total_red,
        "max_cluster": max_cluster,
    }

    # Male: ruby breast patch
    if max_cluster >= 35 or red_frac >= 0.005:
        conf = min(0.75 + red_frac * 4.0, 0.98)
        return SexResult(sex="male", confidence=round(conf, 2), method="grosbeak_ruby_breast", details=details)

    # Female: streaked brown, no red
    if max_cluster <= 10 and red_frac < 0.002:
        return SexResult(sex="female", confidence=0.85, method="grosbeak_brown_streaked", details=details)

    return SexResult(sex=None, confidence=0.0, method="grosbeak_ambiguous", details=details)


def _classify_house_sparrow(img: np.ndarray) -> SexResult:
    """House Sparrow: Male has black throat bib and chestnut nape; female is plain buff-gray."""
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return SexResult(sex=None, confidence=0.0, method="house_sparrow_too_small")

    # Upper-mid chest area (15% to 75% height, 15% to 85% width)
    chest = img[int(h * 0.15) : int(h * 0.75), int(w * 0.15) : int(w * 0.85)]
    if chest.size == 0:
        return SexResult(sex=None, confidence=0.0, method="house_sparrow_crop_empty")

    hsv_chest = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)

    # Black bib: low value in HSV and low RGB channels
    dark_mask = (
        (hsv_chest[:, :, 2] < 55)
        & (chest[:, :, 0] < 65)
        & (chest[:, :, 1] < 65)
        & (chest[:, :, 2] < 70)
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask.astype(np.uint8))
    max_dark_cluster = int(max([s[cv2.CC_STAT_AREA] for i, s in enumerate(stats) if i > 0], default=0))
    dark_pct = (max_dark_cluster / (h * w)) * 100.0

    # Upper half head area for chestnut post-ocular / nape band
    head = img[0 : int(h * 0.55), :]
    hsv_head = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
    chestnut_mask = (
        (hsv_head[:, :, 0] >= 7)
        & (hsv_head[:, :, 0] <= 24)
        & (hsv_head[:, :, 1] >= 55)
        & (hsv_head[:, :, 2] >= 45)
        & (hsv_head[:, :, 2] <= 190)
    )
    num_c, _, c_stats, _ = cv2.connectedComponentsWithStats(chestnut_mask.astype(np.uint8))
    max_chestnut_cluster = int(max([s[cv2.CC_STAT_AREA] for i, s in enumerate(c_stats) if i > 0], default=0))

    details = {
        "max_dark_cluster": max_dark_cluster,
        "dark_pct": round(dark_pct, 2),
        "max_chestnut_cluster": max_chestnut_cluster,
    }

    # Male: significant black bib cluster or combination of bib + chestnut
    if max_dark_cluster >= 200 or dark_pct >= 2.2 or (max_dark_cluster >= 80 and max_chestnut_cluster >= 60):
        conf = min(0.70 + (dark_pct / 5.0), 0.98)
        return SexResult(sex="male", confidence=round(conf, 2), method="house_sparrow_black_bib", details=details)

    # Female: clean buff/gray throat with no black bib and no chestnut
    if max_dark_cluster <= 35 and dark_pct < 0.4 and max_chestnut_cluster <= 50:
        return SexResult(sex="female", confidence=0.85, method="house_sparrow_plain_buff", details=details)

    return SexResult(sex=None, confidence=0.0, method="house_sparrow_ambiguous", details=details)


def classify_sex(crop: np.ndarray | None, species_name: str | None) -> SexResult:
    """Classify the sex of a detected bird from its crop if the species is sexually dimorphic.

    Returns SexResult with sex in {"male", "female", None}.
    """
    if crop is None or not species_name:
        return SexResult(sex=None, confidence=0.0, method="no_input")

    from pipeline.classify import _hyphen_insensitive
    norm = _hyphen_insensitive(species_name)

    if norm == "northern cardinal":
        return _classify_cardinal(crop)
    elif norm == "house finch":
        return _classify_finch(crop)
    elif norm == "house sparrow":
        return _classify_house_sparrow(crop)
    elif norm in {"downy woodpecker", "hairy woodpecker"}:
        return _classify_woodpecker(crop)
    elif norm == "rose-breasted grosbeak":
        return _classify_grosbeak(crop)

    return SexResult(sex=None, confidence=0.0, method="monomorphic")
