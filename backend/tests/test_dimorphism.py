"""Unit tests for plumage and sex dimorphism classifier."""
import cv2
import numpy as np
import pytest

from pipeline.dimorphism import (
    DIMORPHIC_SPECIES,
    classify_sex,
    is_dimorphic_species,
)


def test_is_dimorphic_species():
    assert is_dimorphic_species("Northern Cardinal") is True
    assert is_dimorphic_species("House Finch") is True
    assert is_dimorphic_species("Downy Woodpecker") is True
    assert is_dimorphic_species("Hairy Woodpecker") is True
    assert is_dimorphic_species("Rose-breasted Grosbeak") is True

    assert is_dimorphic_species("Blue Jay") is False
    assert is_dimorphic_species("Black-capped Chickadee") is False
    assert is_dimorphic_species("Mourning Dove") is False
    assert is_dimorphic_species(None) is False
    assert is_dimorphic_species("") is False


def test_male_cardinal_synthetic():
    # Synthetic scarlet red bird (BGR: blue=20, green=20, red=220)
    # in an image of size 100x100
    img = np.full((100, 100, 3), (20, 20, 220), dtype=np.uint8)
    res = classify_sex(img, "Northern Cardinal")
    assert res.sex == "male"
    assert res.confidence >= 0.80
    assert res.method == "cardinal_plumage"


def test_female_cardinal_synthetic():
    # Synthetic olive-buff / fawn bird (BGR: blue=60, green=110, red=135)
    # Olive/tan tones with low redness
    img = np.full((100, 100, 3), (60, 110, 135), dtype=np.uint8)
    res = classify_sex(img, "Northern Cardinal")
    assert res.sex == "female"
    assert res.confidence >= 0.75
    assert res.method == "cardinal_plumage"


def test_male_finch_synthetic():
    # Mostly gray/brown body with a red bib/cap cluster
    img = np.full((100, 100, 3), (90, 100, 110), dtype=np.uint8)
    # Draw red bib cluster in upper 50%
    cv2.circle(img, (50, 30), 15, (25, 25, 215), -1)
    res = classify_sex(img, "House Finch")
    assert res.sex == "male"
    assert res.confidence >= 0.70
    assert res.method == "finch_red_bib"


def test_female_finch_synthetic():
    # Plain grayish-brown without red
    img = np.full((100, 100, 3), (90, 100, 110), dtype=np.uint8)
    res = classify_sex(img, "House Finch")
    assert res.sex == "female"
    assert res.confidence >= 0.75
    assert res.method == "finch_brown_streaked"


def test_male_woodpecker_synthetic():
    # Black and white body with a bright red nape spot in the upper half
    img = np.full((100, 100, 3), (20, 20, 20), dtype=np.uint8)
    img[40:80, 30:70] = (220, 220, 220)
    # Red nape spot (radius 5 = ~78 pixels)
    cv2.circle(img, (50, 20), 5, (20, 20, 230), -1)
    res = classify_sex(img, "Downy Woodpecker")
    assert res.sex == "male"
    assert res.confidence >= 0.75
    assert res.method == "woodpecker_red_nape"


def test_female_woodpecker_synthetic():
    # Clean black and white, zero red
    img = np.full((100, 100, 3), (20, 20, 20), dtype=np.uint8)
    img[40:80, 30:70] = (220, 220, 220)
    res = classify_sex(img, "Downy Woodpecker")
    assert res.sex == "female"
    assert res.confidence >= 0.80
    assert res.method == "woodpecker_plain_head"


def test_monomorphic_species_returns_none():
    img = np.full((100, 100, 3), (20, 20, 220), dtype=np.uint8)
    res = classify_sex(img, "Blue Jay")
    assert res.sex is None
    assert res.confidence == 0.0
    assert res.method == "monomorphic"


def test_none_input_handled_gracefully():
    res1 = classify_sex(None, "Northern Cardinal")
    assert res1.sex is None

    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    res2 = classify_sex(img, None)
    assert res2.sex is None
