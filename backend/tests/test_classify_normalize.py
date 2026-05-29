"""Unit tests for the pure helpers in pipeline.classify.

The string-normalization helpers and the lighting pre-processor are all
pure CPU/numpy/OpenCV functions — they don't touch torch or transformers,
so we can test them without the heavy ML deps.
"""
from __future__ import annotations

import numpy as np

from pipeline.classify import (
    MODEL_TYPO_FIXES,
    _hyphen_insensitive,
    _normalize_for_display,
    _preprocess_for_classifier,
)


# ────────────────────────────────────────────────────────────────────────
# _hyphen_insensitive
# ────────────────────────────────────────────────────────────────────────

def test_hyphen_insensitive_handles_hyphens_and_caps():
    """The original use case: the model writes labels with inconsistent
    hyphenation and casing."""
    assert _hyphen_insensitive("WHITE-THROATED SPARROW") == \
           _hyphen_insensitive("White-throated Sparrow") == \
           _hyphen_insensitive("WHITE THROATED SPARROW")


def test_hyphen_insensitive_strips_apostrophes():
    """gpiosenka writes 'ANNAS HUMMINGBIRD' for what eBird/Haikubox call
    'Anna's Hummingbird'. After stripping, both normalize the same way."""
    assert _hyphen_insensitive("Anna's Hummingbird") == "annas hummingbird"
    assert _hyphen_insensitive("ANNAS HUMMINGBIRD") == "annas hummingbird"
    assert _hyphen_insensitive("Anna's Hummingbird") == _hyphen_insensitive("ANNAS HUMMINGBIRD")


def test_hyphen_insensitive_collapses_whitespace_runs():
    """Tabs / double-spaces from copy-paste shouldn't break matching."""
    assert _hyphen_insensitive("Northern   Cardinal") == "northern cardinal"
    assert _hyphen_insensitive("Northern\tCardinal") == "northern cardinal"


# ────────────────────────────────────────────────────────────────────────
# _normalize_for_display + MODEL_TYPO_FIXES
# ────────────────────────────────────────────────────────────────────────

def test_normalize_for_display_compound_hyphens():
    """The general purpose: title-case + restore compound-adjective hyphens."""
    assert _normalize_for_display("DARK EYED JUNCO") == "Dark-eyed Junco"
    assert _normalize_for_display("WHITE THROATED SPARROW") == "White-throated Sparrow"


def test_normalize_for_display_overrides_model_typos():
    """A small curated table corrects the gpiosenka dataset's hard-coded
    typos so the user sees the right common name in the feed."""
    assert _normalize_for_display("BLACKBURNIAM WARBLER") == "Blackburnian Warbler"
    assert _normalize_for_display("VERMILION FLYCATHER") == "Vermilion Flycatcher"
    assert _normalize_for_display("EASTERN TOWEE") == "Eastern Towhee"
    assert _normalize_for_display("BROWN CREPPER") == "Brown Creeper"


def test_model_typo_fixes_covers_all_known_dataset_typos():
    """Sanity check that the dict isn't accidentally emptied."""
    assert len(MODEL_TYPO_FIXES) >= 4
    # And the values are reasonable common-name spellings.
    for typo, canonical in MODEL_TYPO_FIXES.items():
        assert typo == typo.upper(), "Keys should be the model's ALL-CAPS form"
        assert canonical != typo, "Values should differ from the typo"
        assert canonical[0].isupper(), "Values should be Title Case"


# ────────────────────────────────────────────────────────────────────────
# CLAHE pre-processing (_preprocess_for_classifier)
# ────────────────────────────────────────────────────────────────────────

def test_preprocess_preserves_shape_and_dtype():
    """Whatever lighting we boost, the output dimensions and dtype must
    match the input — downstream code does crop_for_model[:, :, ::-1]."""
    img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
    out = _preprocess_for_classifier(img)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_preprocess_boosts_contrast_on_low_dynamic_range_input():
    """CLAHE's job is to widen the histogram on a flat/dim crop."""
    # A "dim" crop: gray with very narrow dynamic range.
    img = np.full((100, 100, 3), 80, dtype=np.uint8)
    img[20:80, 20:80] = 90   # a barely-visible patch
    out = _preprocess_for_classifier(img)
    # The processed image should have at least as much spread as the input
    # (and likely more) on the L channel.
    import cv2 as _cv2
    in_l = _cv2.cvtColor(img, _cv2.COLOR_BGR2LAB)[:, :, 0]
    out_l = _cv2.cvtColor(out, _cv2.COLOR_BGR2LAB)[:, :, 0]
    assert out_l.std() >= in_l.std()


def test_preprocess_does_not_shift_neutral_color():
    """CLAHE operates only on L; running on a uniform gray crop shouldn't
    introduce color cast (a/b channels must stay neutral)."""
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    out = _preprocess_for_classifier(img)
    # Check that the a, b channels stay near 128 (LAB neutral).
    import cv2 as _cv2
    a, b = _cv2.cvtColor(out, _cv2.COLOR_BGR2LAB)[:, :, 1:3].transpose(2, 0, 1)
    assert abs(int(a.mean()) - 128) <= 1
    assert abs(int(b.mean()) - 128) <= 1
