"""Unit tests for the pure label-normalization helpers in pipeline.classify.

We isolate these from the rest of classify.py so the test runner doesn't
need torch/transformers — the helpers are pure-string functions.
"""
from __future__ import annotations

from pipeline.classify import (
    MODEL_TYPO_FIXES,
    _hyphen_insensitive,
    _normalize_for_display,
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
