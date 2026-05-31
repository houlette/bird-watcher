"""Read-only species directory used by the active-learning picker.

Returns two groups of species so the picker can render the yard's known
species at the top (with audio-detection counts to hint familiarity)
and a broader NA-bird list below for less-common-at-this-yard species
the user still wants to label (pigeons, raptors, vagrant species, etc.).

The 'Not a bird' and 'Unknown bird' sentinels are excluded from both
groups — the picker surfaces them separately pinned at the top.
"""
from __future__ import annotations

from fastapi import APIRouter

from db.families import FAMILY_MEMBERS, FAMILY_NAMES
from db.models import SENTINEL_LABELS
from na_birds import NA_BIRD_SPECIES
from pipeline import calibration
from pipeline.classify import NA_BACKYARD_ALLOWLIST, _normalize_for_display

router = APIRouter()


@router.get("")
async def list_species() -> dict:
    """Return picker options as two groups: yard (Haikubox-heard) and broader NA."""
    cal = calibration._load_calibration()  # noqa: SLF001 — module-internal but stable
    if cal and isinstance(cal.get("species"), dict):
        yard_items = [
            {"name": name, "total": info.get("total", 0)}
            for name, info in cal["species"].items()
            if isinstance(info, dict)
            and info.get("total", 0) >= calibration.MIN_DETECTIONS_FOR_ALLOWLIST
            and name not in SENTINEL_LABELS
        ]
        yard_items.sort(key=lambda r: r["total"], reverse=True)
        source = "calibration"
    else:
        yard_items = [
            {"name": _normalize_for_display(name), "total": 0}
            for name in NA_BACKYARD_ALLOWLIST
            if name not in SENTINEL_LABELS
        ]
        yard_items.sort(key=lambda r: r["name"])
        source = "fallback"

    # Broader NA list, minus anything already in the yard list (avoid dupes
    # in the picker UI) and minus family names (surfaced separately). Case
    # insensitive comparison on the display name.
    yard_names_norm = {r["name"].lower() for r in yard_items}
    extra_items = [
        {"name": s, "total": 0}
        for s in NA_BIRD_SPECIES
        if s.lower() not in yard_names_norm
        and s not in SENTINEL_LABELS
        and s not in FAMILY_NAMES
    ]
    extra_items.sort(key=lambda r: r["name"])

    # Family group: the catch-all labels the user picks when they know
    # roughly what it is but not the species. Member lists travel with
    # each entry so the picker can show a "e.g., House, Song, Junco…"
    # hint without re-fetching.
    families = [
        {"name": fam, "members": list(members)}
        for fam, members in FAMILY_MEMBERS.items()
    ]
    families.sort(key=lambda r: r["name"])

    return {
        "source": source,
        "species": yard_items,    # legacy shape — preserves backward compat for any caller
        "yard": yard_items,
        "extra": extra_items,
        "families": families,
    }
