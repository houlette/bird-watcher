"""Family-level catch-all labels for when the user knows the bird's
broad type but can't ID the species.

Each family is a Species row with `is_family=True`. The member lists
below are reverse-looked-up so:
  - `species_to_families("House Sparrow")` returns ["Sparrow"]
  - `families_contains_species("Sparrow", "House Sparrow")` returns True

These mappings drive the partial-credit classifier-accuracy metric
(stats.py) and the soft-label expansion in any future retraining.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

# family common-name → list of species common-names that belong.
# Hand-curated; keep aligned with the species names the classifier
# emits and the user picks from. New families should:
#   1. Be visually distinguishable on a feeder cam (a user can say
#      "yes that's a warbler" without species-level expertise).
#   2. Cover ≥ 3 species we routinely see — otherwise the user might
#      as well pick the species directly.
FAMILY_MEMBERS: dict[str, list[str]] = {
    "Sparrow": [
        "House Sparrow",
        "Song Sparrow",
        "White-throated Sparrow",
        "Chipping Sparrow",
        "Field Sparrow",
        "Fox Sparrow",
        "American Tree Sparrow",
        "Swamp Sparrow",
        "Dark-eyed Junco",  # New World sparrow taxonomically
        "White-crowned Sparrow",
        "Savannah Sparrow",
    ],
    "Warbler": [
        "Yellow-rumped Warbler",
        "Pine Warbler",
        "Yellow Warbler",
        "Black-and-white Warbler",
        "Black-throated Blue Warbler",
        "Black-throated Green Warbler",
        "Blackburnian Warbler",
        "Common Yellowthroat",
        "Ovenbird",
        "American Redstart",
        "Magnolia Warbler",
        "Palm Warbler",
    ],
    "Woodpecker": [
        "Downy Woodpecker",
        "Hairy Woodpecker",
        "Red-bellied Woodpecker",
        "Northern Flicker",
        "Pileated Woodpecker",
        "Yellow-bellied Sapsucker",
    ],
    "Finch": [
        "House Finch",
        "Purple Finch",
        "American Goldfinch",
    ],
}


# Reverse index built once at import time.
_SPECIES_TO_FAMILIES: dict[str, list[str]] = {}
for _fam, _members in FAMILY_MEMBERS.items():
    for _sp in _members:
        _SPECIES_TO_FAMILIES.setdefault(_sp, []).append(_fam)


FAMILY_NAMES = frozenset(FAMILY_MEMBERS.keys())


def family_members(family: str) -> list[str]:
    """Return the species list for a family, or [] if unknown."""
    return FAMILY_MEMBERS.get(family, [])


def species_to_families(species: str) -> list[str]:
    """Return all families containing this species (usually 0 or 1)."""
    return _SPECIES_TO_FAMILIES.get(species, [])


def family_contains(family: str, species: str) -> bool:
    """True if `species` is one of `family`'s member species."""
    return species in FAMILY_MEMBERS.get(family, ())


def is_family_label(name: str) -> bool:
    """True if `name` is a known family-level label."""
    return name in FAMILY_NAMES


def ensure_family_species_rows(db: Session) -> int:
    """Idempotent seed: ensure a Species row exists for every family in
    FAMILY_MEMBERS, with is_family=True. Returns the count created.

    Called from db.session.init_db() so a fresh deploy gets the families
    populated without a manual step. Re-running is safe — we look up by
    common_name (which is unique).
    """
    # Late import to avoid a top-level cycle (db.models imports
    # db.session which calls into us).
    from db.models import Species

    created = 0
    for family in FAMILY_MEMBERS:
        existing = db.query(Species).filter(Species.common_name == family).one_or_none()
        if existing is None:
            db.add(Species(
                common_name=family,
                scientific_name="",  # family-level taxonomy gets messy; leave blank
                is_rare=False,
                is_family=True,
            ))
            created += 1
        elif not existing.is_family:
            # Older row from before the column existed — flip the flag.
            existing.is_family = True
    if created:
        db.commit()
    return created
