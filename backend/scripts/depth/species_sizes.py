"""Cornell-published mean body lengths (cm) for backyard species.

Hand-curated from allaboutbirds.org for the species we actually see at
this feeder (top ~30 most-corrected on the live DB as of 2026-05-31).
Where Cornell gives a range, we use the midpoint.

Species not in this table return `None` from `expected_length_cm()` and
are excluded from the rank-correlation sanity check, but they still
appear in the per-species distribution plots.

The validation script uses these as a *reference truth* against which
the depth-derived size proxy is benchmarked. If our proxy's species
ranking correlates with this table (Spearman ≥ 0.6), the proxy is at
least directionally right.
"""
from __future__ import annotations

# species_common_name → length in cm (Cornell midpoint)
SPECIES_LENGTH_CM: dict[str, float] = {
    # Big — pigeons, doves, raptors, corvids
    "Cooper's Hawk": 42.0,        # 37–47
    "Red-tailed Hawk": 53.5,      # 45–65
    "Sharp-shinned Hawk": 28.0,   # 24–34
    "American Crow": 45.0,        # 40–53
    "Common Grackle": 31.0,       # 28–34
    "Rock Pigeon": 32.0,          # 29–36
    "Mourning Dove": 28.0,        # 23–34
    "Northern Flicker": 31.0,     # 28–36
    "Blue Jay": 28.0,             # 25–30
    "European Starling": 21.5,    # 20–23
    "Northern Mockingbird": 24.5, # 22–28
    "Gray Catbird": 22.5,         # 21–24
    "American Robin": 24.5,       # 23–28
    "Brown Thrasher": 28.0,       # 23–30
    "Northern Cardinal": 22.0,    # 21–23
    # Medium — most songbirds
    "Red-bellied Woodpecker": 24.0,  # 23–26
    "Downy Woodpecker": 16.0,        # 14–17
    "Hairy Woodpecker": 23.0,        # 18–26
    "House Finch": 14.0,             # 13–15
    "Purple Finch": 14.5,            # 12–16
    "American Goldfinch": 12.5,      # 11–14
    "Tufted Titmouse": 16.0,         # 14–17
    "Black-capped Chickadee": 13.5,  # 12–15
    "Carolina Chickadee": 12.5,      # 11–13
    "White-breasted Nuthatch": 14.5, # 13–15
    "Red-breasted Nuthatch": 11.0,   # 11
    "Song Sparrow": 15.0,            # 12–17
    "White-throated Sparrow": 17.0,  # 16–18
    "Dark-eyed Junco": 15.5,         # 14–17
    "House Sparrow": 15.5,           # 14–17
    "Chipping Sparrow": 14.0,        # 12–15
    "Carolina Wren": 13.5,           # 12–14
    "House Wren": 12.0,              # 11–13
    # Small — wood-warblers, kinglets, hummingbirds
    "Yellow-rumped Warbler": 13.5,   # 13–14
    "Common Yellowthroat": 12.0,
    "Ovenbird": 14.5,
    "Pine Warbler": 13.5,
    "Yellow Warbler": 13.0,
    "Ruby-throated Hummingbird": 8.5,  # 7–9
    "Ruby-crowned Kinglet": 10.5,
    "Golden-crowned Kinglet": 9.5,
}


def expected_length_cm(species: str) -> float | None:
    """Lookup; returns None for species not in the curated table."""
    return SPECIES_LENGTH_CM.get(species)
