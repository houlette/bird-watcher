"""Acoustic botany for Chrono-Chirps, the Biome page.

Pure functions, no DB and no IO, in the same spirit as pipeline.palette
(which this module leans on for call pitch and body mass) and
pipeline.tavern. Everything is deterministic on the species name, so the
garden looks identical on a reload and two devices agree.

Three ideas carry the design:

  1. **A plant is a SPECIES, not a detection.** This is the one place the
     creative pages break from the "a flight is a detection" rule, and it
     is deliberate: the Haikubox does not track individuals, it reports
     that a House Sparrow was audible in a three-second window. Seven
     hundred of those are one sparrow population singing all day, not
     seven hundred birds. So each species heard grows one plant, and the
     calls are what make it grow.

  2. **Pitch chooses the plant.** A species' `call_hz` from the palette
     picks its botanical form and its bloom hue, so the garden is laid
     out by the same axis a spectrogram is: low callers are the deep
     indigo cushion moss along the ground, high whistlers are the pale
     violet florets above them.

  3. **Vitality is call density and persistence, not loudness.** The
     design document asks for Haikubox `specSum` spectral energy here.
     That value lives behind the AppSync GraphQL API, whose credentials
     are not in this repo (see CREATIVE_PROJECTS.md), and the v2 REST
     feed this project polls returns neither energy nor a BirdNET score:
     every `confidence` in `haikubox_detections` is NULL. So `vitality()`
     is built from what the cache really holds, how often a species was
     heard and how much of the day it was heard across, and it takes an
     optional confidence so the blend is already there on the day scores
     start arriving. `ENERGY_SOURCE_*` names which input was used, and
     the page prints it rather than implying a microphone reading it
     never took.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

from pipeline.palette import species_style

# Which measurement `vitality()` actually had to work with. The API passes
# one of these through so the page can say so out loud.
ENERGY_SOURCE_CALLS = "calls"
ENERGY_SOURCE_CALLS_SCORE = "calls+score"
# Reserved for the day the AppSync client lands and real spectral energy
# replaces the density estimate.
ENERGY_SOURCE_SPECTRAL = "spectral"


# ── Botanical forms ─────────────────────────────────────────────────────
#
# Each form is an L-system the canvas expands and draws with a turtle. The
# alphabet is deliberately tiny, because BiomeCanvas has to implement all
# of it:
#
#   F  draw a segment forward        [  push position and heading
#   X  a bud, drawn as nothing       ]  pop position and heading
#   +  turn right by `angle`         L  place a leaf here
#   -  turn left by `angle`          O  place a bloom here
#
# `girth` and `reach` scale the stem thickness and the segment length
# before the species' own mass and vitality adjust them. `petals` is how
# many the bloom gets; a moss cushion raises spore capsules rather than
# flowers, so it gets the fewest.
#
# Bands are by call pitch in hertz, low to high. The boundaries are chosen
# so this yard's regulars land where the design document puts them: doves
# and pigeons in the moss, jays and grackles as woody spires, cardinal and
# catbird as climbing vines, sparrows and starlings as ferns, and the
# chickadee at 3800 Hz just inside the floret band with the warblers.

FORMS: dict[str, dict[str, Any]] = {
    "moss": {
        "id": "moss",
        "title": "Cushion moss",
        "band": "Low callers, under 1.2 kHz",
        "axiom": "F",
        "rules": {"F": "F[++FL][--FO][+FL][-FL]"},
        "angle": 34,
        "girth": 1.5,
        "reach": 0.55,
        "petals": 3,
        "blurb": "Spreads low and dense. Coos and caws settle into it.",
    },
    "spire": {
        "id": "spire",
        "title": "Woody spire",
        "band": "1.2 to 2 kHz",
        "axiom": "F",
        "rules": {"F": "FF[+FL][-FO]F"},
        "angle": 17,
        "girth": 1.35,
        "reach": 1.05,
        "petals": 4,
        "blurb": "One thick trunk and few branches. Jays and grackles.",
    },
    "vine": {
        "id": "vine",
        "title": "Climbing vine",
        "band": "2 to 3 kHz",
        "axiom": "F",
        "rules": {"F": "F[+FL]F[-FO]F"},
        "angle": 23,
        "girth": 0.85,
        "reach": 1.0,
        "petals": 6,
        "blurb": "Curls upward in loops. Whistled songs with real tune.",
    },
    "frond": {
        "id": "frond",
        "title": "Fern frond",
        "band": "3 to 3.7 kHz",
        "axiom": "X",
        "rules": {"X": "F[-X]F[+X]LX", "F": "FF"},
        "angle": 26,
        "girth": 0.7,
        "reach": 0.85,
        "petals": 5,
        "blurb": "Leaflets all the way up. Chatter, chirps and trills.",
    },
    "floret": {
        "id": "floret",
        "title": "Floret",
        "band": "Above 3.7 kHz",
        "axiom": "X",
        "rules": {"X": "F[+X]F[-X]+O", "F": "FF"},
        "angle": 20,
        "girth": 0.5,
        "reach": 0.8,
        "petals": 5,
        "blurb": "Fine stems and a bloom at every tip. Thin high whistles.",
    },
}

# Upper hertz bound of each band, matched in order. Anything above the last
# bound is a floret.
_BANDS: list[tuple[float, str]] = [
    (1200.0, "moss"),
    (2000.0, "spire"),
    (3000.0, "vine"),
    (3700.0, "frond"),
]

# The hue ramp runs across the audible band the yard actually uses, from a
# dove at roughly 400 Hz to a Blackpoll Warbler near 8 kHz. Pitch is
# perceived logarithmically, so the ramp is too: otherwise every songbird
# between 2 and 8 kHz would crowd into the top fifth of the scale.
HZ_FLOOR = 400.0
HZ_CEILING = 8000.0

# Bloom hue in degrees, low pitch to high: deep blue, through indigo and
# violet, to a pale rose. The design document asks for "deep indigo moss"
# at the bottom and "delicate blue/violet florets" at the top, so the ramp
# stays inside one half of the wheel rather than running a full rainbow,
# which keeps thirty species on screen looking like one garden.
#
# The span is wider than the two named colours need because of where the
# species actually fall: almost everything this yard hears sings between 2
# and 8 kHz, which is the top half of the ramp, and a narrower span left
# two thirds of the garden the same magenta.
BLOOM_HUE_LOW = 215.0
BLOOM_HUE_HIGH = 350.0

# Foliage runs the other way across a narrow green band, deep moss-green
# for the low callers and a yellow-green for the high ones.
FOLIAGE_HUE_LOW = 142.0
FOLIAGE_HUE_HIGH = 79.0

# How deep the L-system is expanded, by how many times the species was
# heard. Each step multiplies the string length by three or four, so five
# is both the visual ceiling (a denser plant reads as a blur) and the
# practical one.
_DEPTH_STEPS: list[tuple[int, int]] = [(3, 2), (12, 3), (60, 4)]
MAX_DEPTH = 5

# Hard ceiling on the turtle string any one plant expands to. A garden is
# twenty to thirty plants and the canvas redraws all of them while they
# grow, so the budget that matters is the total, not the single plant.
#
# The forms branch at very different rates: a cushion moss splits five
# ways per pass and passes 17,000 symbols at depth 5, while a floret
# reaches 440. Capping the string rather than the depth is what keeps
# those comparable, and it means a moss stops getting denser after depth 3
# and grows through reach, girth and bloom size instead.
MAX_SYMBOLS = 2_600


def plant_form(call_hz: float) -> dict[str, Any]:
    """The botanical form for a species' call pitch."""
    for ceiling, form_id in _BANDS:
        if call_hz < ceiling:
            return dict(FORMS[form_id])
    return dict(FORMS["floret"])


def pitch_position(call_hz: float) -> float:
    """Where a call pitch sits on the audible ramp, in [0, 1].

    Logarithmic, because an octave is an octave whether it starts at 400
    Hz or at 4 kHz, and the garden should space species the way an ear
    does rather than the way a linear axis does.
    """
    hz = min(HZ_CEILING, max(HZ_FLOOR, float(call_hz or HZ_FLOOR)))
    span = math.log2(HZ_CEILING) - math.log2(HZ_FLOOR)
    return (math.log2(hz) - math.log2(HZ_FLOOR)) / span


def depth_for(calls: int) -> int:
    """How many rewrite passes a species' call count earns."""
    for threshold, depth in _DEPTH_STEPS:
        if calls < threshold:
            return depth
    return MAX_DEPTH


def vitality(
    calls: int,
    busiest: int,
    spread: float,
    mean_confidence: float | None = None,
) -> tuple[float, str]:
    """How alive a plant looks, in [0, 1], and which inputs produced it.

    `calls` is this species' count for the window, `busiest` the largest
    count any species in the same window reached, and `spread` the share
    of the window between its first and last call. Density is taken on a
    log scale because one House Sparrow can out-call the whole rest of the
    yard forty to one, and a linear share would leave every other plant a
    seedling.

    `mean_confidence` is the BirdNET score when the API gives one. It
    never has so far, so the common return says so.
    """
    ceiling = max(1, int(busiest))
    density = math.log1p(max(0, calls)) / math.log1p(ceiling) if ceiling > 1 else 1.0
    base = 0.68 * density + 0.32 * _clamp(spread)

    if mean_confidence is None:
        return round(_clamp(base), 3), ENERGY_SOURCE_CALLS

    # BirdNET scores bunch between 0.1 and 0.9; stretch that to the full
    # range so a 0.6 reads as middling rather than as thriving.
    score = _clamp((float(mean_confidence) - 0.1) / 0.8)
    return round(_clamp(0.75 * base + 0.25 * score), 3), ENERGY_SOURCE_CALLS_SCORE


def plant(
    common_name: str,
    *,
    calls: int,
    busiest: int,
    spread: float,
    mean_confidence: float | None = None,
) -> dict[str, Any]:
    """Everything the canvas needs to grow one species, bar its timings.

    The caller adds the temporal fields (`hours`, `first_heard`,
    `last_heard`) and the cross-pollination fields, which need the
    database. This function is the part that is purely a function of the
    species and its call count.
    """
    style = species_style(common_name)
    hz = float(style["call_hz"])
    mass = float(style["mass_g"])
    form = plant_form(hz)
    energy, source = vitality(calls, busiest, spread, mean_confidence)
    t = pitch_position(hz)
    depth = _bounded_depth(form["axiom"], form["rules"], depth_for(calls))

    # A heavy bird grows a thick stem whatever it sounds like, so a
    # Mourning Dove's moss is stout and a gnatcatcher's floret is a thread.
    # The fourth root keeps a 1.1 kg gull from being ten times a wren.
    mass_girth = (min(1200.0, max(5.0, mass)) / 40.0) ** 0.25

    return {
        "species": common_name,
        "form": form["id"],
        "form_title": form["title"],
        "form_blurb": form["blurb"],
        "axiom": form["axiom"],
        "rules": dict(form["rules"]),
        "depth": depth,
        # Small deterministic wobble per species so two ferns of the same
        # depth are not the same fern. The canvas applies it as a
        # per-branch jitter seeded by `seed`.
        "angle": round(form["angle"] + _jitter(common_name, "angle") * 4.0, 2),
        "girth": round(form["girth"] * mass_girth * (0.75 + 0.45 * energy), 3),
        "reach": round(form["reach"] * (0.62 + 0.5 * energy), 3),
        "energy": energy,
        "energy_source": source,
        "pitch": round(t, 4),
        "call_hz": round(hz, 1),
        "mass_g": mass,
        "bloom": {
            "hue": round(BLOOM_HUE_LOW + (BLOOM_HUE_HIGH - BLOOM_HUE_LOW) * t, 1),
            "sat": round(0.40 + 0.26 * energy, 3),
            # Low callers bloom dark and high ones pale, which is what
            # makes the ground read as deep indigo and the canopy as
            # delicate without changing the hue ramp underneath.
            "light": round(0.42 + 0.21 * t, 3),
            "petals": form["petals"],
            "size": round(0.55 + 0.45 * energy, 3),
        },
        "foliage": {
            "hue": round(FOLIAGE_HUE_LOW + (FOLIAGE_HUE_HIGH - FOLIAGE_HUE_LOW) * t, 1),
            "sat": round(0.30 + 0.16 * energy, 3),
            "light": round(0.30 + 0.14 * t, 3),
        },
        "plumage": {
            "primary": style["primary"],
            "accent": style["accent"],
        },
        "seed": _seed(common_name),
    }


def expanded_length(axiom: str, rules: dict[str, str], depth: int) -> int:
    """Length the client's L-system string will reach, without building it.

    The canvas caps its own expansion, but the API caps `depth` first so a
    species heard ten thousand times in a day cannot hand the browser a
    megabyte of turtle commands. Counting symbols is cheap; concatenating
    them is not.
    """
    counts: dict[str, int] = {}
    for ch in axiom:
        counts[ch] = counts.get(ch, 0) + 1
    for _ in range(max(0, depth)):
        nxt: dict[str, int] = {}
        for ch, n in counts.items():
            for out in rules.get(ch, ch):
                nxt[out] = nxt.get(out, 0) + n
        counts = nxt
    return sum(counts.values())


def _bounded_depth(axiom: str, rules: dict[str, str], depth: int) -> int:
    """The deepest expansion at or below `depth` that stays under the cap."""
    while depth > 1 and expanded_length(axiom, rules, depth) > MAX_SYMBOLS:
        depth -= 1
    return depth


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _seed(common_name: str) -> int:
    """A stable 31-bit seed per species, for the canvas's own jitter."""
    return int.from_bytes(hashlib.md5(common_name.encode("utf-8")).digest()[:4], "big") >> 1


def _jitter(common_name: str, salt: str) -> float:
    """A stable value in [-1, 1] per species and purpose."""
    digest = hashlib.md5(f"{common_name}:{salt}".encode("utf-8")).digest()
    return digest[0] / 127.5 - 1.0
