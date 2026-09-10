"""Tavern rules for The Perch & Flagon: archetypes, economy, and lore.

Pure functions, no DB and no IO, in the same spirit as pipeline.palette
(which this module leans on for plumage colour, body mass and call pitch).
Everything here is deterministic: the same detection always produces the
same patron, the same order, the same arrival line and the same payment,
so the room looks identical on a reload and two devices agree.

Three ideas carry most of the design:

  1. **A patron is a DETECTION, not a visit.** Same rule the Art page
     follows. A visit with forty detections is forty tracks, usually forty
     separate birds, and pouring one flagon for the lot of them would
     under-count the room by a wide margin.

  2. **A bird with no species label is a cloaked stranger.** Most rows in
     this database have `species_id IS NULL`, which is the classifier
     declining to commit rather than a confirmed sighting. They belong in
     the room because they are real motion, but they pay a single shilling
     and leave no name in the ledger, so the economy stays weighted toward
     birds somebody actually identified.

  3. **Rarity is local.** A species' tier comes from its share of this
     yard's own identified detections, not from a range map. A Rock Pigeon
     is common here whatever a field guide says, and the first Baltimore
     Oriole should feel rare on the evening it walks in.
"""
from __future__ import annotations

import random
from typing import Any

from pipeline.palette import species_style

# ── Archetypes ──────────────────────────────────────────────────────────
#
# `seat` names a region of the room, not a coordinate: the canvas picks the
# exact spot. `dwell_beats` is how long the patron lingers relative to the
# others, before the real visit duration adjusts it. `tip` is the flat
# bonus the archetype adds to a payment, which is where "chickadees tip
# swiftly" and "nobody hands a hawk a bill" live.

ARCHETYPES: dict[str, dict[str, Any]] = {
    "messenger": {
        "id": "messenger",
        "title": "Hurried messenger",
        "role": "Scout of the canopy roads",
        "seat": "bar",
        "dwell_beats": 1.0,
        "tip": 2,
        "blurb": "Dashes in, eats standing up, is gone before the door swings shut.",
        "drinks": ["a thimble of sap cider", "hot thistle tea", "a splash of dew wine"],
        "dishes": ["sunflower crumbles", "a thistle wafer", "cracked millet, to go"],
        "lines": [
            "No time, no time. Seed and a story, quickly.",
            "Word from the hedge road: the feeder is full.",
            "Keep the change, I am three hedges behind.",
            "Standing room is fine. I am not stopping.",
        ],
    },
    "adventurer": {
        "id": "adventurer",
        "title": "Boisterous adventurer",
        "role": "Sellsword of the suet cage",
        "seat": "booth",
        "dwell_beats": 3.2,
        "tip": 1,
        "blurb": "Takes the corner booth, orders the hearty platter, stays for hours.",
        "drinks": ["a full flagon of acorn stout", "black elderberry mead", "the strong stuff"],
        "dishes": ["the suet platter", "a whole peanut, shell on", "roast mealworm skewers"],
        "lines": [
            "The corner booth. Yes, that one. It is mine now.",
            "Another platter, and do not spare the suet.",
            "I have held this perch against three grackles. Ask anyone.",
            "Quiet down, the lot of you, I am telling a story.",
        ],
    },
    "local": {
        "id": "local",
        "title": "Regular of the house",
        "role": "Fixture of the long bench",
        "seat": "bench",
        "dwell_beats": 4.0,
        "tip": 0,
        "blurb": "Settles onto the bench, talks to nobody in particular, leaves last.",
        "drinks": ["the usual", "small beer, warmed", "a mug of barley broth"],
        "dishes": ["cracked corn", "a bowl of millet", "yesterday's bread, softened"],
        "lines": [
            "The usual, and no hurry about it.",
            "I was here before the roof was.",
            "Cold out. Warmer in. That is the whole of my news.",
            "Mind the bench, that dent is mine.",
        ],
    },
    "minstrel": {
        "id": "minstrel",
        "title": "Wandering minstrel",
        "role": "Singer for supper",
        "seat": "hearth",
        "dwell_beats": 2.6,
        "tip": 3,
        "blurb": "Sings for the room, is paid in seed, and the hearth burns brighter for it.",
        "drinks": ["spiced rosehip wine", "honey water, for the throat", "elderflower cordial"],
        "dishes": ["a supper of berries", "soft suet and apple", "whatever the kitchen spares"],
        "lines": [
            "One song, then supper. That was the bargain.",
            "This one is about a hawk and a very brave wren.",
            "Hush now. The good part is coming.",
            "I will take the hearth chair, if the house allows.",
        ],
    },
    "wanderer": {
        "id": "wanderer",
        "title": "Traveller from far water",
        "role": "Come a long way, going further",
        "seat": "hearth",
        "dwell_beats": 2.2,
        "tip": 4,
        "blurb": "Arrives damp, pays in odd coin, and speaks of ponds nobody here has seen.",
        "drinks": ["reed-water brandy", "salt tea", "something from a flask of their own"],
        "dishes": ["pondweed and grain", "a plate of duckweed greens", "fish, if the house has any"],
        "lines": [
            "You are a long way from any water. I noticed.",
            "A bed, if you have one. A dry corner, if you do not.",
            "I have flown since before light. Feed me first, questions after.",
            "Odd coin, but good coin. Weigh it if you like.",
        ],
    },
    "hunter": {
        "id": "hunter",
        "title": "Hooded mercenary",
        "role": "Nobody asks their business",
        "seat": "rafters",
        "dwell_beats": 1.6,
        "tip": 0,
        "blurb": "The room goes quiet. Takes the rafters, watches the door, pays nothing.",
        "drinks": ["water", "nothing", "whatever is nearest"],
        "dishes": ["nothing from the kitchen", "a plate, untouched", "meat, rare"],
        "lines": [
            "The room went quiet on its own. I said nothing.",
            "I will take the rafters. Carry on.",
            "I am not staying. I am waiting.",
            "Your other guests may leave whenever they like.",
        ],
    },
    "stranger": {
        "id": "stranger",
        "title": "Cloaked stranger",
        "role": "Face never quite caught",
        "seat": "shadow",
        "dwell_beats": 1.4,
        "tip": 0,
        "blurb": "The camera saw something move. The classifier would not swear to what.",
        "drinks": ["whatever is cheapest", "an unfinished cup", "nothing at all"],
        "dishes": ["seed, taken standing", "a crust", "nothing the kitchen remembers"],
        "lines": [
            "A hood, a shadow, and a coin on the bar.",
            "Gone before anyone thought to ask a name.",
            "You did not see me. Neither did the camera.",
            "Somebody was here. That much the house is sure of.",
        ],
    },
}

# Explicit assignments for the species this yard actually logs, plus the
# eastern feeder set the palette already catalogues. Anything missing falls
# through to the family words, then to body mass.
SPECIES_ARCHETYPE: dict[str, str] = {
    # This yard's regulars.
    "Mourning Dove": "local",
    "Rock Pigeon": "local",
    "House Sparrow": "local",
    "Dark-eyed Junco": "local",
    "House Finch": "local",
    "Northern Cardinal": "minstrel",
    "Gray Catbird": "minstrel",
    "American Robin": "minstrel",
    "Northern Mockingbird": "minstrel",
    "Carolina Wren": "minstrel",
    "Ovenbird": "minstrel",
    "Song Sparrow": "minstrel",
    "Common Grackle": "adventurer",
    "European Starling": "adventurer",
    "Blue Jay": "adventurer",
    "Eastern Towhee": "adventurer",
    "Black-capped Chickadee": "messenger",
    "Carolina Chickadee": "messenger",
    "Tufted Titmouse": "messenger",
    "White-breasted Nuthatch": "messenger",
    "Red-breasted Nuthatch": "messenger",
    "American Goldfinch": "messenger",
    "Downy Woodpecker": "adventurer",
    "Hairy Woodpecker": "adventurer",
    "Red-bellied Woodpecker": "adventurer",
    "Northern Flicker": "adventurer",
    "Pileated Woodpecker": "adventurer",
    "Mallard": "wanderer",
    "Wood Duck": "wanderer",
    "Herring Gull": "wanderer",
    "Dunlin": "wanderer",
    "Osprey": "hunter",
    "Cooper's Hawk": "hunter",
    # Plausible visitors the palette already knows.
    "Baltimore Oriole": "minstrel",
    "Scarlet Tanager": "minstrel",
    "Rose-breasted Grosbeak": "minstrel",
    "Indigo Bunting": "minstrel",
    "Purple Finch": "local",
    "White-throated Sparrow": "minstrel",
    "Eastern Bluebird": "minstrel",
    "Red-winged Blackbird": "adventurer",
}

FAMILY_ARCHETYPE: dict[str, str] = {
    "Sparrow": "local",
    "Dove": "local",
    "Pigeon": "local",
    "Finch": "local",
    "Warbler": "minstrel",
    "Thrush": "minstrel",
    "Wren": "minstrel",
    "Woodpecker": "adventurer",
    "Jay": "adventurer",
    "Crow": "adventurer",
    "Blackbird": "adventurer",
    "Hawk": "hunter",
    "Owl": "hunter",
    "Falcon": "hunter",
    "Eagle": "hunter",
    "Duck": "wanderer",
    "Goose": "wanderer",
    "Gull": "wanderer",
    "Heron": "wanderer",
    "Sandpiper": "wanderer",
    "Chickadee": "messenger",
    "Titmouse": "messenger",
    "Nuthatch": "messenger",
    "Kinglet": "messenger",
    "Hummingbird": "messenger",
}

UNKNOWN_BIRD_ARCHETYPE = "local"


def archetype_for(common_name: str | None, mass_g: float | None = None) -> dict[str, Any]:
    """Which sort of patron walks through the door for this species.

    Order of resolution: no name at all is a stranger, then the explicit
    table, then a family word, then body mass as a last resort so a species
    nobody has catalogued still lands somewhere sensible rather than
    defaulting every stranger onto the same bench.
    """
    if not common_name:
        return dict(ARCHETYPES["stranger"])

    if common_name in SPECIES_ARCHETYPE:
        return dict(ARCHETYPES[SPECIES_ARCHETYPE[common_name]])

    if common_name == "Unknown bird":
        return dict(ARCHETYPES[UNKNOWN_BIRD_ARCHETYPE])

    words = {w.strip(",.") for w in common_name.split()}
    for family, arch in FAMILY_ARCHETYPE.items():
        if family in words:
            return dict(ARCHETYPES[arch])

    grams = mass_g if mass_g is not None else species_style(common_name)["mass_g"]
    if grams >= 400:
        return dict(ARCHETYPES["wanderer"])
    if grams >= 55:
        return dict(ARCHETYPES["adventurer"])
    if grams <= 16:
        return dict(ARCHETYPES["messenger"])
    return dict(ARCHETYPES["local"])


# ── Rarity ──────────────────────────────────────────────────────────────
#
# Thresholds are shares of this yard's identified detections. The bands are
# deliberately wide at the bottom: with a few hundred labelled rows, one
# extra sighting moves a small share a long way, and a tier that flickers
# between visits would make the payment for the same bird jump around.

RARITY_TIERS = ("common", "regular", "uncommon", "rare")
_RARITY_THRESHOLDS = ((0.08, "common"), (0.02, "regular"), (0.005, "uncommon"))
TIER_BONUS = {"common": 0, "regular": 2, "uncommon": 5, "rare": 12}


def rarity_tier(share: float) -> str:
    """Tier for a species holding `share` of the yard's identified sightings."""
    for cutoff, tier in _RARITY_THRESHOLDS:
        if share >= cutoff:
            return tier
    return "rare"


def rarity_tiers(counts: dict[str, int]) -> dict[str, str]:
    """Tier every species at once, given yard-wide identified counts."""
    total = sum(counts.values())
    if total <= 0:
        return {name: "rare" for name in counts}
    return {name: rarity_tier(n / total) for name, n in counts.items()}


# ── Economy ─────────────────────────────────────────────────────────────
#
# Seed Shillings are earned, never granted. A patron pays a flat cover, a
# bonus for how long they stayed, a bonus for how rare they are here, the
# archetype's tip, and two for an arrival Haikubox heard as well as saw.
# Strangers are the exception and pay one flat: an unlabelled crop should
# not be able to buy the stained glass.

COVER_CHARGE = 3
STRANGER_PAYMENT = 1
AUDIO_CONFIRMED_BONUS = 2
# What the strongbox held when the user took the house on. Everything the
# feeder recorded before the tavern first opened pays into this purse, and
# the purse stops at this figure however long the archive is.
#
# The alternative, crediting the whole backlog, hands a yard with two years
# of clips every upgrade in the catalogue on the first evening and leaves
# nothing to play for. A fixed cap also means opening night looks the same
# on a database I can measure and one I cannot: a few purchases now, the
# rest earned by birds that have not landed yet. Whether 1200 is the right
# figure is a guess, and the honest place to revisit it is after a week of
# use.
FOUNDING_PURSE_CAP = 1200
# Ninety seconds of feeder time is a long visit. Past that the bonus stops
# growing, so one bird asleep on the perch cannot out-earn a busy morning.
DWELL_CAP_SECONDS = 90.0
DWELL_SECONDS_PER_SHILLING = 15.0


def patron_payment(
    *,
    archetype_id: str,
    tier: str,
    duration_seconds: float,
    audio_confirmed: bool,
) -> int:
    """Shillings one patron leaves on the table."""
    if archetype_id == "stranger":
        return STRANGER_PAYMENT

    dwell = min(max(duration_seconds, 0.0), DWELL_CAP_SECONDS)
    payment = COVER_CHARGE
    payment += int(dwell // DWELL_SECONDS_PER_SHILLING)
    payment += TIER_BONUS.get(tier, 0)
    payment += ARCHETYPES.get(archetype_id, ARCHETYPES["local"])["tip"]
    if audio_confirmed:
        payment += AUDIO_CONFIRMED_BONUS
    return payment


# ── Tavern upgrades ─────────────────────────────────────────────────────
#
# Costs ladder from an evening's takings to a season of them. `effect` is
# the flag the canvas reads; adding an upgrade here without teaching the
# canvas its flag buys the user a line of text and nothing else, so keep
# the two in step.

UPGRADES: list[dict[str, Any]] = [
    {
        "id": "tallow_candles",
        "name": "Tallow candles",
        "category": "Light",
        "cost": 40,
        "effect": "candles",
        "blurb": "Six candles down the length of the bar. They gutter when the door opens.",
    },
    {
        "id": "oak_tables",
        "name": "Oak tables",
        "category": "Furniture",
        "cost": 120,
        "effect": "tables",
        "blurb": "Trestles out, proper oak in. The bench regulars claim they cannot tell.",
    },
    {
        "id": "thistle_pantry",
        "name": "Thistle pantry",
        "category": "Kitchen",
        "cost": 150,
        "effect": "pantry",
        "blurb": "Thistle, nyjer and cracked millet in stone jars behind the bar.",
    },
    {
        "id": "brass_lanterns",
        "name": "Brass lanterns",
        "category": "Light",
        "cost": 180,
        "effect": "lanterns",
        "blurb": "Warm light that reaches the far wall, so the shadow table is a table again.",
    },
    {
        "id": "acoustic_perches",
        "name": "Acoustic perches",
        "category": "Furniture",
        "cost": 260,
        "effect": "perches",
        "blurb": "Hazel rods over the hearth. A singer on one of these carries to the road.",
    },
    {
        "id": "corner_booth",
        "name": "The corner booth",
        "category": "Furniture",
        "cost": 320,
        "effect": "booth",
        "blurb": "High-backed, curtained, and permanently spoken for by whichever jay arrives first.",
    },
    {
        "id": "sunflower_cask",
        "name": "Sunflower cask",
        "category": "Kitchen",
        "cost": 380,
        "effect": "cask",
        "blurb": "A cask of black-oil sunflower on tap. Doubles as a table after midnight.",
    },
    {
        "id": "suet_hearth",
        "name": "Miniature suet hearth",
        "category": "Light",
        "cost": 420,
        "effect": "hearth",
        "blurb": "A second small fire kept for suet alone. The woodpeckers noticed immediately.",
    },
    {
        "id": "guest_ledger",
        "name": "The guest ledger",
        "category": "House",
        "cost": 500,
        "effect": "ledger",
        "blurb": "A bound book by the door. Fills in each visitor's history and lore in the guestbook.",
    },
    {
        "id": "rafter_roost",
        "name": "Rafter roost",
        "category": "Furniture",
        "cost": 640,
        "effect": "roost",
        "blurb": "Straw and shelter up in the beams, for guests who prefer to watch the door.",
    },
    {
        "id": "stained_glass",
        "name": "Stained-glass window",
        "category": "Light",
        "cost": 900,
        "effect": "glass",
        "blurb": "Coloured light across the floorboards, which moves as the real sun does.",
    },
    {
        "id": "bard_in_residence",
        "name": "Bard in residence",
        "category": "House",
        "cost": 1200,
        "effect": "bard",
        "blurb": "A resident player who answers every singer in the room, whether or not one is here.",
    },
    {
        "id": "painted_sign",
        "name": "A painted signboard",
        "category": "House",
        "cost": 2200,
        "effect": "sign",
        "blurb": "The Perch and Flagon, in gold leaf, swinging over the door. The house is real now.",
    },
]

UPGRADES_BY_ID: dict[str, dict[str, Any]] = {u["id"]: u for u in UPGRADES}


def upgrade_cost(upgrade_id: str) -> int | None:
    up = UPGRADES_BY_ID.get(upgrade_id)
    return None if up is None else int(up["cost"])


def spent_on(unlocked: list[str] | None) -> int:
    """Total spent, recomputed from what is owned rather than stored.

    Storing a running balance invites drift: one failed write during an
    unlock and the house is either richer or poorer than its own furniture.
    Ids that are no longer in the catalogue count as zero.
    """
    return sum(upgrade_cost(u) or 0 for u in (unlocked or []))


# ── Flavour ─────────────────────────────────────────────────────────────
#
# Every generator below takes an explicit seed, normally the detection id,
# so a patron's line and order are fixed for the life of that row.

MISHAPS: list[dict[str, str]] = [
    {
        "id": "draught",
        "title": "A stray draught",
        "line": "The door swings on nothing at all and every candle on the bar lies flat.",
    },
    {
        "id": "squirrel",
        "title": "The pantry squirrel",
        "line": "Something grey is in the seed jars again. The cook is holding a broom and a grudge.",
    },
    {
        "id": "spinner",
        "title": "The spinner in the yard",
        "line": "The wind spinner catches the light, the room turns to look, and there is nobody there.",
    },
    {
        "id": "shadow",
        "title": "A shadow, only",
        "line": "A shape crosses the window. Two patrons duck. It was a cloud.",
    },
    {
        "id": "leaf",
        "title": "One determined leaf",
        "line": "A leaf blows in, circles the hearth twice, and is served before anyone notices.",
    },
    {
        "id": "glint",
        "title": "Sun on the glass",
        "line": "Light off the far window fools the whole common room for a moment.",
    },
    {
        "id": "cat",
        "title": "The neighbour's cat",
        "line": "The cat walks past the window without stopping. The room pretends not to have flinched.",
    },
    {
        "id": "rain",
        "title": "Rain at the shutters",
        "line": "The shutters rattle, the fire ducks, and the bench regulars order another round.",
    },
]

# Backstory parts. Kept short: the guestbook shows one of these per
# species, and a paragraph of invented biography under a real photograph
# reads as a claim about the bird rather than as tavern furniture.
#
# Nine of each and three sentence frames, because with a handful of
# options two species out of three opened with the same clause and the
# whole book read as one template.
_ORIGINS = [
    "came down the hedge road in a bad winter",
    "arrived on the back of a storm and stayed",
    "was born three gardens over and has never said so",
    "walked in during a power cut and was assumed to be staff",
    "followed a seed cart here and lost interest in the cart",
    "claims a house in the pines, which nobody has seen",
    "turned up owing money in two counties",
    "came for one night in a wet spring and has not left",
    "was carried in on somebody else's coat and never explained it",
]
_QUIRKS = [
    "will not sit with their back to the door",
    "counts the seed before eating any of it",
    "settles every tab in exact coin, always",
    "has a running argument with the weathervane",
    "sings only when the hearth is lit",
    "leaves by the window, out of habit",
    "insists the third table is colder than the others",
    "never takes the first cup poured",
    "keeps one eye on the rafters throughout",
]
_ERRANDS = [
    "waiting on word from the far orchard",
    "owed a favour by somebody in the rafters",
    "keeping a list of everyone who displaced them at the suet",
    "saving for passage south, allegedly",
    "looking for a nest site with a better view",
    "here for the quiet, which they will not be getting",
    "avoiding a hawk they will not name",
    "teaching a younger relative the route",
    "settling a dispute about who found the feeder first",
]


def _cap(phrase: str) -> str:
    return phrase[0].upper() + phrase[1:] if phrase else phrase


def _rng(*parts: Any) -> random.Random:
    return random.Random("|".join(str(p) for p in parts))


def arrival_line(archetype_id: str, seed: Any) -> str:
    """What the patron says on the way in."""
    arch = ARCHETYPES.get(archetype_id, ARCHETYPES["local"])
    return _rng("line", archetype_id, seed).choice(arch["lines"])


def order_for(archetype_id: str, seed: Any) -> dict[str, str]:
    """Drink and dish, fixed per detection."""
    arch = ARCHETYPES.get(archetype_id, ARCHETYPES["local"])
    rng = _rng("order", archetype_id, seed)
    return {"drink": rng.choice(arch["drinks"]), "dish": rng.choice(arch["dishes"])}


def mishap_for(seed: Any) -> dict[str, str]:
    """The tavern event a 'Not a bird' detection turns into."""
    return dict(_rng("mishap", seed).choice(MISHAPS))


def lore_for(common_name: str, archetype_id: str) -> dict[str, str]:
    """A guestbook entry for one species: a house title and a short history.

    Seeded on the species name alone, so a species reads the same way every
    time it appears and the guestbook is stable between sessions.
    """
    rng = _rng("lore", common_name)
    arch = ARCHETYPES.get(archetype_id, ARCHETYPES["local"])
    origin = rng.choice(_ORIGINS)
    quirk = rng.choice(_QUIRKS)
    errand = rng.choice(_ERRANDS)

    # Two frames, not three. A third that led with the quirk kept producing
    # sentences like "Settles every tab in exact coin, always, and was born
    # three gardens over", which no amount of extra vocabulary fixes.
    if rng.random() < 0.5:
        story = f"{common_name} {origin}. {_cap(quirk)}. Currently {errand}."
    else:
        story = f"{_cap(origin)}, or so the house was told. {_cap(quirk)}. Currently {errand}."

    return {"title": arch["title"], "role": arch["role"], "backstory": story}


def dwell_beats(archetype_id: str, duration_seconds: float) -> float:
    """How long this patron stays in the room, relative to the others.

    The archetype sets the shape and the real visit duration nudges it, so
    a dove that genuinely sat for two minutes outstays one that did not,
    without a chickadee ever becoming furniture.
    """
    arch = ARCHETYPES.get(archetype_id, ARCHETYPES["local"])
    base = float(arch["dwell_beats"])
    observed = min(max(duration_seconds, 0.0), DWELL_CAP_SECONDS) / DWELL_CAP_SECONDS
    return round(base * (0.75 + 0.5 * observed), 3)
