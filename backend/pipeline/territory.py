"""Zone control and pecking order for Territory: Feeder Wars.

Pure functions plus one cached JSON load, in the same spirit as
pipeline.palette, pipeline.tavern and pipeline.biome. Everything is
deterministic, so the same day always scores the same way.

Four ideas carry the design, and the third is the one that constrains
everything else:

  1. **A contender is a DETECTION, not a visit.** Same rule the Art and
     Tavern pages follow. A visit with four detections is four tracks,
     usually four birds, and the whole point here is which of them was
     where.

  2. **Zones are places in this yard, not quadrants.** The four hot spots
     come from where birds really land: the hanging cage, the pedestal
     dish, the mulch bed on the ground, and the fence rail on the right.
     They match the contours in `data/heatmaps/location_heatmap.png`.
     A bird outside every zone is in the wider yard and holds nothing.

  3. **Occupancy is cheap; displacement is expensive.** Saying a bird was
     in a zone needs one bbox, which every row has. Saying bird B pushed
     bird A off a perch needs both tracks on a shared clock, which needs
     `Detection.track_frames`. That column was added when this page was
     built, so it is NULL on every row captured before then: those visits
     can be scored for occupancy and can never yield a displacement.
     `dwell()` and `displacements()` both report which case they were in
     rather than quietly averaging the two together.

  4. **An unidentified bird holds ground but joins no faction.** Most rows
     in this database have `species_id IS NULL`. They are real motion
     sitting on a real perch, so they count toward a zone being busy, but
     nothing can be said about who won, so they are tallied as unclaimed
     rather than assigned to somebody.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from pipeline.palette import FRAME_H, FRAME_W, species_style

log = logging.getLogger(__name__)

# Frames are sampled at 3 fps by pipeline.frames.extract_frames, so one
# step of `track_frames` is a third of a second. If that target ever
# changes, this constant is what re-times every dwell and displacement on
# rows captured after the change, and old rows will be quietly wrong.
FRAME_SECONDS = 1.0 / 3.0

# How long after a newcomer lands the sitting bird has to leave for it to
# count as being pushed off. The design document says 1.0 s; at 3 fps that
# is three frames.
DISPLACE_WINDOW_FRAMES = 3

# A track has to be in a zone for at least this many frames before it
# counts as holding it, rather than merely passing through the airspace on
# the way somewhere else.
MIN_HOLD_FRAMES = 2

# Where a generated zone map lives, if one has been calibrated. Same
# directory and same treatment as yard_priors.json: absent is normal, and
# the built-in map below is the documented fallback.
ZONES_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration" / "zones.json"


# ── The yard ────────────────────────────────────────────────────────────
#
# Centres and radii are fractions of the 4K frame, read off the 25/50/75%
# density contours in the location heatmap. x is a fraction of the frame
# width and y a fraction of its height, the same convention
# pipeline.palette normalises flight paths with.
#
# One consequence worth being explicit about: because the frame is 16:9, a
# radius that is equal in both normalised axes draws an ellipse on screen,
# about twice as wide as it is tall. That suits a feeder, where birds
# spread sideways along a tray or a rail far more than they stack
# vertically, and it matches the shape of the contours themselves.
#
# Assignment is nearest-centre-within-radius, which makes the map a
# Voronoi partition with a cap on how far a zone reaches. Overlapping
# radii are fine and expected: the dish and the ground bed genuinely do
# blend into each other from this camera angle.

DEFAULT_ZONES: list[dict[str, Any]] = [
    {
        "id": "cage",
        "name": "The Hanging Cage",
        "blurb": "The feeder on the shepherd's hook, left of frame. Hard to reach and easy to defend.",
        "x": 0.471,
        "y": 0.259,
        "radius": 0.095,
    },
    {
        "id": "dish",
        "name": "The Dish",
        "blurb": "The pedestal tray in the middle. The busiest ground in the yard and the most fought over.",
        "x": 0.701,
        "y": 0.442,
        "radius": 0.115,
    },
    {
        "id": "ground",
        "name": "The Drop Zone",
        "blurb": "The mulch bed below the feeders, where everything spilled ends up. Wide open and hard to hold.",
        "x": 0.637,
        "y": 0.807,
        "radius": 0.165,
    },
    {
        "id": "rail",
        "name": "The Rail",
        "blurb": "The fence rail on the right. A staging post more than a table.",
        "x": 0.944,
        "y": 0.384,
        "radius": 0.075,
    },
]

_zones_cache: tuple[float, list[dict[str, Any]]] | None = None


def load_zones() -> tuple[list[dict[str, Any]], str]:
    """The zone map and where it came from.

    Returns `(zones, source)` where source is "calibrated" if a generated
    `zones.json` was read and "built-in" if the map above was used. The
    caller passes that through to the page, because a built-in map is a
    claim about where this specific feeder stands, and it stops being true
    the day somebody moves the shepherd's hook.

    Cached on the file's mtime, the same way pipeline.calibration handles
    yard_priors.json, so a recalibration is picked up without a restart.
    """
    global _zones_cache
    try:
        mtime = ZONES_PATH.stat().st_mtime
    except OSError:
        return [dict(z) for z in DEFAULT_ZONES], "built-in"

    if _zones_cache is not None and _zones_cache[0] == mtime:
        return [dict(z) for z in _zones_cache[1]], "calibrated"

    try:
        raw = json.loads(ZONES_PATH.read_text())
        zones = [z for z in raw.get("zones", []) if _well_formed(z)]
    except (OSError, ValueError) as exc:
        log.warning("zones.json unreadable, falling back to the built-in map: %s", exc)
        return [dict(z) for z in DEFAULT_ZONES], "built-in"

    if not zones:
        return [dict(z) for z in DEFAULT_ZONES], "built-in"
    _zones_cache = (mtime, zones)
    return [dict(z) for z in zones], "calibrated"


def reset_zone_cache_for_tests() -> None:
    global _zones_cache
    _zones_cache = None


def _well_formed(zone: Any) -> bool:
    return (
        isinstance(zone, dict)
        and isinstance(zone.get("id"), str)
        and all(isinstance(zone.get(k), (int, float)) for k in ("x", "y", "radius"))
    )


def zone_for(bbox: Any, zones: list[dict[str, Any]]) -> str | None:
    """Which zone a bbox sits in, or None for the wider yard.

    Measured from the box's centre, in normalised frame coordinates. See
    the note above DEFAULT_ZONES: a zone is an ellipse on screen, wider
    than it is tall, which is the shape a feeder's traffic actually makes.
    """
    centre = _centre(bbox)
    if centre is None:
        return None
    cx, cy = centre

    best: str | None = None
    best_d = math.inf
    for z in zones:
        d = math.hypot(cx - z["x"], cy - z["y"])
        if d <= z["radius"] and d < best_d:
            best_d = d
            best = z["id"]
    return best


def _centre(bbox: Any) -> tuple[float, float] | None:
    """Box centre as a fraction of the frame, x by width and y by height."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return (x + w / 2) / FRAME_W, (y + h / 2) / FRAME_H


# ── Factions ────────────────────────────────────────────────────────────
#
# Grouped for the turf war rather than for a taxonomist. The roadmap's
# four (Jays, Finches, Picidae, Paridae) are all here, plus three this
# yard forced: doves and pigeons are the two most-logged birds on the
# camera and would otherwise be unaligned, the mimids and thrushes are the
# next block down, and the cardinal is top-five here on its own.
#
# `weight` is the faction's shoving power, used only to describe a
# matchup; who actually won a perch comes from the footage, never from
# this number.

FACTIONS: dict[str, dict[str, Any]] = {
    "syndicate": {
        "id": "syndicate",
        "name": "The Jay Syndicate",
        "style": "Heavyweight aggressors",
        "blurb": "Corvids, grackles and starlings. Arrive loudly, take the best seat, leave when finished.",
        "color": "#2a6fb8",
        "weight": 5,
        "members": [
            "Blue Jay", "American Crow", "Fish Crow", "Common Grackle",
            "European Starling", "Red-winged Blackbird", "Brown-headed Cowbird",
            "Boat-tailed Grackle", "Common Raven",
        ],
    },
    "order": {
        "id": "order",
        "name": "The Columbid Order",
        "style": "Immovable bulk",
        "blurb": "Doves and pigeons. They do not fight for ground so much as occupy it until something larger arrives.",
        "color": "#a08b76",
        "weight": 4,
        "members": ["Mourning Dove", "Rock Pigeon", "Eurasian Collared-Dove", "White-winged Dove"],
    },
    "court": {
        "id": "court",
        "name": "The Cardinal Court",
        "style": "Bold soloists",
        "blurb": "Cardinals, grosbeaks and tanagers. Heavy enough to hold a perch, too proud to share one.",
        "color": "#c43b32",
        "weight": 3,
        "members": [
            "Northern Cardinal", "Rose-breasted Grosbeak", "Scarlet Tanager",
            "Summer Tanager", "Indigo Bunting", "Baltimore Oriole", "Orchard Oriole",
        ],
    },
    "chorus": {
        "id": "chorus",
        "name": "The Mockers' Chorus",
        "style": "Watchful middleweights",
        "blurb": "Mimids and thrushes. They work the edges, take the ground, and complain about everything.",
        "color": "#7b7f86",
        "weight": 3,
        "members": [
            "Gray Catbird", "Northern Mockingbird", "American Robin", "Wood Thrush",
            "Hermit Thrush", "Brown Thrasher", "Eastern Bluebird", "Cedar Waxwing",
            "Ovenbird", "Veery",
        ],
    },
    "guild": {
        "id": "guild",
        "name": "The Picidae Guild",
        "style": "Suet specialists",
        "blurb": "Woodpeckers and nuthatches. Vertical approach, single-minded, gone before anyone objects.",
        "color": "#b32428",
        "weight": 3,
        "members": [
            "Downy Woodpecker", "Hairy Woodpecker", "Red-bellied Woodpecker",
            "Northern Flicker", "Pileated Woodpecker", "Yellow-bellied Sapsucker",
            "White-breasted Nuthatch", "Red-breasted Nuthatch", "Brown Creeper",
        ],
    },
    "coalition": {
        "id": "coalition",
        "name": "The Finch Coalition",
        "style": "High-frequency swarms",
        "blurb": "Finches, sparrows and juncos. No individual wins anything; the flock wins by arithmetic.",
        "color": "#ba373e",
        "weight": 2,
        "members": [
            "House Sparrow", "Song Sparrow", "White-throated Sparrow", "Chipping Sparrow",
            "Field Sparrow", "Fox Sparrow", "American Tree Sparrow", "Swamp Sparrow",
            "White-crowned Sparrow", "Savannah Sparrow", "Dark-eyed Junco",
            "House Finch", "Purple Finch", "American Goldfinch", "Eastern Towhee",
        ],
    },
    "outriders": {
        "id": "outriders",
        "name": "The Outriders",
        "style": "Not feeder birds at all",
        "blurb": "Raptors, waterfowl and shorebirds. They are passing through, and everything else leaves while they do.",
        "color": "#5c6675",
        "weight": 6,
        "members": [
            "Cooper's Hawk", "Sharp-shinned Hawk", "Red-tailed Hawk", "Broad-winged Hawk",
            "Merlin", "American Kestrel", "Osprey", "Great Blue Heron",
            "Mallard", "Wood Duck", "Canada Goose", "Herring Gull", "Ring-billed Gull",
            "Dunlin", "Killdeer", "Ring-Necked Pheasant", "Wild Turkey", "Common Nighthawk",
            "Belted Kingfisher",
        ],
    },
    "scouts": {
        "id": "scouts",
        "name": "The Paridae Scouts",
        "style": "Darting skirmishers",
        "blurb": "Chickadees, titmice and wrens. In and out between everyone else's shifts, never contesting anything.",
        "color": "#5b656c",
        "weight": 1,
        "members": [
            "Black-capped Chickadee", "Carolina Chickadee", "Tufted Titmouse",
            "Carolina Wren", "House Wren", "Winter Wren", "Blue-gray Gnatcatcher",
            "Ruby-crowned Kinglet", "Golden-crowned Kinglet", "Chimney Swift",
            "Barn Swallow",
        ],
    },
}

# Family-level labels ("Sparrow", "Woodpecker") are real rows in the
# species table, so they need somewhere to stand. Matched as a whole word,
# the same way pipeline.palette matches its family styles.
FAMILY_FACTIONS: dict[str, str] = {
    "Sparrow": "coalition",
    "Finch": "coalition",
    "Woodpecker": "guild",
    "Dove": "order",
    "Warbler": "scouts",
    "Hawk": "outriders",
    "Gull": "outriders",
}

_MEMBER_INDEX: dict[str, str] = {
    member: faction["id"]
    for faction in FACTIONS.values()
    for member in faction["members"]
}


def faction_for(common_name: str | None) -> str | None:
    """Which faction a species fights for, or None if nothing can be said.

    None means exactly that: the bird was not identified, or it was a
    sentinel, so it holds ground without holding it for anybody. Callers
    must tally those as unclaimed rather than inventing an allegiance.

    An uncatalogued real species falls back on body mass from the palette,
    because a stranger that weighs as much as a grackle will behave like
    one at a feeder, and the alternative is a growing pile of birds that
    belong to nobody.
    """
    if not common_name:
        return None

    hit = _MEMBER_INDEX.get(common_name)
    if hit:
        return hit

    words = {w.strip(",.") for w in common_name.split()}
    for family, faction_id in FAMILY_FACTIONS.items():
        if family in words:
            return faction_id

    # Every member of the Syndicate is named above, so the weight ladder
    # below never routes anything there. A stranger heavier than a pigeon
    # is far likelier to be a hawk passing through than a grackle.
    mass = float(species_style(common_name)["mass_g"])
    if mass >= 250:
        return "outriders"
    if mass >= 60:
        return "chorus"
    if mass >= 25:
        return "court"
    return "scouts"


# ── Holding ground ──────────────────────────────────────────────────────


def zone_spans(
    track_bboxes: Any,
    track_frames: Any,
    zones: list[dict[str, Any]],
) -> dict[str, tuple[int, int, int]]:
    """Per zone, the `(first_frame, last_frame, frames_present)` of a track.

    Needs both columns. Without `track_frames` there is no clock, so this
    returns nothing and the caller falls back to `dwell()` plus the best
    frame's zone, which is enough for occupancy and not enough for a
    pecking order.
    """
    if not isinstance(track_bboxes, list) or not isinstance(track_frames, list):
        return {}
    if len(track_bboxes) != len(track_frames) or not track_bboxes:
        return {}

    spans: dict[str, list[int]] = {}
    for bbox, frame in zip(track_bboxes, track_frames):
        if not isinstance(frame, (int, float)):
            continue
        zid = zone_for(bbox, zones)
        if zid is None:
            continue
        spans.setdefault(zid, []).append(int(frame))

    out: dict[str, tuple[int, int, int]] = {}
    for zid, frames in spans.items():
        if len(frames) < MIN_HOLD_FRAMES:
            continue
        out[zid] = (min(frames), max(frames), len(frames))
    return out


def dwell(track_bboxes: Any, track_frames: Any) -> tuple[float, str]:
    """How long a track was on camera, in seconds, and how well it is known.

    Returns `(seconds, kind)`:

      "measured"  — from `track_frames`, so it is the real span between the
                    first and last frame the bird was seen in, gaps and
                    all.
      "estimated" — from the number of boxes alone, which undercounts any
                    track the IoU matcher briefly lost. Better than
                    nothing and worse than the above.
      "unknown"   — a single-frame row with no track history at all, which
                    is most of what was captured before per-frame tracking
                    landed.
    """
    frames_ok = (
        isinstance(track_frames, list)
        and isinstance(track_bboxes, list)
        and len(track_frames) == len(track_bboxes)
        and len(track_frames) >= 2
    )
    if frames_ok:
        nums = [float(f) for f in track_frames if isinstance(f, (int, float))]
        if len(nums) >= 2:
            return round((max(nums) - min(nums) + 1) * FRAME_SECONDS, 2), "measured"

    if isinstance(track_bboxes, list) and len(track_bboxes) >= 2:
        return round(len(track_bboxes) * FRAME_SECONDS, 2), "estimated"

    return 0.0, "unknown"


def displacements(
    contenders: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Who pushed whom off which perch, within one visit.

    `contenders` is one dict per detection in the visit, each carrying at
    least `detection_id`, `species`, `track_bboxes` and `track_frames`.

    The rule, which is the design document's rule made precise: bird A is
    displaced by bird B in zone Z when all four hold.

      1. Both were in Z for at least `MIN_HOLD_FRAMES`.
      2. A got there first.
      3. They overlapped, so A was still in Z when B arrived.
      4. A's last frame in Z falls within `DISPLACE_WINDOW_FRAMES` of B's
         arrival, which at 3 fps is the one second the design asks for.

    Anything weaker than that is two birds sharing a feeder, which happens
    constantly and means nothing. Tracks missing `track_frames` cannot
    satisfy the rule at all and are silently skipped, which is why the
    caller has to report how many visits were even eligible.
    """
    prepared = []
    for c in contenders:
        spans = zone_spans(c.get("track_bboxes"), c.get("track_frames"), zones)
        if spans:
            prepared.append((c, spans))

    events: list[dict[str, Any]] = []
    for loser, loser_spans in prepared:
        for winner, winner_spans in prepared:
            if loser is winner:
                continue
            for zid, (l_first, l_last, _) in loser_spans.items():
                span = winner_spans.get(zid)
                if span is None:
                    continue
                w_first, _w_last, _ = span
                if w_first <= l_first:
                    continue  # the winner was there first; nobody was pushed
                if l_last < w_first:
                    continue  # the loser had already gone
                if l_last - w_first > DISPLACE_WINDOW_FRAMES:
                    continue  # the loser stayed on, so this was sharing
                events.append(
                    {
                        "zone": zid,
                        "winner_detection_id": winner.get("detection_id"),
                        "winner": winner.get("species"),
                        "winner_faction": faction_for(winner.get("species")),
                        "loser_detection_id": loser.get("detection_id"),
                        "loser": loser.get("species"),
                        "loser_faction": faction_for(loser.get("species")),
                        "at_frame": w_first,
                        "at_seconds": round(w_first * FRAME_SECONDS, 2),
                        "held_seconds": round((l_last - l_first + 1) * FRAME_SECONDS, 2),
                    }
                )

    events.sort(key=lambda e: (e["at_frame"], str(e["zone"])))
    return events
