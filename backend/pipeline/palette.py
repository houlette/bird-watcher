"""Plumage palettes and flight-path synthesis for the /api/art visualiser.

Two jobs, both pure functions with no DB or IO:

  1. `species_style()` maps a common name to a plumage colour, a rough body
     mass, and a nominal call frequency. The colours are what the Art page
     draws each flightline in, so they need to be recognisable at a glance
     and to survive being laid over both the Sage (light) and Twilight
     (dark) backgrounds.

  2. `flight_points()` turns a detection's bounding boxes into a normalised
     path the canvas can stroke. Detections that carry `track_bboxes` get
     their REAL per-frame path; the rest get a synthesised arrival-perch-
     departure arc. See that function's docstring for why the distinction
     matters and how callers should surface it.

Both are deterministic: the same detection always produces the same colour
and the same path, so the artwork is stable across reloads and the export
matches what was on screen.
"""
from __future__ import annotations

import colorsys
import hashlib
import math
import random
from typing import Any

# The Reolink records 4K, and every bbox in the DB is in full-frame pixel
# coordinates at this size (see Detection.bbox). Paths are normalised
# against these so the canvas can draw at any display size.
FRAME_W = 3840
FRAME_H = 2160

# How many points a returned path carries. Enough for a smooth curve after
# the client resamples it, small enough that 200 flights stay well under a
# megabyte of JSON.
TRACKED_SAMPLES = 32
APPROACH_SAMPLES = 10
DWELL_SAMPLES = 14
DEPART_SAMPLES = 8


# ── Plumage palette ─────────────────────────────────────────────────────
#
# `primary` is the stroke colour, `accent` a contrasting second plumage
# note (a mask, an epaulet, a belly) used for highlights, `mass_g` a
# typical adult body mass driving line weight and node size, `call_hz` the
# rough centre of the species' vocal range, which the optional arrival
# chimes are tuned to.
#
# The list leads with the species this yard actually logs — Mourning Dove,
# Rock Pigeon, Northern Cardinal, Gray Catbird and House Sparrow are the
# top five by a wide margin — and then covers the plausible eastern-NA
# feeder set so a first Baltimore Oriole doesn't fall back to a hash colour.
SPECIES_STYLES: dict[str, dict[str, Any]] = {
    # ── This yard's regulars ────────────────────────────────────────────
    "Mourning Dove": {"primary": "#a08b76", "accent": "#63584e", "call_hz": 480, "mass_g": 120.0},
    "Rock Pigeon": {"primary": "#6b7a8c", "accent": "#2e7d6b", "call_hz": 400, "mass_g": 300.0},
    "Northern Cardinal": {"primary": "#c43b32", "accent": "#2b1e1e", "call_hz": 2200, "mass_g": 42.0},
    "Gray Catbird": {"primary": "#7b7f86", "accent": "#8a5a4a", "call_hz": 2900, "mass_g": 37.0},
    "House Sparrow": {"primary": "#8d7355", "accent": "#3f3a34", "call_hz": 3300, "mass_g": 28.0},
    "Common Grackle": {"primary": "#3a4472", "accent": "#17423d", "call_hz": 1500, "mass_g": 115.0},
    "European Starling": {"primary": "#46586a", "accent": "#c9a96e", "call_hz": 3100, "mass_g": 82.0},
    "American Robin": {"primary": "#b85427", "accent": "#383f38", "call_hz": 2300, "mass_g": 77.0},
    "House Finch": {"primary": "#ba373e", "accent": "#7a6b5c", "call_hz": 3600, "mass_g": 20.0},
    "Dark-eyed Junco": {"primary": "#5a646b", "accent": "#f0ede6", "call_hz": 4600, "mass_g": 19.0},
    "Northern Mockingbird": {"primary": "#93999f", "accent": "#e8e6e0", "call_hz": 3400, "mass_g": 49.0},
    "Blue Jay": {"primary": "#2a6fb8", "accent": "#ffffff", "call_hz": 1600, "mass_g": 85.0},
    "Black-capped Chickadee": {"primary": "#5b656c", "accent": "#f4f1ea", "call_hz": 3800, "mass_g": 11.5},
    "Downy Woodpecker": {"primary": "#b32428", "accent": "#222222", "call_hz": 3400, "mass_g": 27.0},
    "Carolina Wren": {"primary": "#8f4f2c", "accent": "#eee5cb", "call_hz": 3000, "mass_g": 21.0},
    "Eastern Towhee": {"primary": "#2f2b2c", "accent": "#a5462c", "call_hz": 3500, "mass_g": 42.0},
    "Ovenbird": {"primary": "#7a7f5c", "accent": "#c2802a", "call_hz": 4300, "mass_g": 19.0},
    "Mallard": {"primary": "#3d6b4f", "accent": "#8a6a3d", "call_hz": 700, "mass_g": 1100.0},
    "Wood Duck": {"primary": "#2f6a6e", "accent": "#a8452f", "call_hz": 900, "mass_g": 650.0},
    "Cooper's Hawk": {"primary": "#5c6675", "accent": "#a35847", "call_hz": 1300, "mass_g": 450.0},
    "Herring Gull": {"primary": "#8fa0ad", "accent": "#d8a12a", "call_hz": 1100, "mass_g": 1100.0},

    # ── Heard far more often than seen ──────────────────────────────────
    #
    # The Haikubox logs these in the hundreds while the feeder camera
    # almost never catches them: they pass overhead, work the hedge, or
    # sing from a tree well outside the frame. The Biome page is built on
    # the audio cache, so without real entries here its second-biggest
    # plant would be drawing a hash colour at a hash pitch.
    "Chimney Swift": {"primary": "#6e6a66", "accent": "#3a3835", "call_hz": 6000, "mass_g": 23.0},
    "Northern Parula": {"primary": "#4a7fb5", "accent": "#d9c43f", "call_hz": 7000, "mass_g": 8.6},
    "Magnolia Warbler": {"primary": "#d6bd3a", "accent": "#2b2b2b", "call_hz": 5000, "mass_g": 8.7},
    "Blackpoll Warbler": {"primary": "#d8d4cc", "accent": "#2a2a2a", "call_hz": 8500, "mass_g": 12.0},
    "Wilson's Warbler": {"primary": "#e0c23a", "accent": "#1c1c1c", "call_hz": 4500, "mass_g": 7.5},
    "Common Yellowthroat": {"primary": "#d4b52f", "accent": "#2c2c2c", "call_hz": 3900, "mass_g": 10.0},
    "Cedar Waxwing": {"primary": "#b99a6b", "accent": "#b8452f", "call_hz": 7000, "mass_g": 32.0},
    "House Wren": {"primary": "#7a5f46", "accent": "#cbb89a", "call_hz": 4000, "mass_g": 11.0},
    "Winter Wren": {"primary": "#5e4433", "accent": "#c2ad92", "call_hz": 5500, "mass_g": 9.0},
    "Blue-gray Gnatcatcher": {"primary": "#8794a8", "accent": "#f0eee8", "call_hz": 6500, "mass_g": 6.0},
    "Fish Crow": {"primary": "#2f3338", "accent": "#585f66", "call_hz": 800, "mass_g": 290.0},

    # ── Plausible visitors not yet logged here ──────────────────────────
    "American Goldfinch": {"primary": "#d9aa1e", "accent": "#222222", "call_hz": 4200, "mass_g": 13.0},
    "Tufted Titmouse": {"primary": "#7d8d99", "accent": "#d6945a", "call_hz": 3200, "mass_g": 21.0},
    "Carolina Chickadee": {"primary": "#626b71", "accent": "#edeae1", "call_hz": 3900, "mass_g": 10.5},
    "White-breasted Nuthatch": {"primary": "#4e6a82", "accent": "#f2ede4", "call_hz": 2000, "mass_g": 21.0},
    "Red-breasted Nuthatch": {"primary": "#945c43", "accent": "#3f5466", "call_hz": 2600, "mass_g": 10.0},
    "Hairy Woodpecker": {"primary": "#a81f24", "accent": "#1c1c1c", "call_hz": 2900, "mass_g": 65.0},
    "Red-bellied Woodpecker": {"primary": "#cf332b", "accent": "#bfb8aa", "call_hz": 1900, "mass_g": 68.0},
    "Northern Flicker": {"primary": "#a8763f", "accent": "#c0472f", "call_hz": 1800, "mass_g": 130.0},
    "Pileated Woodpecker": {"primary": "#bd1519", "accent": "#111111", "call_hz": 1200, "mass_g": 300.0},
    "Baltimore Oriole": {"primary": "#e05c07", "accent": "#1a1a1a", "call_hz": 2400, "mass_g": 34.0},
    "Scarlet Tanager": {"primary": "#d1232a", "accent": "#111111", "call_hz": 2800, "mass_g": 28.0},
    "Indigo Bunting": {"primary": "#2b56a8", "accent": "#0d1b3a", "call_hz": 4500, "mass_g": 14.5},
    "Rose-breasted Grosbeak": {"primary": "#d91e36", "accent": "#1c1c1c", "call_hz": 2100, "mass_g": 45.0},
    "Purple Finch": {"primary": "#9c2a4c", "accent": "#6b545b", "call_hz": 3500, "mass_g": 25.0},
    "Song Sparrow": {"primary": "#845436", "accent": "#e8dec8", "call_hz": 3700, "mass_g": 20.0},
    "White-throated Sparrow": {"primary": "#7a5c4a", "accent": "#dfd772", "call_hz": 4100, "mass_g": 26.0},
    "Eastern Bluebird": {"primary": "#2f74ad", "accent": "#a34828", "call_hz": 2700, "mass_g": 31.0},
    "Red-winged Blackbird": {"primary": "#33322f", "accent": "#d62828", "call_hz": 2500, "mass_g": 64.0},
    "Osprey": {"primary": "#6a7482", "accent": "#e6e2d8", "call_hz": 2400, "mass_g": 1600.0},
}

# Family-level labels ("Sparrow", "Warbler") are real rows in the species
# table, not species. Matched as a whole word against the common name so
# "Sparrow" catches the family label and "Field Sparrow" alike, without
# "Woodpecker" swallowing an unrelated name that merely contains it.
FAMILY_STYLES: dict[str, dict[str, Any]] = {
    "Sparrow": {"primary": "#8d7355", "accent": "#d9cfbf", "call_hz": 3700, "mass_g": 22.0},
    "Finch": {"primary": "#b3423f", "accent": "#807062", "call_hz": 3800, "mass_g": 18.0},
    "Warbler": {"primary": "#b59a2c", "accent": "#576b50", "call_hz": 5000, "mass_g": 11.0},
    "Woodpecker": {"primary": "#ad252b", "accent": "#242424", "call_hz": 2600, "mass_g": 55.0},
    "Dove": {"primary": "#a08b76", "accent": "#63584e", "call_hz": 500, "mass_g": 110.0},
    "Hawk": {"primary": "#5c6675", "accent": "#a35847", "call_hz": 1400, "mass_g": 500.0},
    "Gull": {"primary": "#8fa0ad", "accent": "#d8a12a", "call_hz": 1100, "mass_g": 900.0},
}

# "Unknown bird" is a positive human confirmation that something was a bird
# without a species commitment, so it earns a deliberate neutral rather than
# the anonymous fallback — on a heavily-unreviewed day these dominate the
# canvas and want to read as "unlabelled", not as a species colour.
UNKNOWN_STYLE: dict[str, Any] = {
    "primary": "#8a9285", "accent": "#5e6a55", "call_hz": 2800, "mass_g": 30.0,
}

# Detections the classifier rejected outright carry no species at all.
DEFAULT_STYLE: dict[str, Any] = {
    "primary": "#6f7d63", "accent": "#384720", "call_hz": 2800, "mass_g": 30.0,
}


def species_style(common_name: str | None) -> dict[str, Any]:
    """Plumage colour, body mass and call frequency for one species name.

    Falls through exact match, family-label match, then a deterministic
    hash colour so a species we've never catalogued still gets a stable,
    distinguishable hue instead of sharing the generic olive with every
    other stranger. The hash branch returns hex like every other branch —
    the canvas parses hex only, and an `hsl(...)` string here silently
    collapsed every uncatalogued species to one fallback colour.
    """
    if not common_name:
        return dict(DEFAULT_STYLE)

    if common_name in SPECIES_STYLES:
        return dict(SPECIES_STYLES[common_name])

    if common_name == "Unknown bird":
        return dict(UNKNOWN_STYLE)

    words = {w.strip(",.") for w in common_name.split()}
    for family, style in FAMILY_STYLES.items():
        if family in words:
            return dict(style)

    return _hash_style(common_name)


def _hash_style(common_name: str) -> dict[str, Any]:
    """A stable pseudo-random plumage for an uncatalogued species.

    Saturation and lightness are pinned to a mid band so the colour stays
    legible on both the Sage paper and the Twilight ground; only hue varies.
    """
    digest = hashlib.md5(common_name.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    return {
        "primary": _hsl_hex(hue, 0.46, 0.44),
        "accent": _hsl_hex((hue + 0.08) % 1.0, 0.32, 0.30),
        "call_hz": 1500 + digest[1] * 12,
        "mass_g": 15.0 + digest[2] / 4.0,
    }


def _hsl_hex(h: float, s: float, ell: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, ell, s)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


# ── Flight paths ────────────────────────────────────────────────────────


def flight_points(
    bbox: list | None,
    track_bboxes: list | None = None,
    seed: int = 0,
) -> tuple[list[dict[str, float]], str]:
    """Build one flightline for a detection, normalised to [0, 1].

    Returns `(points, kind)` where kind is either:

      "tracked"     — the path came from `track_bboxes`, so it is where the
                      bird actually went, frame by frame.
      "synthesized" — the detection has only a single best-frame bbox, so
                      the arrival and departure are invented around a real
                      perch position.

    Callers MUST pass `kind` through to the client. Most rows in the
    database predate `track_bboxes` and get the synthesised branch, and a
    visualisation that presents an invented arc as observed data is lying
    about what the camera saw. The perch point is real in both cases; only
    the approach and departure are made up.

    Each point is `{x, y, w, h, t}`: centre position and box size as
    fractions of the 4K frame, and `t` as position along the path in
    [0, 1]. Synthesis is seeded, so a given detection always draws the
    same arc.
    """
    usable = _usable_boxes(track_bboxes)
    if len(usable) >= 3:
        return _resample(_normalise(usable), TRACKED_SAMPLES), "tracked"

    anchor = _usable_boxes([bbox] if bbox else [])
    if anchor:
        px, py, pw, ph = _normalise(anchor)[0]
    else:
        # No bbox at all shouldn't happen (the column is non-nullable), but
        # a malformed JSON value would land here. Put the bird on the
        # feeder tray rather than dropping the flight.
        px, py, pw, ph = 0.62, 0.55, 0.03, 0.03

    return _synthesise(px, py, pw, ph, seed), "synthesized"


def _usable_boxes(boxes: list | None) -> list[list[float]]:
    """Keep only well-formed `[x, y, w, h]` entries with positive extent."""
    if not isinstance(boxes, list):
        return []
    out: list[list[float]] = []
    for b in boxes:
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            continue
        try:
            x, y, w, h = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        out.append([x, y, w, h])
    return out


def _normalise(boxes: list[list[float]]) -> list[tuple[float, float, float, float]]:
    """Pixel `[x, y, w, h]` corners to normalised `(cx, cy, w, h)` centres."""
    return [
        (
            _clamp((x + w / 2) / FRAME_W),
            _clamp((y + h / 2) / FRAME_H),
            w / FRAME_W,
            h / FRAME_H,
        )
        for x, y, w, h in boxes
    ]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def _pt(x: float, y: float, w: float, h: float, t: float) -> dict[str, float]:
    """One path point, rounded to the precision the canvas can actually use.

    Four decimals is sub-pixel at 4K and keeps a 200-flight response an
    order of magnitude smaller than full float repr would.
    """
    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "w": round(w, 5),
        "h": round(h, 5),
        "t": round(t, 3),
    }


def _resample(
    centres: list[tuple[float, float, float, float]], samples: int
) -> list[dict[str, float]]:
    """Catmull-Rom resample of a real track down to a fixed point count.

    A track can run to hundreds of frames; the canvas only needs a smooth
    curve. Catmull-Rom passes through every control point, so the resampled
    path still visits the positions the bird was actually detected at.
    """
    n = len(centres)
    if n < 2:
        return [_pt(*centres[0], 0.0)] if centres else []

    out: list[dict[str, float]] = []
    for s in range(samples):
        u = s / (samples - 1)
        scaled = u * (n - 1)
        i = min(n - 2, int(scaled))
        local = scaled - i
        p0 = centres[max(0, i - 1)]
        p1 = centres[i]
        p2 = centres[i + 1]
        p3 = centres[min(n - 1, i + 2)]
        out.append(
            _pt(
                _catmull(p0[0], p1[0], p2[0], p3[0], local),
                _catmull(p0[1], p1[1], p2[1], p3[1], local),
                p1[2] + (p2[2] - p1[2]) * local,
                p1[3] + (p2[3] - p1[3]) * local,
                u,
            )
        )
    return out


def _catmull(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


def _synthesise(
    px: float, py: float, pw: float, ph: float, seed: int
) -> list[dict[str, float]]:
    """Invent an arrival-perch-departure arc around a real perch position.

    Shaped after how a small bird reaches a feeder: a curving approach that
    drops below the perch and flares up onto it, a dwell, then a drop-off
    and a climb away. Both legs are cubic Beziers whose control points are
    offset PERPENDICULAR to the chord, so the curvature scales with the
    distance travelled — control points placed along the chord give an arc
    that flattens into a straight line as the leg gets longer, which is how
    a canvas full of these ends up looking like a spirograph.

    The legs are deliberately short, a fifth to two fifths of the frame.
    Where the bird came from is not something the camera recorded, and a
    stroke that crosses the whole frame asserts far more than a single
    bounding box knows. Short arcs also keep the synthesised paths at
    roughly the scale of the real tracked ones, so the two read as one
    picture rather than two.
    """
    rng = random.Random(seed)
    px = _clamp(px, 0.05, 0.95)
    py = _clamp(py, 0.08, 0.92)

    # Approach: in from the nearer side of frame, angled slightly downward
    # more often than not, since birds tend to drop toward a feeder.
    toward_left = px < 0.5
    span_in = rng.uniform(0.17, 0.38)
    angle_in = math.radians(rng.uniform(155, 205) if toward_left else rng.uniform(-25, 25))
    angle_in -= math.radians(rng.uniform(-8, 34))  # bias: coming from above
    entry = (px + math.cos(angle_in) * span_in, py + math.sin(angle_in) * span_in)

    # Departure: broadly CONTINUING the approach heading rather than
    # striking off somewhere new, so the whole flight reads as one sweep
    # through the perch. Letting the two legs point independently turns a
    # shared perch into a starburst of legs — dozens of birds using the
    # same feeder then look like an insect, not like traffic.
    span_out = rng.uniform(0.15, 0.34)
    angle_out = angle_in + math.pi + math.radians(rng.uniform(-48, 48))
    angle_out -= math.radians(rng.uniform(10, 55))  # bias: climbing away
    exit_pt = (px + math.cos(angle_out) * span_out, py + math.sin(angle_out) * span_out)

    points: list[dict[str, float]] = []

    # Approach. The bow puts real curvature in the middle of the leg; the
    # flare pulls the last control point below the perch so the bird rises
    # onto it instead of arriving flat.
    bow_in = rng.choice((-1, 1)) * rng.uniform(0.20, 0.34)
    flare = ph * 2.5 + rng.uniform(0.015, 0.045)
    c1, c2 = _arc_controls(entry, (px, py), bow_in)
    c2 = (c2[0], c2[1] + flare)
    for s in range(APPROACH_SAMPLES):
        u = s / (APPROACH_SAMPLES - 1)
        x, y = _bezier(entry, c1, c2, (px, py), u)
        scale = 0.62 + 0.38 * u  # nearer the camera as it arrives
        points.append(_pt(x, y, pw * scale, ph * scale, u * 0.34))

    # Dwell: a tight shuffle around the perch, scaled to the bird's own box
    # so a pigeon covers more ground than a chickadee.
    #
    # Deliberately a smooth micro-orbit rather than random hops. A renderer
    # that strokes this path as a ribbon has to offset each point along the
    # path normal, and a sequence of random jitter reverses that normal
    # from one point to the next — the ribbon then folds through itself and
    # fills as a hard chevron instead of a stroke. A curve that never
    # doubles back renders cleanly at any width.
    # Kept small and capped: the shuffle is a knot at the perch, not a
    # feature of the composition. A loop scaled straight off the bounding
    # box makes a pigeon orbit wider than its own body, and a dozen of
    # those over one feeder swallow the flight paths entirely. Under one
    # full turn, so the fourteen samples above stay smooth rather than
    # closing into a visible polygon.
    orbit_x = min(pw * 0.5, 0.018)
    orbit_y = min(ph * 0.4, 0.018)
    phase = rng.uniform(0, math.tau)
    turns = rng.uniform(0.6, 1.0) * rng.choice((-1, 1))
    for s in range(DWELL_SAMPLES):
        u = s / max(1, DWELL_SAMPLES - 1)
        a = phase + turns * math.tau * u
        wobble = 1 + 0.25 * math.sin(a * 3 + phase)
        points.append(
            _pt(
                px + math.cos(a) * orbit_x * wobble,
                py + math.sin(a) * orbit_y * wobble,
                pw,
                ph,
                0.34 + u * 0.30,
            )
        )

    # Departure: drop off the perch, then climb out.
    bow_out = rng.choice((-1, 1)) * rng.uniform(0.18, 0.32)
    d1, d2 = _arc_controls((px, py), exit_pt, bow_out)
    d1 = (d1[0], d1[1] + ph * 2.0 + rng.uniform(0.01, 0.03))
    for s in range(1, DEPART_SAMPLES + 1):
        u = s / DEPART_SAMPLES
        x, y = _bezier((px, py), d1, d2, exit_pt, u)
        scale = 1.0 - 0.42 * u  # receding
        points.append(_pt(x, y, pw * scale, ph * scale, 0.64 + u * 0.36))

    return points


def _arc_controls(
    a: tuple[float, float], b: tuple[float, float], bow: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two Bezier controls that bow the chord a->b sideways by `bow` * |a-b|.

    Offsetting perpendicular to the chord is what keeps the curve curved at
    any length; `bow`'s sign picks which way the arc swings.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy) or 1e-6
    nx = -dy / length
    ny = dx / length
    off = bow * length
    return (
        (a[0] + dx * 0.30 + nx * off, a[1] + dy * 0.30 + ny * off),
        (a[0] + dx * 0.70 + nx * off, a[1] + dy * 0.70 + ny * off),
    )


def _bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    """Standard cubic Bézier evaluation."""
    mt = 1 - t
    a = mt * mt * mt
    b = 3 * mt * mt * t
    c = 3 * mt * t * t
    d = t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def chime_note(call_hz: float, index: int) -> float:
    """Snap a species' call frequency onto a pentatonic scale.

    The arrival chimes want to be pleasant when several fire close together,
    which raw call frequencies are not. Quantising to a C-major pentatonic
    keeps each species recognisably high or low against the others while
    guaranteeing any two notes are consonant. `index` nudges repeat
    sightings of the same species between octaves so a busy morning doesn't
    hammer one pitch.
    """
    base = 261.63  # middle C
    steps = [0, 2, 4, 7, 9]  # major pentatonic, in semitones
    semitones_from_base = 12 * math.log2(max(80.0, call_hz) / base)
    octave = int(semitones_from_base // 12)
    within = semitones_from_base - octave * 12
    nearest = min(steps, key=lambda s: abs(s - within))
    octave = max(-1, min(3, octave + (index % 2)))
    return base * (2 ** ((octave * 12 + nearest) / 12))
