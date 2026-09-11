"""Read-side API for Territory: Feeder Wars, the zone-control page.

Scores one camera-local day of feeder footage as a turf war: which faction
held which perch, for how long, and who pushed whom off it.

Four things about this endpoint are easy to get wrong.

  1. **A contender is a DETECTION, not a visit.** Same rule the Art and
     Tavern pages follow, and here it is load-bearing: the whole question
     is which of the four tracks in a clip was on the dish.

  2. **Days are camera-local.** `Visit.started_at` is naive UTC. Same
     treatment as every other surface in this app.

  3. **Occupancy and displacement are not the same claim, and the second
     one mostly cannot be made.** A bird's zone comes from its best-frame
     bbox, which every row has. A displacement needs two tracks on a
     shared clock, which needs `Detection.track_frames`, added when this
     page was built and therefore NULL on everything captured before.
     `summary.timed` and `summary.eligible_visits` say how much of the day
     could even be judged, and the page prints it.

  4. **Unidentified birds hold ground for nobody.** They are counted into
     a zone's traffic and into `unclaimed`, never into a faction's
     seconds. Most rows in this database have no species, so folding them
     into a faction would decide the war by classifier noise.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date as _date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session, joinedload

from db.models import (
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    Detection,
    Species,
    Visit,
)
from db.session import get_db
from pipeline.daylight import _sun_times
from pipeline.palette import species_style
from pipeline.territory import (
    FACTIONS,
    displacements as find_displacements,
    dwell,
    faction_for,
    load_zones,
    zone_for,
    zone_spans,
)
from settings import settings

router = APIRouter()

# Rows that cannot hold ground. "Unknown bird" is NOT here: the user
# confirmed a real bird sat there, which is a genuine occupancy even
# though it joins no faction.
NOT_A_CONTENDER = frozenset({NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL})

# Share of a day's birds that must carry a per-frame clock before control
# is scored in seconds held rather than in landings. Half is a judgement
# call: high enough that a stray timed track cannot decide a zone, low
# enough that the page starts using real dwell as soon as the pipeline has
# been running long enough to matter.
SECONDS_BASIS_COVERAGE = 0.5


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.camera_timezone)


def _to_local(naive_utc: datetime) -> datetime:
    return naive_utc.replace(tzinfo=timezone.utc).astimezone(_tz())


def _local_day_bounds(target: _date) -> tuple[datetime, datetime]:
    """Naive-UTC half-open bounds for one camera-local calendar day."""
    tz = _tz()
    start_local = datetime.combine(target, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(timezone.utc).replace(tzinfo=None),
        end_local.astimezone(timezone.utc).replace(tzinfo=None),
    )


def _day_fraction(local_dt: datetime) -> float:
    seconds = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
    return seconds / 86400.0


def _sun_fractions(target: _date) -> dict[str, float]:
    try:
        sunrise, sunset = _sun_times(target.isoformat())
        return {
            "sunrise": round(_day_fraction(sunrise), 4),
            "sunset": round(_day_fraction(sunset), 4),
        }
    except Exception:  # noqa: BLE001 - astral raises several unrelated types
        return {"sunrise": 0.25, "sunset": 0.79}


def _resolve_date(db: Session, raw: str | None) -> _date:
    """Parse the requested date, or find the most recent day with footage."""
    if raw:
        try:
            return _date.fromisoformat(raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid date, expected YYYY-MM-DD"
            ) from exc

    latest = (
        db.query(Visit.started_at)
        .join(Detection, Detection.visit_id == Visit.id)
        .order_by(desc(Visit.started_at))
        .first()
    )
    if latest is None:
        return datetime.now(_tz()).date()
    return _to_local(latest[0]).date()


@router.get("/day")
async def get_day(
    date: str | None = Query(
        None, description="Camera-local date, YYYY-MM-DD. Defaults to the most recent day with footage."
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """One camera-local day of feeder footage, scored as a turf war."""
    target = _resolve_date(db, date)
    start_utc, end_utc = _local_day_bounds(target)
    zones, zone_source = load_zones()
    by_id = {z["id"]: z for z in zones}

    rows = (
        db.query(Detection)
        .options(joinedload(Detection.species), joinedload(Detection.visit))
        .join(Visit, Detection.visit_id == Visit.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(Visit.started_at >= start_utc, Visit.started_at < end_utc)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(NOT_A_CONTENDER))
        )
        .order_by(Visit.started_at.asc(), Detection.id.asc())
        .all()
    )

    zone_seconds: dict[str, Counter] = defaultdict(Counter)
    zone_holds: dict[str, Counter] = defaultdict(Counter)
    zone_unclaimed: Counter = Counter()
    zone_traffic: Counter = Counter()
    faction_seconds: Counter = Counter()
    faction_holds: Counter = Counter()
    faction_species: dict[str, Counter] = defaultdict(Counter)
    dwell_quality: Counter = Counter()

    by_visit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    timed = 0
    tracked = 0

    for d in rows:
        name = d.species.common_name if d.species else None
        seconds, kind = dwell(d.track_bboxes, d.track_frames)
        dwell_quality[kind] += 1
        if isinstance(d.track_bboxes, list) and len(d.track_bboxes) >= 2:
            tracked += 1
        has_clock = isinstance(d.track_frames, list) and len(d.track_frames) >= 2
        if has_clock:
            timed += 1

        # A timed track can be split across zones it actually moved
        # between; an untimed one can only be placed where its best frame
        # was, so all of its seconds go there.
        spans = zone_spans(d.track_bboxes, d.track_frames, zones) if has_clock else {}
        if spans:
            total_frames = sum(n for _, _, n in spans.values())
            placement = {
                zid: seconds * (n / total_frames) for zid, (_, _, n) in spans.items()
            }
        else:
            zid = zone_for(d.bbox, zones)
            placement = {zid: seconds} if zid else {}

        faction = faction_for(name)
        for zid, secs in placement.items():
            zone_traffic[zid] += 1
            if faction:
                zone_seconds[zid][faction] += secs
                zone_holds[zid][faction] += 1
                faction_seconds[faction] += secs
                faction_holds[faction] += 1
                if name:
                    faction_species[faction][name] += 1
            else:
                zone_unclaimed[zid] += 1

        by_visit[d.visit_id].append(
            {
                "detection_id": d.id,
                "species": name,
                "track_bboxes": d.track_bboxes,
                "track_frames": d.track_frames,
            }
        )

    # Displacements are per visit, because two birds have to be in the
    # same clip to contest anything. A visit is eligible only when at
    # least two of its contenders carry a clock.
    events: list[dict[str, Any]] = []
    eligible_visits = 0
    contested_visits = 0
    for visit_id, contenders in by_visit.items():
        if len(contenders) < 2:
            continue
        contested_visits += 1
        with_clock = [
            c for c in contenders
            if isinstance(c.get("track_frames"), list) and len(c["track_frames"]) >= 2
        ]
        if len(with_clock) < 2:
            continue
        eligible_visits += 1
        for e in find_displacements(contenders, zones):
            e["visit_id"] = visit_id
            events.append(e)

    # Which measure decides who held a perch.
    #
    # Seconds are the real answer, but only tracks carrying `track_frames`
    # have any, and a day where a handful do is worse than a day where
    # none do: every untimed bird scores zero, so a perch with two real
    # landings on it reads as "nobody held this" while a perch with one
    # timed visitor takes the whole day. Appearances are the honest
    # measure until timings cover most of the day, so the switch happens
    # on coverage rather than on the first timed row. `control_basis`
    # goes to the page, which says which one it means.
    basis = (
        "seconds"
        if rows and timed >= SECONDS_BASIS_COVERAGE * len(rows)
        else "appearances"
    )

    def _score(fid: str, zid: str) -> float:
        return zone_seconds[zid][fid] if basis == "seconds" else zone_holds[zid][fid]

    zone_payload = []
    for z in zones:
        zid = z["id"]
        contenders_here = set(zone_seconds.get(zid, Counter())) | set(
            zone_holds.get(zid, Counter())
        )
        # Ties on the basis break toward whoever has the measured dwell:
        # two factions with one landing each are level on appearances, and
        # the one the camera actually timed is the better answer. Name
        # last, so the order is stable when nothing separates them.
        scored = sorted(
            ((fid, _score(fid, zid)) for fid in contenders_here if fid in FACTIONS),
            key=lambda kv: (-kv[1], -zone_seconds[zid][kv[0]], kv[0]),
        )
        total = sum(v for _, v in scored)
        control = [
            {
                "faction": fid,
                "name": FACTIONS[fid]["name"],
                "color": FACTIONS[fid]["color"],
                "seconds": round(zone_seconds[zid][fid], 1),
                "holds": zone_holds[zid][fid],
                "share": round(v / total, 4) if total else 0.0,
            }
            for fid, v in scored
            if v > 0
        ]
        zone_payload.append(
            {
                **{k: z[k] for k in ("id", "x", "y", "radius")},
                "name": z.get("name", zid),
                "blurb": z.get("blurb", ""),
                "seconds": round(sum(zone_seconds[zid].values()), 1),
                "visits": zone_traffic[zid],
                "unclaimed": zone_unclaimed[zid],
                "control": control,
                "holder": control[0]["faction"] if control else None,
                # Two or more factions with a real share means the perch
                # changed hands or was shared during the day.
                "contested": len([c for c in control if c["share"] >= 0.15]) > 1,
            }
        )

    standings = [
        {
            "faction": fid,
            "name": FACTIONS[fid]["name"],
            "style": FACTIONS[fid]["style"],
            "blurb": FACTIONS[fid]["blurb"],
            "color": FACTIONS[fid]["color"],
            "weight": FACTIONS[fid]["weight"],
            "seconds": round(faction_seconds[fid], 1),
            "holds": faction_holds[fid],
            "zones_held": sum(1 for z in zone_payload if z["holder"] == fid),
            "species": [
                {"species": s, "count": n, "color": species_style(s)["primary"]}
                for s, n in faction_species[fid].most_common(6)
            ],
            "wins": sum(1 for e in events if e["winner_faction"] == fid),
            "losses": sum(1 for e in events if e["loser_faction"] == fid),
        }
        for fid, v in sorted(
            (
                (f, faction_seconds[f] if basis == "seconds" else faction_holds[f])
                for f in set(faction_seconds) | set(faction_holds)
                if f in FACTIONS
            ),
            key=lambda kv: (-kv[1], -faction_seconds[kv[0]], kv[0]),
        )
        if v > 0
    ]

    return {
        "date": target.isoformat(),
        "tz": settings.camera_timezone,
        "sun": _sun_fractions(target),
        "zone_source": zone_source,
        # "seconds" when the day's tracks carry a dwell, "appearances"
        # when none do and control is counted in landings instead.
        "control_basis": basis,
        "zones": zone_payload,
        "standings": standings,
        "displacements": [_render_event(e, by_id) for e in events],
        "dispatch": _dispatch(target, zone_payload, standings, events),
        "summary": {
            "detections": len(rows),
            "visits": len(by_visit),
            "in_a_zone": sum(zone_traffic.values()),
            "unclaimed": sum(zone_unclaimed.values()),
            # How much of the day could be judged at all, which is the
            # number that decides whether the pecking order means anything.
            "tracked": tracked,
            "timed": timed,
            "contested_visits": contested_visits,
            "eligible_visits": eligible_visits,
            "dwell_quality": dict(dwell_quality),
            "quiet": not rows,
        },
    }


def _render_event(event: dict[str, Any], zones: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attach the display names a displacement needs to read as a sentence."""
    zone = zones.get(event["zone"], {})
    winner_faction = event.get("winner_faction")
    loser_faction = event.get("loser_faction")
    return {
        **event,
        "zone_name": zone.get("name", event["zone"]),
        "winner_faction_name": FACTIONS[winner_faction]["name"] if winner_faction else None,
        "winner_color": FACTIONS[winner_faction]["color"] if winner_faction else "#8a9285",
        "loser_faction_name": FACTIONS[loser_faction]["name"] if loser_faction else None,
        "loser_color": FACTIONS[loser_faction]["color"] if loser_faction else "#8a9285",
    }


def _listed(items: list[str]) -> str:
    """Join names the way a sentence does, with an "and" before the last."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _dispatch(
    target: _date,
    zones: list[dict[str, Any]],
    standings: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    """The day's scoreboard as one or two sentences.

    Every clause is a fact from the tallies above. When nothing held long
    enough to score, it says so rather than inventing a battle: a quiet
    yard reading as a dramatic stalemate would be the easiest possible way
    for this page to start lying.
    """
    held = [z for z in zones if z["holder"]]
    if not held:
        return "Nothing held the yard today. Every bird the camera caught was passing through."

    # Grouped by holder, because one faction taking three perches should
    # read as one clause and not as the same sentence three times.
    by_faction: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for z in held:
        top = z["control"][0]
        by_faction.setdefault(top["faction"], []).append(z["name"])
        names[top["faction"]] = top["name"]

    ranked = sorted(by_faction.items(), key=lambda kv: (-len(kv[1]), names[kv[0]]))
    line = "; ".join(f"{names[fid]} held {_listed(zs)}" for fid, zs in ranked) + "."

    disputed = [z["name"] for z in zones if z["contested"]]
    if disputed:
        was = "was" if len(disputed) == 1 else "were"
        line += f" {_listed(disputed)} {was} contested."

    if events:
        first = events[0]
        winner = first.get("winner") or "an unnamed bird"
        loser = first.get("loser") or "an unnamed bird"
        zone_name = next(
            (z["name"] for z in zones if z["id"] == first["zone"]), first["zone"]
        )
        n = len(events)
        line += (
            f" {n} perch{'es' if n != 1 else ''} changed hands, the first when "
            f"a {winner} took {zone_name} off a {loser}."
        )
    elif standings:
        line += " No perch changed hands on camera."
    return line


@router.get("/dates")
async def get_dates(
    limit: int = Query(60, ge=1, le=365, description="How many recent days with footage to return"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Camera-local days with footage, newest first.

    Grouped in Python for the same reason the Art page's equivalent is:
    SQLite's `date()` bins by UTC and the local boundary falls inside the
    feeder's active hours for part of the year.
    """
    rows = (
        db.query(Visit.started_at, Detection.track_frames)
        .join(Detection, Detection.visit_id == Visit.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(NOT_A_CONTENDER))
        )
        .all()
    )

    totals: Counter[_date] = Counter()
    timed: Counter[_date] = Counter()
    for started_at, track_frames in rows:
        if started_at is None:
            continue
        day = _to_local(started_at).date()
        totals[day] += 1
        if isinstance(track_frames, list) and len(track_frames) >= 2:
            timed[day] += 1

    days = sorted(totals, reverse=True)[:limit]
    return {
        "tz": settings.camera_timezone,
        "dates": [
            {
                "date": day.isoformat(),
                "detection_count": totals[day],
                # How many carry a clock, so the picker can show which days
                # a pecking order could be read from at all.
                "timed_count": timed[day],
            }
            for day in days
        ],
    }
