"""Read-side API for Chrono-Chirps, the Biome page's living audio garden.

Serves one camera-local day of Haikubox audio as a garden: one plant per
species heard, grown from how often and how widely across the day it
called. Four things are easy to get wrong here and worth stating up front:

  1. **A plant is a SPECIES, not a detection.** The Art and Tavern pages
     both treat a detection as the unit, because the camera tracks
     individual birds. The Haikubox does not: it reports that a House
     Sparrow was audible in a three-second window, seven hundred times a
     day, which is one hedge full of sparrows rather than seven hundred
     arrivals. So the species is the unit, and the call count is what the
     plant is grown from. See pipeline.biome.

  2. **This page reads the audio table, not the detections table.** Every
     other surface in the app is built on `detections`. The garden is
     built on `haikubox_detections`, which has its own species vocabulary
     (Chimney Swift, Northern Parula and Blackpoll Warbler are heard
     hundreds of times and seen essentially never) and its own gaps.

  3. **Days are camera-local.** `HaikuboxDetection.detected_at` is naive
     UTC. Binning by UTC would cut the dawn chorus off the front of the
     day for half the year. Same rule the Art and Tavern pages follow.

  4. **Cross-pollination is reported at two strengths.** The design
     document wants a golden pollinator when a bird is confirmed on
     camera and microphone at once, which is `Detection.audio_confirmed`.
     That flag is strict, a ninety-second correlation window, and it
     fires rarely: on this database it is true for a handful of rows in
     total. So the endpoint also returns the looser case, the same
     species logged by the camera somewhere in the same local day, and
     marks which is which with `confirmed`. The page draws the strict
     ones bright and the loose ones pale, and says so.
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
    UNKNOWN_BIRD_LABEL,
    Detection,
    HaikuboxDetection,
    Species,
    Visit,
)
from db.session import get_db
from pipeline.biome import plant
from pipeline.daylight import _sun_times
from pipeline.palette import chime_note, species_style
from settings import settings

router = APIRouter()

# Labels that cannot pollinate anything. "Unknown bird" joins the two feed
# sentinels here, unlike on the Art page: a pollinator has to carry a
# species name to reach the right plant, and an unnamed bird has none.
NOT_A_POLLINATOR = frozenset({NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL, UNKNOWN_BIRD_LABEL})

# Most pollinators the canvas is asked to fly at once. Past a dozen they
# stop reading as an event and start reading as weather.
MAX_POLLINATORS = 14


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
    """Position within the local calendar day, in [0, 1)."""
    seconds = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
    return seconds / 86400.0


def _sun_fractions(target: _date) -> dict[str, float]:
    """Sunrise and sunset as day fractions, for the terrarium's sky.

    The garden darkens outside these, which is also when the dawn chorus
    reads as early rather than as ordinary. Falls back to rough civil
    hours if astral cannot resolve the date.
    """
    try:
        sunrise, sunset = _sun_times(target.isoformat())
        return {
            "sunrise": round(_day_fraction(sunrise), 4),
            "sunset": round(_day_fraction(sunset), 4),
        }
    except Exception:  # noqa: BLE001 - astral raises several unrelated types
        return {"sunrise": 0.25, "sunset": 0.79}


def _resolve_date(db: Session, raw: str | None) -> _date:
    """Parse the requested date, or find the most recent day with audio.

    Deliberately the most recent day the *microphone* heard something,
    which is not always the most recent day the camera saw something. The
    Haikubox can be offline for a week while the feeder stays busy, and
    opening this page on an empty garden when there is a full one two days
    back is the wrong default.
    """
    if raw:
        try:
            return _date.fromisoformat(raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid date, expected YYYY-MM-DD"
            ) from exc

    latest = (
        db.query(HaikuboxDetection.detected_at)
        .order_by(desc(HaikuboxDetection.detected_at))
        .first()
    )
    if latest is None:
        return datetime.now(_tz()).date()
    return _to_local(latest[0]).date()


@router.get("/garden")
async def get_garden(
    date: str | None = Query(
        None, description="Camera-local date, YYYY-MM-DD. Defaults to the most recent day with audio."
    ),
    limit: int = Query(40, ge=1, le=80, description="Max plants returned, the most-heard first"),
    min_calls: int = Query(
        1, ge=1, le=500, description="Drop species heard fewer times than this"
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """One camera-local day of yard audio, as a garden of plants."""
    target = _resolve_date(db, date)
    start_utc, end_utc = _local_day_bounds(target)

    rows = (
        db.query(
            HaikuboxDetection.species_common_name,
            HaikuboxDetection.detected_at,
            HaikuboxDetection.confidence,
        )
        .filter(HaikuboxDetection.detected_at >= start_utc)
        .filter(HaikuboxDetection.detected_at < end_utc)
        .all()
    )

    heard = _gather(rows)
    busiest = max((h["calls"] for h in heard.values()), default=0)

    # Kept in call order for the cut, then handed back low pitch to high so
    # the canvas can plant the garden as a spectrogram runs, deep callers
    # along the ground and thin whistles above them, without sorting it
    # again on the client.
    keep = sorted(
        (h for h in heard.values() if h["calls"] >= min_calls),
        key=lambda h: -h["calls"],
    )[:limit]

    total_calls = sum(h["calls"] for h in heard.values())
    plants = [_plant(h, busiest=busiest, total_calls=total_calls) for h in keep]
    plants.sort(key=lambda p: p["pitch"])

    pollinators = _pollinators(db, start_utc, end_utc, {p["species"] for p in plants})
    for p in plants:
        p["flowering"] = any(v["species"] == p["species"] for v in pollinators)
        p["confirmed"] = any(
            v["species"] == p["species"] and v["confirmed"] for v in pollinators
        )

    chorus = [0] * 24
    for h in heard.values():
        for i, n in enumerate(h["hours"]):
            chorus[i] += n

    return {
        "date": target.isoformat(),
        "tz": settings.camera_timezone,
        "sun": _sun_fractions(target),
        "plants": plants,
        "pollinators": pollinators,
        "chorus": chorus,
        "summary": {
            "returned": len(plants),
            "species_heard": len(heard),
            "calls": total_calls,
            "busiest": busiest,
            "peak_hour": max(range(24), key=lambda i: chorus[i]) if total_calls else None,
            "truncated": len(heard) > len(plants),
            "quiet": total_calls == 0,
            # Every plant on a given day shares one source, so reporting it
            # per garden rather than per plant keeps the page from having
            # to reconcile two answers.
            "energy_source": plants[0]["energy_source"] if plants else None,
        },
    }


def _gather(rows: list[tuple[str, datetime, float | None]]) -> dict[str, dict[str, Any]]:
    """Fold one day of audio rows into per-species totals in local time."""
    out: dict[str, dict[str, Any]] = {}
    for name, when, confidence in rows:
        if not name or when is None:
            continue
        local = _to_local(when)
        entry = out.get(name)
        if entry is None:
            entry = out[name] = {
                "species": name,
                "calls": 0,
                "hours": [0] * 24,
                "first": local,
                "last": local,
                "scores": [],
            }
        entry["calls"] += 1
        entry["hours"][local.hour] += 1
        if local < entry["first"]:
            entry["first"] = local
        if local > entry["last"]:
            entry["last"] = local
        if confidence is not None:
            entry["scores"].append(float(confidence))
    return out


def _plant(heard: dict[str, Any], *, busiest: int, total_calls: int) -> dict[str, Any]:
    """Grow one species, then hang its timings on the result."""
    first = _day_fraction(heard["first"])
    last = _day_fraction(heard["last"])
    scores = heard["scores"]
    # None rather than zero when the API sent no scores. A species heard
    # all day with no confidence attached is not a species the box was
    # unsure about, and averaging an empty list into 0.0 would wither it.
    mean_score = sum(scores) / len(scores) if scores else None

    p = plant(
        heard["species"],
        calls=heard["calls"],
        busiest=busiest,
        spread=max(0.0, last - first),
        mean_confidence=mean_score,
    )
    p.update(
        {
            "calls": heard["calls"],
            "share": round(heard["calls"] / total_calls, 4) if total_calls else 0.0,
            "hours": heard["hours"],
            "first_heard": round(first, 5),
            "last_heard": round(last, 5),
            "first_heard_at": heard["first"].isoformat(),
            "last_heard_at": heard["last"].isoformat(),
            "mean_confidence": round(mean_score, 3) if mean_score is not None else None,
            # The ambient chorus plays this, not the species' real call
            # pitch: thirty raw call frequencies sounding at once is noise.
            # Same treatment the Art page's arrival chimes get.
            "chime_hz": round(chime_note(p["call_hz"], heard["calls"]), 2),
        }
    )
    return p


def _pollinators(
    db: Session,
    start_utc: datetime,
    end_utc: datetime,
    species_heard: set[str],
) -> list[dict[str, Any]]:
    """Camera sightings of species the microphone also heard that day.

    `confirmed` separates the two strengths. True means the pipeline
    itself matched this detection to an audio detection inside
    `settings.audio_correlation_window_seconds`, so the bird was seen and
    heard at the same moment. False means only that the camera logged the
    same species somewhere in the same local day, which is a weaker claim
    and drawn as a paler visitor.

    Audio-confirmed rows are taken first so a busy day cannot fill the cap
    with same-day coincidences and hide the real thing.
    """
    if not species_heard:
        return []

    rows = (
        db.query(Detection)
        .options(joinedload(Detection.species), joinedload(Detection.visit))
        .join(Visit, Detection.visit_id == Visit.id)
        .join(Species, Detection.species_id == Species.id)
        .filter(Visit.started_at >= start_utc, Visit.started_at < end_utc)
        .filter(Species.common_name.in_(species_heard))
        .filter(~Species.common_name.in_(NOT_A_POLLINATOR))
        .order_by(desc(Detection.audio_confirmed), Visit.started_at.asc())
        .limit(MAX_POLLINATORS * 4)
        .all()
    )

    # One visitor per species per strength, so a dove the camera caught
    # ninety times does not bury every other plant's pollinator.
    seen: set[tuple[str, bool]] = set()
    out: list[dict[str, Any]] = []
    for d in rows:
        name = d.species.common_name
        confirmed = bool(d.audio_confirmed)
        key = (name, confirmed)
        if key in seen:
            continue
        seen.add(key)
        when = d.visit.started_at if d.visit else d.created_at
        local = _to_local(when)
        out.append(
            {
                "detection_id": d.id,
                "species": name,
                "species_id": d.species_id,
                "at": local.isoformat(),
                "time_of_day": round(_day_fraction(local), 5),
                "confirmed": confirmed,
                "crop_url": f"/media/{d.crop_path}" if d.crop_path else None,
                "color": species_style(name)["primary"],
            }
        )
        if len(out) >= MAX_POLLINATORS:
            break
    return out


@router.get("/dates")
async def get_dates(
    limit: int = Query(60, ge=1, le=365, description="How many recent days with audio to return"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Camera-local days the microphone heard something, newest first.

    Grouped in Python rather than SQL for the same reason the Art page's
    equivalent is: SQLite's `date()` bins by UTC, and the local boundary
    falls inside the yard's active hours for part of the year. One indexed
    column over the audio cache, which holds a few thousand rows a week.
    """
    rows = db.query(
        HaikuboxDetection.detected_at, HaikuboxDetection.species_common_name
    ).all()

    per_day: dict[_date, Counter] = defaultdict(Counter)
    totals: Counter[_date] = Counter()
    for when, name in rows:
        if when is None:
            continue
        day = _to_local(when).date()
        totals[day] += 1
        if name:
            per_day[day][name] += 1

    days = sorted(totals, reverse=True)[:limit]
    return {
        "tz": settings.camera_timezone,
        "dates": [
            {
                "date": day.isoformat(),
                "call_count": totals[day],
                "species_count": len(per_day[day]),
                "top_species": [
                    {"species": name, "count": n}
                    for name, n in per_day[day].most_common(3)
                ],
            }
            for day in days
        ],
    }
