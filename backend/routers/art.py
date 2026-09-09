"""Read-side API for the Art page's generative flightline canvas.

Serves one day of feeder activity as normalised flight paths the PWA can
stroke onto a canvas. Two things about this endpoint are easy to get wrong
and worth stating up front:

  1. **A flight is a DETECTION, not a visit.** Detection is documented as
     "one identified bird (per-track aggregate) within a visit" — a visit
     with forty detections is forty separate tracks, usually forty separate
     birds. Treating a visit as one flight and stringing its detections
     into a path draws a line between two different birds perched in
     different places, which is not a thing that happened.

  2. **Days are camera-local.** Visit.started_at is naive UTC; the feeder's
     meaningful clock is settings.camera_timezone. Binning by UTC date
     would split a summer evening's activity across two "days" and put the
     mandala's noon in the wrong place. Same reasoning as
     pipeline.stats.compute_species_activity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date as _date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session, joinedload

from db.models import NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL, Detection, Species, Visit
from db.session import get_db
from pipeline.daylight import _sun_times
from pipeline.palette import chime_note, flight_points, species_style
from settings import settings

router = APIRouter()

# Same pair the default feed hides (see routers/detections.HIDDEN_FROM_FEED):
# false positives and crops the user retired as unidentifiable. "Unknown
# bird" is NOT hidden — it is a real bird sighting, just species-unspecified,
# and it gets its own neutral plumage in the palette.
HIDDEN_LABELS = frozenset({NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL})


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.camera_timezone)


def _local_day_bounds(target: _date) -> tuple[datetime, datetime]:
    """Naive-UTC half-open bounds for one camera-local calendar day.

    Naive because that is how Visit.started_at is stored, so the comparison
    happens in the same frame SQLite holds the column in.
    """
    tz = _tz()
    start_local = datetime.combine(target, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return (
        start_local.astimezone(timezone.utc).replace(tzinfo=None),
        end_local.astimezone(timezone.utc).replace(tzinfo=None),
    )


def _to_local(naive_utc: datetime) -> datetime:
    return naive_utc.replace(tzinfo=timezone.utc).astimezone(_tz())


def _day_fraction(local_dt: datetime) -> float:
    """Position within the local calendar day, in [0, 1)."""
    seconds = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
    return seconds / 86400.0


def _sun_fractions(target: _date) -> dict[str, float]:
    """Sunrise and sunset as day fractions, for shading the night arc.

    The scrubber opens at dawn rather than at midnight, and the mandala
    dims the hours the camera was asleep, so the canvas needs to know where
    the light was. Falls back to rough civil hours if astral can't resolve
    the date (polar latitudes, bad config).
    """
    try:
        sunrise, sunset = _sun_times(target.isoformat())
        return {
            "sunrise": round(_day_fraction(sunrise), 4),
            "sunset": round(_day_fraction(sunset), 4),
        }
    except Exception:  # noqa: BLE001 - astral raises several unrelated types
        return {"sunrise": 0.25, "sunset": 0.79}


def _evenly_sample(rows: list, limit: int) -> list:
    """Thin a chronological list down to `limit`, keeping the day's shape.

    Taking the first N instead would make a busy day's artwork cover only
    its first hour, which reads as "the birds went home at 9am". Sampling
    evenly across the ordered list keeps morning, midday and evening all
    represented, and always keeps the first and last flight of the day.
    """
    n = len(rows)
    if n <= limit:
        return rows
    if limit == 1:
        return [rows[0]]
    return [rows[round(i * (n - 1) / (limit - 1))] for i in range(limit)]


@router.get("/trajectories")
async def get_trajectories(
    date: str | None = Query(
        None, description="Camera-local date, YYYY-MM-DD. Defaults to the most recent active day."
    ),
    limit: int = Query(80, ge=1, le=200, description="Max flights returned, sampled across the day"),
    species_id: int | None = Query(None, description="Restrict to one species"),
    include_unidentified: bool = Query(
        True,
        description="Include detections the classifier rejected (species_id IS NULL). "
        "They are real motion the camera caught and they carry the day's density, "
        "but on an unreviewed day they outnumber labelled birds several to one.",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """One camera-local day of feeder activity as normalised flight paths."""
    target = _resolve_date(db, date)
    start_utc, end_utc = _local_day_bounds(target)

    q = (
        db.query(Detection.id)
        .join(Visit, Detection.visit_id == Visit.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(Visit.started_at >= start_utc, Visit.started_at < end_utc)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(HIDDEN_LABELS))
        )
    )
    if species_id is not None:
        q = q.filter(Detection.species_id == species_id)
    if not include_unidentified:
        q = q.filter(Detection.species_id.isnot(None))

    # Two passes on purpose. The first pulls only ids, in time order, which
    # is cheap even on the busiest day and gives an exact count to sample
    # against; the second hydrates just the flights that survived. Capping
    # the first pass instead would both under-report the day's total and
    # silently drop its evening, which is precisely what the even sampling
    # exists to prevent.
    ordered_ids = [
        row[0] for row in q.order_by(Visit.started_at.asc(), Detection.id.asc()).all()
    ]
    total_available = len(ordered_ids)
    chosen = _evenly_sample(ordered_ids, limit)

    by_id = {
        d.id: d
        for d in db.query(Detection)
        .options(joinedload(Detection.species), joinedload(Detection.visit))
        .filter(Detection.id.in_(chosen))
        .all()
    }
    flights = [_flight(by_id[i]) for i in chosen if i in by_id]

    counts: Counter[str] = Counter(f["species"] for f in flights)
    palette = {}
    for f in flights:
        palette.setdefault(f["species"], f["style"]["primary"])

    return {
        "date": target.isoformat(),
        "tz": settings.camera_timezone,
        "sun": _sun_fractions(target),
        "flights": flights,
        "summary": {
            "returned": len(flights),
            "total_available": total_available,
            "truncated": total_available > len(flights),
            "tracked": sum(1 for f in flights if f["path_kind"] == "tracked"),
            "species_counts": dict(counts.most_common()),
            "species_colors": palette,
        },
    }


def _flight(d: Detection) -> dict[str, Any]:
    """Serialise one detection as a flightline."""
    name = d.species.common_name if d.species else None
    style = species_style(name)
    # The optional arrival chimes play this, not the raw call frequency:
    # real bird calls clash when two land at once. See palette.chime_note.
    style["chime_hz"] = round(chime_note(style["call_hz"], d.id), 2)
    points, kind = flight_points(d.bbox, d.track_bboxes, seed=d.id)

    captured = d.visit.started_at if d.visit else d.created_at
    local = _to_local(captured)

    duration = 0.0
    if d.visit and d.visit.ended_at and d.visit.ended_at > d.visit.started_at:
        duration = min(120.0, (d.visit.ended_at - d.visit.started_at).total_seconds())

    return {
        "detection_id": d.id,
        "visit_id": d.visit_id,
        # Offset-aware, unlike the naive-UTC timestamps the detections feed
        # returns — the client should render this as-is, not append a "Z".
        "started_at": local.isoformat(),
        "time_of_day": round(_day_fraction(local), 5),
        "duration_seconds": round(duration, 1),
        "species": name or "Unidentified",
        "species_id": d.species_id,
        "scientific_name": d.species.scientific_name if d.species else None,
        "confidence": round(d.confidence or 0.0, 3),
        "audio_confirmed": bool(d.audio_confirmed),
        "crop_url": f"/media/{d.crop_path}" if d.crop_path else None,
        "style": style,
        # "tracked" means these are the frames the bird was actually seen
        # in; "synthesized" means only the perch is real. The UI says which.
        "path_kind": kind,
        "points": points,
    }


def _resolve_date(db: Session, raw: str | None) -> _date:
    """Parse the requested date, or find the most recent day with activity."""
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
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(HIDDEN_LABELS))
        )
        .order_by(desc(Visit.started_at))
        .first()
    )
    if latest is None:
        return datetime.now(_tz()).date()
    return _to_local(latest[0]).date()


@router.get("/dates")
async def get_dates(
    limit: int = Query(60, ge=1, le=365, description="How many recent active days to return"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Camera-local days that have feeder activity, newest first.

    Grouped in Python rather than SQL because SQLite's `date()` would bin by
    UTC, and the local/UTC day boundary falls inside the feeder's active
    hours for part of the year. The scan is two indexed columns over the
    detections table; if it ever stops being cheap, PipelineStatsDaily is
    the place to cache it.
    """
    rows = (
        db.query(Visit.started_at, Species.common_name)
        .join(Detection, Detection.visit_id == Visit.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(HIDDEN_LABELS))
        )
        .all()
    )

    per_day: dict[_date, Counter] = defaultdict(Counter)
    totals: Counter[_date] = Counter()
    for started_at, common_name in rows:
        if started_at is None:
            continue
        day = _to_local(started_at).date()
        totals[day] += 1
        if common_name:
            per_day[day][common_name] += 1

    days = sorted(totals, reverse=True)[:limit]
    return {
        "tz": settings.camera_timezone,
        "dates": [
            {
                "date": day.isoformat(),
                "flight_count": totals[day],
                "top_species": [
                    {"species": name, "count": n}
                    for name, n in per_day[day].most_common(3)
                ],
            }
            for day in days
        ],
    }
