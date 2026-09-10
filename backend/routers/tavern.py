"""Read and write API for The Perch & Flagon, the tavern companion page.

The tavern is a second reading of the same data the feed shows: every
detection is a patron walking through a door. Three rules shape the
endpoints here, and all three are the ones that are easy to get wrong.

  1. **A patron is a DETECTION, not a visit.** A visit with forty
     detections is forty tracks, usually forty different birds. Same rule
     the Art page follows, for the same reason.

  2. **The ledger is derived, never stored.** Shillings earned are
     recomputed from the detections table on every read of /state, and
     shillings spent are recomputed from the list of upgrades the user
     owns. Nothing is incremented in place, so the house cannot end up
     richer than the birds that paid for it, and restoring a database
     backup restores the right balance along with the right birds.

  3. **A display toggle never moves the money.** Hiding the unidentified
     crops hides them from the room only. They still pay their single
     shilling into the ledger, because they still happened.
"""
from __future__ import annotations

from collections import Counter
from datetime import date as _date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from db.models import (
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    Detection,
    Species,
    TavernState,
    Visit,
)
from db.session import get_db
from pipeline.daylight import _sun_times
from pipeline.palette import chime_note, species_style
from pipeline.tavern import (
    FOUNDING_PURSE_CAP,
    UPGRADES,
    UPGRADES_BY_ID,
    archetype_for,
    arrival_line,
    dwell_beats,
    lore_for,
    mishap_for,
    order_for,
    patron_payment,
    rarity_tiers,
    spent_on,
)
from settings import settings

router = APIRouter()

# Rows the room does not admit. "Not a bird" comes back as a mishap event
# rather than being dropped, since a wind spinner setting the candles off
# is the whole joke. "Poor quality" is a crop the user retired, and it has
# nothing to show.
NOT_A_PATRON = frozenset({NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL})

# The upgrade that turns the guestbook from a list of names into a book
# with histories in it.
LEDGER_UPGRADE = "guest_ledger"


# ── Local time ──────────────────────────────────────────────────────────
# Visit.started_at is naive UTC. The feeder's meaningful clock is
# settings.camera_timezone, and the hearth should be dark when the yard is,
# so everything user-facing is converted before it leaves this module.


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


def _hearth(now_local: datetime) -> dict[str, Any]:
    """Which way the light is going, for the room's lighting.

    The canvas needs more than "day or night": the window behind the bar
    warms at dawn, goes flat at noon and burns out at dusk. Falls back to
    rough civil hours when astral cannot resolve the date.
    """
    try:
        sunrise, sunset = _sun_times(now_local.date().isoformat())
        rise = sunrise.hour + sunrise.minute / 60
        set_ = sunset.hour + sunset.minute / 60
    except Exception:  # noqa: BLE001 - astral raises several unrelated types
        rise, set_ = 6.0, 19.0

    hour = now_local.hour + now_local.minute / 60
    if hour < rise - 0.75 or hour > set_ + 0.75:
        phase = "night"
    elif hour < rise + 1.0:
        phase = "dawn"
    elif hour > set_ - 1.0:
        phase = "dusk"
    else:
        phase = "day"

    return {
        "phase": phase,
        "sunrise_hour": round(rise, 3),
        "sunset_hour": round(set_, 3),
        "local_hour": round(hour, 3),
    }


# ── Shared query pieces ─────────────────────────────────────────────────


def _patron_query(db: Session):
    """Detections that count as guests, joined to their visit and species."""
    return (
        db.query(Detection)
        .options(joinedload(Detection.species), joinedload(Detection.visit))
        .join(Visit, Detection.visit_id == Visit.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(NOT_A_PATRON))
        )
    )


def _mishap_query(db: Session):
    return (
        db.query(Detection)
        .options(joinedload(Detection.visit))
        .join(Visit, Detection.visit_id == Visit.id)
        .join(Species, Detection.species_id == Species.id)
        .filter(Species.common_name == NOT_A_BIRD_LABEL)
    )


def _yard_tiers(db: Session) -> dict[str, str]:
    """Rarity tier per species, from this yard's own identified sightings."""
    rows = (
        db.query(Species.common_name, func.count(Detection.id))
        .join(Detection, Detection.species_id == Species.id)
        .filter(~Species.common_name.in_(NOT_A_PATRON))
        .group_by(Species.common_name)
        .all()
    )
    return rarity_tiers({name: n for name, n in rows})


def _duration(visit: Visit | None) -> float:
    if visit is None or visit.ended_at is None:
        return 0.0
    seconds = (visit.ended_at - visit.started_at).total_seconds()
    return max(0.0, seconds)


def _state_row(db: Session) -> TavernState:
    """The one saved row, created on first contact.

    `opened_at` is stamped here, which makes the first request to any
    tavern endpoint the moment the house changed hands. Everything the
    camera recorded before that instant is history the user inherited.
    """
    row = db.get(TavernState, 1)
    if row is None:
        row = TavernState(
            id=1,
            unlocked=[],
            opened_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


# ── Serialisation ───────────────────────────────────────────────────────


def _patron(d: Detection, tiers: dict[str, str], *, stale: bool = False) -> dict[str, Any]:
    """One detection dressed as a guest of the house."""
    name = d.species.common_name if d.species else None
    style = species_style(name)
    style["chime_hz"] = round(chime_note(style["call_hz"], d.id), 2)

    arch = archetype_for(name, style["mass_g"])
    tier = tiers.get(name, "rare") if name else "common"
    duration = _duration(d.visit)
    captured = d.visit.started_at if d.visit else d.created_at

    return {
        "detection_id": d.id,
        "visit_id": d.visit_id,
        "species": name,
        "species_id": d.species_id,
        "scientific_name": d.species.scientific_name if d.species else None,
        # Offset-aware, in the feeder's zone. Render as-is; appending a "Z"
        # would move every arrival by the local offset.
        "arrived_at": _to_local(captured).isoformat(),
        "duration_seconds": round(duration, 1),
        "audio_confirmed": bool(d.audio_confirmed),
        "confidence": round(d.confidence or 0.0, 3),
        "crop_url": f"/media/{d.crop_path}" if d.crop_path else None,
        "archetype": {
            "id": arch["id"],
            "title": arch["title"],
            "role": arch["role"],
            "seat": arch["seat"],
            "blurb": arch["blurb"],
        },
        "style": style,
        "rarity": tier,
        "payment": patron_payment(
            archetype_id=arch["id"],
            tier=tier,
            duration_seconds=duration,
            audio_confirmed=bool(d.audio_confirmed),
        ),
        "order": order_for(arch["id"], d.id),
        "line": arrival_line(arch["id"], d.id),
        "dwell": dwell_beats(arch["id"], duration),
        # True when this guest is being shown because the room would
        # otherwise be empty, not because they just walked in.
        "stale": stale,
    }


def _event(d: Detection) -> dict[str, Any]:
    captured = d.visit.started_at if d.visit else d.created_at
    mishap = mishap_for(d.id)
    return {
        "detection_id": d.id,
        "visit_id": d.visit_id,
        "at": _to_local(captured).isoformat(),
        "crop_url": f"/media/{d.crop_path}" if d.crop_path else None,
        **mishap,
    }


def _ledger(db: Session, state: TavernState, tiers: dict[str, str]) -> dict[str, Any]:
    """Earnings recomputed from scratch, every time.

    One pass over the detections table, which is a few thousand rows here.
    If the yard ever outgrows that, PipelineStatsDaily is where a per-day
    cache belongs, keyed the same way the funnel counts are.

    Two pots, split at the moment the user first opened the tavern. Birds
    that landed before that pay into a founding purse that stops at
    FOUNDING_PURSE_CAP, so a long archive is a good start rather than an
    instant win; birds that land afterwards pay in full.
    """
    unlocked = list(state.unlocked or [])
    opened_at = state.opened_at
    rows = (
        db.query(
            Species.common_name,
            Visit.started_at,
            Visit.ended_at,
            Detection.audio_confirmed,
        )
        .select_from(Detection)
        .join(Visit, Detection.visit_id == Visit.id)
        .outerjoin(Species, Detection.species_id == Species.id)
        .filter(
            (Species.common_name.is_(None)) | (~Species.common_name.in_(NOT_A_PATRON))
        )
        .all()
    )

    inherited = 0
    since_opening = 0
    named = 0
    for name, started, ended, audio in rows:
        duration = 0.0
        if ended is not None and started is not None:
            duration = max(0.0, (ended - started).total_seconds())
        arch = archetype_for(name)
        if name:
            named += 1
        paid = patron_payment(
            archetype_id=arch["id"],
            tier=tiers.get(name, "rare") if name else "common",
            duration_seconds=duration,
            audio_confirmed=bool(audio),
        )
        if opened_at is not None and started is not None and started < opened_at:
            inherited += paid
        else:
            since_opening += paid

    purse = min(inherited, FOUNDING_PURSE_CAP)
    earned = purse + since_opening
    spent = spent_on(unlocked)
    return {
        # What the house has to spend: the purse plus the takings since.
        "earned": earned,
        "founding_purse": purse,
        "purse_cap": FOUNDING_PURSE_CAP,
        # Everything the birds ever paid, including the part of the
        # inheritance the strongbox could not hold. Shown as a statistic,
        # not as money.
        "earned_all_time": inherited + since_opening,
        "since_opening": since_opening,
        "spent": spent,
        "balance": earned - spent,
        "patrons_served": len(rows),
        "named_patrons": named,
    }


def _upgrade_list(unlocked: list[str], balance: int) -> list[dict[str, Any]]:
    owned = set(unlocked)
    return [
        {
            **u,
            "owned": u["id"] in owned,
            "affordable": u["id"] not in owned and balance >= u["cost"],
        }
        for u in UPGRADES
    ]


# ── Endpoints ───────────────────────────────────────────────────────────


@router.get("/state")
async def get_state(
    window_minutes: int = Query(
        180, ge=5, le=1440, description="How far back an arrival still counts as being in the room"
    ),
    limit: int = Query(24, ge=1, le=80, description="Most guests to seat at once"),
    strangers: bool = Query(
        True,
        description="Seat detections the classifier would not name. They pay their "
        "shilling into the ledger either way; this only decides whether they are "
        "drawn in the room.",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Everything the tavern page needs for a first paint."""
    state = _state_row(db)
    unlocked = list(state.unlocked or [])
    tiers = _yard_tiers(db)
    ledger = _ledger(db, state, tiers)

    now_local = datetime.now(_tz())
    cutoff = (now_local - timedelta(minutes=window_minutes)).astimezone(
        timezone.utc
    ).replace(tzinfo=None)

    q = _patron_query(db)
    if not strangers:
        q = q.filter(Detection.species_id.isnot(None))

    recent = (
        q.filter(Visit.started_at >= cutoff)
        .order_by(desc(Visit.started_at), desc(Detection.id))
        .limit(limit)
        .all()
    )

    # An empty room at midnight is accurate and dull. Fall back to the last
    # guests the camera saw and mark each one stale, so the page can say
    # "the room is quiet, here is who was in it" without implying that a
    # dove from Tuesday is currently at the bar.
    quiet = not recent
    if quiet:
        recent = (
            q.order_by(desc(Visit.started_at), desc(Detection.id)).limit(limit).all()
        )

    patrons = [_patron(d, tiers, stale=quiet) for d in recent]

    events = [
        _event(d)
        for d in _mishap_query(db)
        .filter(Visit.started_at >= cutoff)
        .order_by(desc(Visit.started_at))
        .limit(8)
        .all()
    ]

    day_start, day_end = _local_day_bounds(now_local.date())
    today_rows = (
        _patron_query(db)
        .filter(Visit.started_at >= day_start, Visit.started_at < day_end)
        .all()
    )
    today_names = Counter(
        d.species.common_name for d in today_rows if d.species is not None
    )

    newest = db.query(func.max(Detection.id)).scalar() or 0

    # How much walked in while the page was closed. Counted against the
    # marker /seen keeps rather than against a timestamp, so it means "new
    # to you" even when the worker drained a backlog out of order.
    seen_mark = state.last_seen_detection_id
    since_last_seen = (
        _patron_query(db).filter(Detection.id > seen_mark).count() if seen_mark else 0
    )

    return {
        "tz": settings.camera_timezone,
        "now": now_local.isoformat(),
        "hearth": _hearth(now_local),
        "quiet": quiet,
        "window_minutes": window_minutes,
        "patrons": patrons,
        "events": events,
        "ledger": ledger,
        "upgrades": _upgrade_list(unlocked, ledger["balance"]),
        "unlocked": unlocked,
        "today": {
            "arrivals": len(today_rows),
            "species": len(today_names),
            "top": [{"species": n, "count": c} for n, c in today_names.most_common(4)],
        },
        # Poll /arrivals with this to get whatever lands next.
        "cursor": newest,
        "last_seen_detection_id": seen_mark,
        "since_last_seen": since_last_seen,
        "opened_at": _to_local(state.opened_at).isoformat() if state.opened_at else None,
    }


@router.get("/arrivals")
async def get_arrivals(
    after: int = Query(..., ge=0, description="Highest detection id already shown"),
    limit: int = Query(20, ge=1, le=60),
    strangers: bool = Query(True),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Guests and mishaps that arrived since `after`, oldest first.

    Ordered by detection id rather than capture time on purpose. The worker
    can finish a backlog out of order, and the page wants "what is new to
    me", which is what the id tracks. Returning them oldest first lets the
    client seat them in the order they were processed.
    """
    tiers = _yard_tiers(db)

    q = _patron_query(db).filter(Detection.id > after)
    if not strangers:
        q = q.filter(Detection.species_id.isnot(None))
    rows = q.order_by(Detection.id.asc()).limit(limit).all()
    patrons = [_patron(d, tiers) for d in rows]

    events = [
        _event(d)
        for d in _mishap_query(db)
        .filter(Detection.id > after)
        .order_by(Detection.id.asc())
        .limit(limit)
        .all()
    ]

    newest = db.query(func.max(Detection.id)).scalar() or after
    return {
        "cursor": newest,
        "patrons": patrons,
        "events": events,
        # What this batch alone put in the till, so the page can add to the
        # balance it already holds instead of refetching the whole ledger.
        "earned": sum(p["payment"] for p in patrons),
    }


@router.get("/guestbook")
async def get_guestbook(
    limit: int = Query(60, ge=1, le=300),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """One entry per species that has ever been seated, busiest first.

    Histories are withheld until the guest ledger is bought. That is the
    one upgrade with a mechanical effect rather than a decorative one, and
    the withheld text is invented anyway, so nothing factual is gated.
    """
    state = _state_row(db)
    unlocked = list(state.unlocked or [])
    has_ledger = LEDGER_UPGRADE in unlocked
    tiers = _yard_tiers(db)

    rows = (
        db.query(
            Species.common_name,
            Species.scientific_name,
            Species.id,
            func.count(Detection.id),
            func.min(Visit.started_at),
            func.max(Visit.started_at),
            func.sum(Detection.audio_confirmed),
        )
        .select_from(Detection)
        .join(Visit, Detection.visit_id == Visit.id)
        .join(Species, Detection.species_id == Species.id)
        .filter(~Species.common_name.in_(NOT_A_PATRON))
        .group_by(Species.id)
        .order_by(desc(func.count(Detection.id)))
        .limit(limit)
        .all()
    )

    entries = []
    for name, sci, sid, visits, first, last, heard in rows:
        style = species_style(name)
        arch = archetype_for(name, style["mass_g"])
        tier = tiers.get(name, "rare")
        entry = {
            "species": name,
            "species_id": sid,
            "scientific_name": sci,
            "visits": int(visits),
            "first_seen": _to_local(first).isoformat() if first else None,
            "last_seen": _to_local(last).isoformat() if last else None,
            "times_heard": int(heard or 0),
            "rarity": tier,
            "style": style,
            "archetype": {
                "id": arch["id"],
                "title": arch["title"],
                "role": arch["role"],
                "seat": arch["seat"],
                "blurb": arch["blurb"],
            },
        }
        entry["lore"] = lore_for(name, arch["id"]) if has_ledger else None
        entries.append(entry)

    return {"has_ledger": has_ledger, "entries": entries}


@router.post("/unlock")
async def unlock_upgrade(
    upgrade_id: str = Body(..., embed=True),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Buy one upgrade, if the house can afford it."""
    upgrade = UPGRADES_BY_ID.get(upgrade_id)
    if upgrade is None:
        raise HTTPException(status_code=404, detail=f"No such upgrade: {upgrade_id}")

    state = _state_row(db)
    unlocked = list(state.unlocked or [])
    if upgrade_id in unlocked:
        raise HTTPException(status_code=409, detail="Already part of the house")

    tiers = _yard_tiers(db)
    ledger = _ledger(db, state, tiers)
    if ledger["balance"] < upgrade["cost"]:
        raise HTTPException(
            status_code=402,
            detail=f"{upgrade['cost'] - ledger['balance']} more shillings needed",
        )

    unlocked.append(upgrade_id)
    # Reassigned rather than appended in place: SQLAlchemy's JSON column
    # does not track mutations of the list it handed out, so an append
    # alone would be dropped on commit.
    state.unlocked = unlocked
    db.commit()

    ledger = _ledger(db, state, tiers)
    return {
        "unlocked": unlocked,
        "ledger": ledger,
        "upgrades": _upgrade_list(unlocked, ledger["balance"]),
        "bought": upgrade,
    }


@router.post("/seen")
async def mark_seen(
    detection_id: int = Body(..., embed=True),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Remember the newest guest the user has actually watched arrive.

    Only ever moves forward. Two tabs open on the same tavern would
    otherwise take turns walking the marker backwards and replaying the
    same arrivals at each other.
    """
    state = _state_row(db)
    current = state.last_seen_detection_id or 0
    if detection_id > current:
        state.last_seen_detection_id = detection_id
        db.commit()
    return {"last_seen_detection_id": state.last_seen_detection_id}
