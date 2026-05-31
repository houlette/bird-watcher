"""Pipeline stats endpoint backing the /stats page in the PWA."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from db.models import PipelineStatsDaily
from db.session import get_db
from pipeline.stats import (
    compute_daily_stats,
    compute_global_stats,
    serialize_daily,
)

router = APIRouter()


@router.get("")
async def get_stats(
    days: int = Query(30, ge=1, le=365, description="How many trailing days of daily stats to return."),
    db: Session = Depends(get_db),
) -> dict:
    """Return last N days of pipeline funnel stats plus all-time totals.

    Today's row is recomputed live on every request so the page reflects
    the freshest visits without waiting for the nightly job. Historical
    rows are served from the PipelineStatsDaily snapshot table.
    """
    today = datetime.now(timezone.utc).date()
    earliest = today - timedelta(days=days - 1)

    cached = (
        db.query(PipelineStatsDaily)
        .filter(PipelineStatsDaily.date >= earliest, PipelineStatsDaily.date < today)
        .order_by(PipelineStatsDaily.date.asc())
        .all()
    )
    # Recompute today's row inline (NOT persisted — the nightly job owns
    # writes for yesterday's row, and recomputing-on-read for today is
    # cheap enough on a single-day window).
    today_row = compute_daily_stats(db, today)

    daily = [serialize_daily(r) for r in cached] + [serialize_daily(today_row)]

    # Fill gaps: dates in [earliest, today] that have neither a cached row
    # nor any visits should still appear as zeroed entries so the chart
    # has a continuous x-axis.
    have_dates = {row["date"] for row in daily}
    filled: list[dict] = []
    d = earliest
    while d <= today:
        if d.isoformat() not in have_dates:
            # Zero-fill via compute_daily_stats over an empty day rather
            # than constructing a bare PipelineStatsDaily — SQLAlchemy
            # column defaults only apply on flush, so a hand-built
            # instance would have None where we want 0.
            filled.append(serialize_daily(compute_daily_stats(db, d)))
        d = d + timedelta(days=1)
    daily.extend(filled)
    daily.sort(key=lambda r: r["date"])

    totals = compute_global_stats(db)

    return {
        "daily": daily,
        "totals": totals,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
