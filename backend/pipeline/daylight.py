"""Daylight gating: drop visits captured after the sun's down.

The Reolink switches to IR/grayscale below a certain ambient-light
threshold; the species classifier wasn't trained on IR footage and birds
aren't out hunting feeders at night anyway, so processing those clips
just produces a stream of unidentifiable shadow shapes in the feed.

A fixed time-of-day cutoff would drift with the seasons (May sunset is
~8:20 PM, November sunset is ~4:40 PM). Compute the actual sunrise /
sunset for the camera's lat/lon and compare against the clip's capture
timestamp.

Edge: civil-twilight vs sunset. Civil twilight extends ~30 min after
sunset and there's still usable light then — but the Reolink we have in
the field switches to IR *before* sunset on overcast days, so using
sunset itself (the more conservative cutoff) matches our observed
camera behavior better than civil dusk.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from astral import LocationInfo
from astral.sun import sun

from settings import settings

log = logging.getLogger(__name__)

# Small safety margin: stop processing this many minutes before sunset and
# start this many minutes before sunrise. The Reolink's IR transition is
# light-driven rather than clock-driven and tends to fire a bit before
# astronomical sunset on overcast days, so a small buffer matches reality.
PRE_SUNSET_BUFFER_MINUTES = 15
POST_SUNRISE_BUFFER_MINUTES = 15


@lru_cache(maxsize=8)
def _sun_times(local_date_iso: str) -> tuple[datetime, datetime]:
    """Sunrise and sunset for the camera's location on a given LOCAL date.

    We compute against the LOCAL date — not UTC — because in any non-UTC
    timezone the sunset can fall on a different UTC calendar day than the
    local day. Astral interprets its `date` argument in whatever tzinfo
    we pass, so giving it the camera's IANA zone resolves the ambiguity.

    Returns aware datetimes in the camera's local timezone; callers should
    convert their inputs to the same zone before comparing.

    Cached because every Visit on the same local day shares the answer
    and astral's calculation, while cheap, isn't free. The cache is keyed
    by ISO local-date string.
    """
    from datetime import date as _date

    tz = ZoneInfo(settings.camera_timezone)
    loc = LocationInfo(
        latitude=settings.camera_latitude,
        longitude=settings.camera_longitude,
        timezone=settings.camera_timezone,
    )
    d = _date.fromisoformat(local_date_iso)
    s = sun(loc.observer, date=d, tzinfo=tz)
    return s["sunrise"], s["sunset"]


def is_daylight(captured_at: datetime) -> bool:
    """True if `captured_at` falls between sunrise and sunset (with a small
    buffer to match the Reolink's empirically-conservative IR trigger).

    `captured_at` is naive UTC (matches our DB convention) or aware in any
    timezone. We convert to the camera's local zone and compare against
    that day's local sunrise / sunset.
    """
    if captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(settings.camera_timezone)
    local = captured_at.astimezone(tz)
    sunrise, sunset = _sun_times(local.date().isoformat())
    usable_start = sunrise + timedelta(minutes=POST_SUNRISE_BUFFER_MINUTES)
    usable_end = sunset - timedelta(minutes=PRE_SUNSET_BUFFER_MINUTES)
    return usable_start <= local <= usable_end
