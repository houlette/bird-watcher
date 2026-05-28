"""Tests for the daylight gate. Uses fixed lat/lon so sunrise/sunset are
deterministic — the actual production lat/lon comes from settings.py and
doesn't matter for verifying the logic."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline import daylight


@pytest.fixture(autouse=True)
def fixed_location(monkeypatch):
    """Pin the camera to NYC (40.7128, -74.0060) for all tests."""
    monkeypatch.setattr(daylight.settings, "camera_latitude", 40.7128)
    monkeypatch.setattr(daylight.settings, "camera_longitude", -74.0060)
    # The sun-times cache is keyed by date, so clear between tests in case
    # one test computed for a date another test reuses.
    daylight._sun_times.cache_clear()


def test_midday_in_summer_is_daylight():
    """May 27 2026 NYC: noon UTC = 8 AM EDT, full daylight."""
    assert daylight.is_daylight(datetime(2026, 5, 27, 16, 0)) is True  # noon EDT


def test_midnight_is_not_daylight():
    assert daylight.is_daylight(datetime(2026, 5, 27, 4, 0)) is False  # midnight EDT


def test_just_before_dawn_is_not_daylight():
    """May 27 2026 NYC sunrise ~5:30 EDT (9:30 UTC). 9:00 UTC should be dark."""
    assert daylight.is_daylight(datetime(2026, 5, 27, 9, 0)) is False


def test_just_after_sunset_is_not_daylight():
    """May 27 2026 NYC sunset ~8:20 PM EDT (00:20 UTC next day).
    23:30 UTC (7:30 PM EDT) — within the 15-min pre-sunset buffer."""
    assert daylight.is_daylight(datetime(2026, 5, 28, 0, 30)) is False


def test_winter_evening_is_dark_even_at_5pm():
    """Dec 21 2026 NYC sunset ~4:32 PM EST (21:32 UTC). 22:00 UTC = 5pm EST,
    well past sunset even though it'd still be daylight in summer."""
    assert daylight.is_daylight(datetime(2026, 12, 21, 22, 0)) is False


def test_sunset_drifts_seasonally():
    """Same wall-clock hour: bright in summer, dark in winter — proves the
    gate adapts with the calendar rather than using a fixed cutoff."""
    # 11pm UTC = 7pm EDT (summer) or 6pm EST (winter)
    assert daylight.is_daylight(datetime(2026, 6, 21, 23, 0)) is True   # midsummer evening
    assert daylight.is_daylight(datetime(2026, 12, 21, 23, 0)) is False  # midwinter evening
