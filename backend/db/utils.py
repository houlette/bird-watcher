"""Database-facing utility helpers.

`utcnow()` exists because:

  - Python 3.12 deprecated `datetime.datetime.utcnow()` and started warning.
  - The recommended replacement `datetime.now(UTC)` returns a *timezone-aware*
    datetime, but our SQLAlchemy columns are typed `DateTime` (naive UTC) —
    mixing the two raises a separate SQLAlchemy warning and ultimately breaks
    comparisons.

This helper returns naive UTC: the same shape we've been using all along,
but produced via the non-deprecated API. Call it everywhere instead of
`datetime.utcnow()`.
"""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Naive UTC datetime (tzinfo stripped) — the on-disk format."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
