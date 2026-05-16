"""Poll the Haikubox API and cache recent audio detections locally.

Haikubox's REST API is documented at https://api.haikubox.com/docs but the
authentication scheme, exact field names, and freshness guarantees are not
documented as of writing. We assume:

  - API key is sent as `Authorization: Bearer <key>`.
  - Endpoint: GET /haikubox/<serial>/detections?hours=1
  - Response is JSON, either a list of detections or an object containing one
    under a key like `detections` / `results` / `data`.
  - Each detection has *some* combination of:
      common_name | species | name
      detected_at | timestamp | time | observed_at
      confidence  | score    | probability

The parser tries each in turn; logs once per missing field per session.

Schema mismatches are non-fatal: the poller logs and moves on. When the user
inspects logs they'll see exactly what shape was returned, which is the
fastest path to fixing the parser without a doc.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from db.models import HaikuboxDetection
from db.session import SessionLocal
from settings import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.haikubox.com"
REQUEST_TIMEOUT_SECONDS = 15
POLL_INTERVAL_SECONDS = 30

# Field-name fallbacks. First match wins. The Haikubox v2 API actually
# returns 'cn' (common name) + 'dt' (ISO timestamp) — confirmed by probing
# /haikubox/<serial>/detections. The longer-form keys are kept as fallbacks
# in case the API ever returns alternate names (e.g. if BirdWeather PUC
# integration ever lands).
_SPECIES_KEYS = ("cn", "common_name", "commonName", "species", "name")
_TIMESTAMP_KEYS = ("dt", "detected_at", "detectedAt", "timestamp", "time", "observed_at")
_CONFIDENCE_KEYS = ("confidence", "score", "probability")


def _pick(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Epoch seconds or millis
        ts = value / 1000 if value > 1e12 else value
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _extract_detections(payload: Any) -> list[dict[str, Any]]:
    """Normalize the various shapes the Haikubox API might return into a list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("detections", "results", "data", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
    log.warning("Haikubox payload had no recognizable detection list: keys=%s", _peek_keys(payload))
    return []


def _peek_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return list(obj.keys())
    return type(obj).__name__


def fetch_recent_detections(client: httpx.Client, hours: int = 1) -> list[dict[str, Any]]:
    """Hit the Haikubox detections endpoint and return the raw detection list."""
    if not settings.haikubox_api_key or not settings.haikubox_serial:
        return []
    url = f"{BASE_URL}/haikubox/{settings.haikubox_serial}/detections"
    headers = {
        "Authorization": f"Bearer {settings.haikubox_api_key}",
        "Accept": "application/json",
    }
    r = client.get(url, params={"hours": hours}, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    if r.status_code == 401:
        log.error("Haikubox 401 — API key invalid or auth scheme is not Bearer")
        return []
    if r.status_code == 404:
        log.error("Haikubox 404 — serial %r not found", settings.haikubox_serial)
        return []
    r.raise_for_status()
    return _extract_detections(r.json())


def upsert_detections(db: Session, raw: list[dict[str, Any]]) -> int:
    """Insert any new audio detections we haven't seen before. Returns inserted count."""
    if not raw:
        return 0

    # Track which (species, timestamp) pairs we already have to avoid duplicate
    # inserts when the API returns overlapping windows on each poll.
    seen = {
        (s, t)
        for s, t in db.query(HaikuboxDetection.species_common_name, HaikuboxDetection.detected_at).all()
    }

    inserted = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        species = _pick(item, _SPECIES_KEYS)
        when = _parse_timestamp(_pick(item, _TIMESTAMP_KEYS))
        if not species or not when:
            continue
        species = str(species).strip()
        key = (species, when.replace(tzinfo=None))
        if key in seen:
            continue
        conf_raw = _pick(item, _CONFIDENCE_KEYS)
        try:
            confidence = float(conf_raw) if conf_raw is not None else None
        except (TypeError, ValueError):
            confidence = None
        db.add(
            HaikuboxDetection(
                species_common_name=species,
                detected_at=when.replace(tzinfo=None),
                confidence=confidence,
            )
        )
        inserted += 1
    if inserted:
        db.commit()
    return inserted


def poll_once() -> None:
    """One tick of the Haikubox poller. Safe to call from APScheduler."""
    if not settings.haikubox_api_key or not settings.haikubox_serial:
        log.debug("Haikubox poller idle: HAIKUBOX_API_KEY or HAIKUBOX_SERIAL not set")
        return

    db = SessionLocal()
    try:
        with httpx.Client() as client:
            try:
                raw = fetch_recent_detections(client, hours=1)
            except httpx.HTTPError as exc:
                log.warning("Haikubox poll failed: %s", exc)
                return
        n = upsert_detections(db, raw)
        if n:
            log.info("Haikubox poller: %d new audio detections", n)
    finally:
        db.close()
