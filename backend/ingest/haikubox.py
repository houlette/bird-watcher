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
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy.orm import Session

from db.models import (
    SENTINEL_LABELS,
    Detection,
    HaikuboxDetection,
    Species,
    Visit,
)
from db.session import SessionLocal
from db.utils import utcnow
from settings import settings

log = logging.getLogger(__name__)

BASE_URL = "https://api.haikubox.com"
REQUEST_TIMEOUT_SECONDS = 15
POLL_INTERVAL_SECONDS = 30

# How far back the daily widening pull asks for. The live poller fetches
# the last hour every 30 s; this once-a-day pass covers any audio the
# live poller missed (transient API failures, container restarts,
# scheduler skips during a backlog drain). 24 h is the comfortable upper
# bound — the API documents `?hours=N` but its actual ceiling isn't
# spec'd, so widening past a day risks getting silently truncated.
BACKFILL_HOURS = 24

# After backfilling audio, we walk Detection rows captured in this window
# and re-evaluate `audio_confirmed`. A bit wider than BACKFILL_HOURS so a
# visit captured near the boundary still gets a fair look. Bounded so the
# re-correlation pass is cheap even on a busy yard.
RECORRELATE_LOOKBACK_HOURS = 48

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


def _recorrelate_recent_detections(db: Session) -> int:
    """Walk recently-captured detections that aren't audio-confirmed yet
    and re-check them against the (now-wider) audio cache.

    Only touches Detection rows with a real-species label — sentinel
    labels (NAB / Unknown / Poor quality) and Unidentified (species_id
    IS NULL) can't be audio-confirmed by definition. The audio
    correlation window matches fuse._audio_species_set: `[when - W, when]`
    where W = settings.audio_correlation_window_seconds. Same semantics
    as live processing, just executed against a fuller cache.
    """
    cutoff = utcnow() - timedelta(hours=RECORRELATE_LOOKBACK_HOURS)
    window_s = settings.audio_correlation_window_seconds
    rows = (
        db.query(Detection, Species.common_name, Visit.started_at)
        .join(Species, Detection.species_id == Species.id)
        .join(Visit, Detection.visit_id == Visit.id)
        .filter(Detection.audio_confirmed == 0)
        .filter(Visit.started_at >= cutoff)
        .filter(~Species.common_name.in_(SENTINEL_LABELS))
        .all()
    )
    if not rows:
        return 0

    # Pull the unique species + their windows we need to check, then
    # do one query per species (rather than per-detection) so a busy
    # yard with hundreds of recent detections doesn't fan out into
    # hundreds of round-trips.
    confirmed = 0
    species_windows: dict[str, list[tuple[Detection, datetime]]] = {}
    for det, species_name, when in rows:
        species_windows.setdefault(species_name, []).append((det, when))

    for species_name, items in species_windows.items():
        # Earliest + latest capture for this species; pull every audio row
        # for the species in that span and match in Python. Cheap because
        # one species rarely has more than a handful of audio rows per day.
        earliest = min(when for _, when in items) - timedelta(seconds=window_s)
        latest = max(when for _, when in items)
        audio_times = [
            t for (t,) in (
                db.query(HaikuboxDetection.detected_at)
                .filter(HaikuboxDetection.species_common_name == species_name)
                .filter(HaikuboxDetection.detected_at >= earliest)
                .filter(HaikuboxDetection.detected_at <= latest)
                .all()
            )
        ]
        if not audio_times:
            continue
        audio_times.sort()
        for det, when in items:
            lo = when - timedelta(seconds=window_s)
            hi = when
            # Linear scan is fine — handful of audio rows per species
            # per day. If this ever becomes a bottleneck, bisect.
            if any(lo <= t <= hi for t in audio_times):
                det.audio_confirmed = True
                confirmed += 1

    if confirmed:
        db.commit()
    return confirmed


def backfill_and_recorrelate() -> dict[str, int]:
    """Daily widening pass: pull the last BACKFILL_HOURS of audio, then
    re-evaluate `audio_confirmed` on recent detections that the live
    poller never got a chance to correlate.

    Safe to call from APScheduler. Returns the work counts so the
    caller can log / surface metrics. No-op when the API isn't
    configured.
    """
    if not (settings.haikubox_api_key and settings.haikubox_serial):
        log.debug("Haikubox backfill idle: HAIKUBOX_API_KEY or HAIKUBOX_SERIAL unset")
        return {"inserted": 0, "confirmed": 0}

    db = SessionLocal()
    inserted = 0
    confirmed = 0
    try:
        with httpx.Client() as client:
            try:
                raw = fetch_recent_detections(client, hours=BACKFILL_HOURS)
            except httpx.HTTPError as exc:
                log.warning("Haikubox %dh backfill fetch failed: %s", BACKFILL_HOURS, exc)
                raw = []
        inserted = upsert_detections(db, raw)
        confirmed = _recorrelate_recent_detections(db)
    finally:
        db.close()

    log.info(
        "Haikubox backfill (%dh): %d audio rows inserted, %d detection(s) newly audio-confirmed",
        BACKFILL_HOURS, inserted, confirmed,
    )
    return {"inserted": inserted, "confirmed": confirmed}
