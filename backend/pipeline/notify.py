"""Web Push notifications for rare-species detections and smart notification tiers.

Policy:
  1. Resident species muting: common resident species (Mourning Dove, Rock Pigeon,
     House Sparrow, generic Sparrow) are silenced from real-time push alerts by default
     to eliminate alert fatigue.
  2. Rarity alerting: if a species has not been recorded in `subscription.notify_window_days`
     days (default 30):
       - If never seen before in yard history: "✨ New Yard Visitor: {species}"
       - If returning after absence: "🚨 Rare Visitor: {species} (First visit in N+ days)"
  3. Daily first arrival: if `subscription.notify_daily_first` is enabled (default True),
     regular visitors (Cardinals, Blue Jays, Woodpeckers) send a notification on their
     first arrival each calendar day (camera local time), then remain quiet for subsequent visits.

Each subscription has its own window and tier preferences so the user can tune
sensitivity from the Settings page. Subscriptions that return 404/410 from the push service
are deleted automatically (the browser unsubscribed or revoked permission).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from db.models import Detection, PushSubscription, Species
from db.utils import utcnow
from settings import settings

log = logging.getLogger(__name__)

# Species that visit constantly and dominate feeder traffic (~80% of volume).
# Muted from real-time push by default to eliminate alert fatigue.
RESIDENT_SPECIES = frozenset({
    "Mourning Dove",
    "Rock Pigeon",
    "House Sparrow",
    "Sparrow",
})


def _local_day_start(when: datetime) -> datetime:
    """Return naive UTC datetime corresponding to 00:00:00 of the camera-local calendar day for `when`."""
    tz = ZoneInfo(settings.camera_timezone)
    when_utc = when.replace(tzinfo=timezone.utc)
    local_day = when_utc.astimezone(tz).date()
    start_local = datetime.combine(local_day, time.min, tzinfo=tz)
    return start_local.astimezone(timezone.utc).replace(tzinfo=None)


def is_lifer(db: Session, species_id: int, when: datetime) -> bool:
    """True if this species has NO detections anywhere in the database before `when - 1s`."""
    prior = (
        db.query(Detection.id)
        .filter(Detection.species_id == species_id)
        .filter(Detection.created_at < when - timedelta(seconds=1))
        .first()
    )
    return prior is None


def is_first_today(db: Session, species_id: int, when: datetime) -> bool:
    """True if this species has no prior detection today (since local midnight) before `when - 1s`."""
    start_utc = _local_day_start(when)
    prior = (
        db.query(Detection.id)
        .filter(Detection.species_id == species_id)
        .filter(Detection.created_at >= start_utc)
        .filter(Detection.created_at < when - timedelta(seconds=1))
        .first()
    )
    return prior is None


def is_rare(db: Session, species_id: int, when: datetime, window_days: int) -> bool:
    """True if this species has no other detection in the last window_days
    days *before* `when`. We exclude `when` itself so the call site can pass
    the just-persisted detection's timestamp without trivially self-matching.

    A small fudge factor (1 second) on the upper bound handles the same-instant
    case when multiple tracks from the same visit hit this check concurrently.
    """
    cutoff = when - timedelta(days=window_days)
    prior = (
        db.query(Detection.id)
        .filter(Detection.species_id == species_id)
        .filter(Detection.created_at >= cutoff)
        .filter(Detection.created_at < when - timedelta(seconds=1))
        .first()
    )
    return prior is None


def build_payload(
    detection: Detection,
    species: Species,
    tier: str = "rare",
    window_days: int = 30,
) -> dict:
    """The data the service worker receives when the push arrives."""
    pct = int(detection.confidence * 100)
    if tier == "yard_lifer":
        title = f"✨ New Yard Visitor: {species.common_name}"
        body = f"First time recorded at the feeder! ({pct}% confident)"
    elif tier == "rare":
        title = f"🚨 Rare Visitor: {species.common_name}"
        body = f"First visit in {window_days}+ days ({pct}% confident)"
    elif tier == "daily_first":
        title = species.common_name
        body = f"First visit today at the feeder ({pct}% confident)"
    else:
        title = species.common_name
        body = f"Spotted at the feeder ({pct}% confident)"

    return {
        "title": title,
        "body": body,
        "icon": f"/media/{detection.crop_path}",
        "data": {
            "detection_id": detection.id,
            "visit_id": detection.visit_id,
            "species": species.common_name,
            "tier": tier,
            "url": f"/?d={detection.id}",
        },
    }


def dispatch_for_detection(db: Session, detection: Detection) -> int:
    """Evaluate notification tiers for detection and push to active subscriptions."""
    if not detection.species_id:
        return 0
    species = db.get(Species, detection.species_id)
    if species is None:
        return 0

    if not settings.vapid_public_key:
        log.debug("VAPID not configured — skipping push for detection %d", detection.id)
        return 0

    private_key_path = Path(settings.vapid_private_pem_path)
    if not private_key_path.exists():
        log.warning("VAPID private key missing at %s — skipping push", private_key_path)
        return 0
    vapid_private_key_pem = private_key_path.read_text()

    subs = db.query(PushSubscription).all()
    if not subs:
        return 0

    det_time = detection.created_at or utcnow()
    sent = 0
    for sub in subs:
        mute_res = sub.mute_residents if sub.mute_residents is not None else True
        if mute_res and species.common_name in RESIDENT_SPECIES:
            continue

        window_days = sub.notify_window_days if sub.notify_window_days is not None else 30
        notify_daily = sub.notify_daily_first if sub.notify_daily_first is not None else True

        tier: str | None = None
        if is_rare(db, detection.species_id, det_time, window_days):
            if is_lifer(db, detection.species_id, det_time):
                tier = "yard_lifer"
            else:
                tier = "rare"
        elif notify_daily and is_first_today(db, detection.species_id, det_time):
            tier = "daily_first"

        if tier is None:
            continue

        payload = json.dumps(build_payload(detection, species, tier=tier, window_days=window_days))
        if _send_push(sub, payload, vapid_private_key_pem):
            sent += 1
        else:
            # 404/410 means the subscription is dead — clean up.
            log.info("Removing dead subscription %d", sub.id)
            db.delete(sub)

    if sent:
        log.info("Pushed detection %d (%s) to %d subscriber(s)", detection.id, species.common_name, sent)
    db.commit()
    return sent


def _send_push(sub: PushSubscription, payload: str, vapid_private_pem: str) -> bool:
    """Send one push. Returns True on success, False if the subscription is
    permanently dead (404 / 410) and should be deleted by the caller. Other
    transient errors log and return True so we don't lose a real subscription
    just because the push service is temporarily flaky."""
    from pywebpush import WebPushException, webpush  # noqa: WPS433

    try:
        webpush(
            subscription_info={
                "endpoint": sub.endpoint,
                "keys": {"p256dh": sub.p256dh, "auth": sub.auth},
            },
            data=payload,
            vapid_private_key=vapid_private_pem,
            vapid_claims={"sub": settings.vapid_subject},
        )
        return True
    except WebPushException as exc:
        status = getattr(exc.response, "status_code", None) if exc.response is not None else None
        if status in (404, 410):
            return False  # gone — caller deletes
        log.warning("Push failed for subscription %d (status=%s): %s", sub.id, status, exc)
        return True  # keep the sub; transient
