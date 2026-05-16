"""Web Push notifications for rare-species detections.

Policy: a push is sent when a confirmed detection's species hasn't been
recorded in the last `subscription.notify_window_days` days. With the
default 30-day window, that means:

  - "First time this Baltimore Oriole has been seen since last year" → push.
  - "Another House Sparrow at the feeder" → no push (we saw one yesterday).

Each subscription has its own window so the user can tune sensitivity from
the Settings page. Subscriptions that return 404/410 from the push service
are deleted automatically (the browser unsubscribed or revoked permission).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from db.models import Detection, PushSubscription, Species
from settings import settings

log = logging.getLogger(__name__)


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


def build_payload(detection: Detection, species: Species) -> dict:
    """The data the service worker receives when the push arrives."""
    return {
        "title": species.common_name,
        "body": f"Spotted at the feeder ({int(detection.confidence * 100)}% confident)",
        "icon": f"/media/{detection.crop_path}",
        "data": {
            "detection_id": detection.id,
            "visit_id": detection.visit_id,
            "species": species.common_name,
            "url": f"/?d={detection.id}",
        },
    }


def dispatch_for_detection(db: Session, detection: Detection) -> int:
    """If `detection` qualifies as rare, push to every active subscription
    whose `notify_window_days` makes this detection rare for them."""
    if not detection.species_id:
        return 0
    species = db.query(Species).get(detection.species_id)
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
    payload = json.dumps(build_payload(detection, species))

    subs = db.query(PushSubscription).all()
    sent = 0
    for sub in subs:
        if not is_rare(db, detection.species_id, detection.created_at, sub.notify_window_days):
            continue
        if _send_push(sub, payload, vapid_private_key_pem):
            sent += 1
        else:
            # 404/410 means the subscription is dead — clean up.
            log.info("Removing dead subscription %d", sub.id)
            db.delete(sub)
    if sent:
        log.info("Pushed detection %d (%s) to %d subscriber(s)", detection.id, species.common_name, sent)
    if subs:
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
