"""Web Push subscription management + VAPID public key endpoint."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import PushSubscription
from db.session import get_db
from settings import settings

router = APIRouter()


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class SubscribeRequest(BaseModel):
    endpoint: str
    keys: SubscriptionKeys
    notify_window_days: int = 30


@router.get("/vapid_public_key")
async def vapid_public_key() -> dict:
    """Return the VAPID public key so the browser can subscribe via PushManager.

    Empty string when push hasn't been configured yet — the frontend should
    show a 'push not available' state in that case rather than failing hard.
    """
    return {"public_key": settings.vapid_public_key}


@router.post("/subscribe")
async def subscribe(req: SubscribeRequest, db: Session = Depends(get_db)) -> dict:
    existing = db.query(PushSubscription).filter_by(endpoint=req.endpoint).one_or_none()
    if existing:
        existing.p256dh = req.keys.p256dh
        existing.auth = req.keys.auth
        existing.notify_window_days = req.notify_window_days
    else:
        db.add(
            PushSubscription(
                endpoint=req.endpoint,
                p256dh=req.keys.p256dh,
                auth=req.keys.auth,
                notify_window_days=req.notify_window_days,
            )
        )
    db.commit()
    return {"ok": True}


@router.delete("/subscribe")
async def unsubscribe(endpoint: str, db: Session = Depends(get_db)) -> dict:
    db.query(PushSubscription).filter_by(endpoint=endpoint).delete()
    db.commit()
    return {"ok": True}
