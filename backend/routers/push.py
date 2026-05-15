"""Web Push subscription management (Phase 5 wiring; stubbed for Phase 1)."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.models import PushSubscription
from db.session import get_db

router = APIRouter()


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class SubscribeRequest(BaseModel):
    endpoint: str
    keys: SubscriptionKeys
    rare_threshold: int = 5


@router.post("/subscribe")
async def subscribe(req: SubscribeRequest, db: Session = Depends(get_db)) -> dict:
    existing = db.query(PushSubscription).filter_by(endpoint=req.endpoint).one_or_none()
    if existing:
        existing.p256dh = req.keys.p256dh
        existing.auth = req.keys.auth
        existing.rare_threshold = req.rare_threshold
    else:
        db.add(
            PushSubscription(
                endpoint=req.endpoint,
                p256dh=req.keys.p256dh,
                auth=req.keys.auth,
                rare_threshold=req.rare_threshold,
            )
        )
    db.commit()
    return {"ok": True}


@router.delete("/subscribe")
async def unsubscribe(endpoint: str, db: Session = Depends(get_db)) -> dict:
    db.query(PushSubscription).filter_by(endpoint=endpoint).delete()
    db.commit()
    return {"ok": True}
