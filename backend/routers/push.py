import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.models import PushSubscription
from db.session import get_db
from settings import settings

router = APIRouter()


def _is_private_or_reserved_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except ValueError:
        return False


def validate_push_endpoint(endpoint: str) -> str:
    """Validate push endpoint URL to prevent SSRF against internal hosts or metadata services."""
    try:
        parsed = urlparse(endpoint)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid push endpoint URL") from exc

    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="Push endpoint must use HTTPS scheme")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Push endpoint missing hostname")

    lower_host = hostname.lower()
    if lower_host in {"localhost"} or lower_host.endswith((".localhost", ".local", ".internal", ".arpa")):
        raise HTTPException(status_code=400, detail="Push endpoint cannot point to local hostnames")

    if _is_private_or_reserved_ip(lower_host):
        raise HTTPException(status_code=400, detail="Push endpoint cannot point to private or reserved IP addresses")

    # If domain resolves, ensure it does not resolve to private/loopback/link-local IP
    try:
        addr_info = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        for _, _, _, _, sockaddr in addr_info:
            if _is_private_or_reserved_ip(sockaddr[0]):
                raise HTTPException(status_code=400, detail="Push endpoint resolves to a private or reserved IP")
    except socket.gaierror:
        # In test or offline environments, unresolvable domains are permitted as long as not IP literals
        pass

    return endpoint


class SubscriptionKeys(BaseModel):
    p256dh: str = Field(..., max_length=256)
    auth: str = Field(..., max_length=256)


class SubscribeRequest(BaseModel):
    endpoint: str = Field(..., max_length=2048)
    keys: SubscriptionKeys
    notify_window_days: int = Field(30, ge=1, le=365)


@router.get("/vapid_public_key")
async def vapid_public_key() -> dict:
    """Return the VAPID public key so the browser can subscribe via PushManager.

    Empty string when push hasn't been configured yet — the frontend should
    show a 'push not available' state in that case rather than failing hard.
    """
    return {"public_key": settings.vapid_public_key}


@router.post("/subscribe")
async def subscribe(req: SubscribeRequest, db: Session = Depends(get_db)) -> dict:
    validate_push_endpoint(req.endpoint)
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
