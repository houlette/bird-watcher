"""Tests for /api/push router: VAPID public key and push subscription lifecycle."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import PushSubscription
from db.session import Base, get_db
from main import app
from settings import settings


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_vapid_public_key_empty(client, monkeypatch):
    """Returns empty public key string if push is not configured."""
    monkeypatch.setattr(settings, "vapid_public_key", "")
    res = client.get("/api/push/vapid_public_key")
    assert res.status_code == 200
    assert res.json() == {"public_key": ""}


def test_vapid_public_key_configured(client, monkeypatch):
    """Returns the configured VAPID public key."""
    monkeypatch.setattr(settings, "vapid_public_key", "test-public-key-xyz")
    res = client.get("/api/push/vapid_public_key")
    assert res.status_code == 200
    assert res.json() == {"public_key": "test-public-key-xyz"}


def test_subscribe_create_and_update(client, db_session):
    """POST /api/push/subscribe creates a new subscription and updates on conflict."""
    payload = {
        "endpoint": "https://push.example.com/sub/12345",
        "keys": {
            "p256dh": "key_p256dh_val",
            "auth": "auth_val",
        },
        "notify_window_days": 14,
    }
    # Initial subscription creation
    res = client.post("/api/push/subscribe", json=payload)
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    sub = db_session.query(PushSubscription).filter_by(endpoint=payload["endpoint"]).one_or_none()
    assert sub is not None
    assert sub.p256dh == "key_p256dh_val"
    assert sub.auth == "auth_val"
    assert sub.notify_window_days == 14

    # Update existing subscription
    payload_updated = {
        "endpoint": "https://push.example.com/sub/12345",
        "keys": {
            "p256dh": "new_key_p256dh",
            "auth": "new_auth",
        },
        "notify_window_days": 60,
    }
    res = client.post("/api/push/subscribe", json=payload_updated)
    assert res.status_code == 200

    subs = db_session.query(PushSubscription).filter_by(endpoint=payload["endpoint"]).all()
    assert len(subs) == 1
    assert subs[0].p256dh == "new_key_p256dh"
    assert subs[0].auth == "new_auth"
    assert subs[0].notify_window_days == 60


def test_unsubscribe(client, db_session):
    """DELETE /api/push/subscribe removes the subscription by endpoint."""
    sub = PushSubscription(
        endpoint="https://push.example.com/sub/to_delete",
        p256dh="key1",
        auth="auth1",
        notify_window_days=30,
    )
    db_session.add(sub)
    db_session.commit()

    assert db_session.query(PushSubscription).filter_by(endpoint="https://push.example.com/sub/to_delete").count() == 1

    res = client.delete("/api/push/subscribe", params={"endpoint": "https://push.example.com/sub/to_delete"})
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    assert db_session.query(PushSubscription).filter_by(endpoint="https://push.example.com/sub/to_delete").count() == 0

    # Idempotent delete on non-existent endpoint
    res2 = client.delete("/api/push/subscribe", params={"endpoint": "https://push.example.com/sub/to_delete"})
    assert res2.status_code == 200
    assert res2.json() == {"ok": True}


def test_subscribe_rejects_ssrf_and_invalid_endpoints(client):
    """POST /api/push/subscribe rejects non-https, localhost, and private IP endpoints."""
    bad_endpoints = [
        "http://push.example.com/sub/123",  # Non-HTTPS
        "https://localhost/push",           # Localhost
        "https://127.0.0.1/push",           # Loopback IP
        "https://169.254.169.254/latest",   # AWS/GCP Link-local metadata
        "https://10.0.0.1/push",            # RFC 1918 10.x.x.x
        "https://192.168.1.1/push",         # RFC 1918 192.168.x.x
        "https://172.16.0.1/push",          # RFC 1918 172.16.x.x
        "https://internal.local/push",      # .local internal domain
    ]
    for ep in bad_endpoints:
        res = client.post(
            "/api/push/subscribe",
            json={
                "endpoint": ep,
                "keys": {"p256dh": "key", "auth": "auth"},
            },
        )
        assert res.status_code == 400, f"Expected 400 for {ep}, got {res.status_code}"


def test_subscribe_with_smart_tier_preferences(client, db_session):
    """POST /api/push/subscribe sets mute_residents and notify_daily_first."""
    payload = {
        "endpoint": "https://push.example.com/sub/tiers",
        "keys": {"p256dh": "k1", "auth": "a1"},
        "notify_window_days": 15,
        "mute_residents": False,
        "notify_daily_first": False,
    }
    res = client.post("/api/push/subscribe", json=payload)
    assert res.status_code == 200

    sub = db_session.query(PushSubscription).filter_by(endpoint=payload["endpoint"]).one()
    assert sub.mute_residents is False
    assert sub.notify_daily_first is False
    assert sub.notify_window_days == 15

    # Test GET /api/push/subscription returns the saved preferences
    get_res = client.get("/api/push/subscription", params={"endpoint": payload["endpoint"]})
    assert get_res.status_code == 200
    assert get_res.json() == {
        "endpoint": payload["endpoint"],
        "notify_window_days": 15,
        "mute_residents": False,
        "notify_daily_first": False,
    }


def test_get_subscription_not_found(client):
    """GET /api/push/subscription returns 404 for unknown endpoint."""
    res = client.get("/api/push/subscription", params={"endpoint": "https://push.example.com/sub/unknown"})
    assert res.status_code == 404

