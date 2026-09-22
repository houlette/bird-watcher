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
