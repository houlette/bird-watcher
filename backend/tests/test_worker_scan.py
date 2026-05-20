"""Tests for the filesystem-scan path that bridges SFTP uploads to Visit rows.

We don't exercise the APScheduler harness — just the function that finds new
clip files on disk and creates Visit rows for them. Tests use a real on-disk
clips directory inside tmp_path and monkeypatch the module's CLIPS_DIR.
"""
from __future__ import annotations

import os
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Visit
from db.session import Base
from pipeline import worker


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal


@pytest.fixture()
def clips_dir(tmp_path, monkeypatch):
    d = tmp_path / "clips"
    d.mkdir()
    monkeypatch.setattr(worker, "CLIPS_DIR", d)
    return d


def _make_file(path, age_seconds: float = 60.0) -> None:
    path.write_bytes(b"\xff\xd8\xff\xe0fake jpg payload")
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))


def test_scan_creates_visit_for_new_jpg(db, clips_dir, monkeypatch):
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "RecS_20260520_001234.jpg", age_seconds=60)
    n = worker._scan_clips_dir()
    assert n == 1

    session: Session = db()
    try:
        visits = session.query(Visit).all()
        assert len(visits) == 1
        assert visits[0].clip_path == "clips/RecS_20260520_001234.jpg"
        assert visits[0].processed_at is None  # pending, will be picked up next tick
    finally:
        session.close()


def test_scan_skips_files_still_being_written(db, clips_dir, monkeypatch):
    """A file with a very recent mtime might still be uploading; skip it."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "partial.jpg", age_seconds=1)
    n = worker._scan_clips_dir()
    assert n == 0


def test_scan_skips_files_already_tracked(db, clips_dir, monkeypatch):
    """Running scan twice on the same file should only create one Visit."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "RecS_repeat.jpg", age_seconds=60)
    assert worker._scan_clips_dir() == 1
    assert worker._scan_clips_dir() == 0

    session = db()
    try:
        assert session.query(Visit).count() == 1
    finally:
        session.close()


def test_scan_ignores_non_media_files(db, clips_dir, monkeypatch):
    """A README.txt sitting in the clips dir shouldn't become a Visit."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "README.txt", age_seconds=60)
    _make_file(clips_dir / "RecS.jpg", age_seconds=60)
    assert worker._scan_clips_dir() == 1


def test_scan_handles_multiple_files_in_one_tick(db, clips_dir, monkeypatch):
    monkeypatch.setattr(worker, "SessionLocal", db)
    for i in range(4):
        _make_file(clips_dir / f"RecS_{i}.jpg", age_seconds=60)
    n = worker._scan_clips_dir()
    assert n == 4

    session = db()
    try:
        assert session.query(Visit).count() == 4
    finally:
        session.close()


def test_scan_picks_up_mp4_too(db, clips_dir, monkeypatch):
    """Video uploads from the webhook path also work even though scan is
    primarily for SFTP — they don't have a Visit row at scan time only if
    something else races; defensive coverage."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "motion.mp4", age_seconds=60)
    assert worker._scan_clips_dir() == 1
