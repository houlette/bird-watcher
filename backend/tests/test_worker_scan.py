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
    # Patch both CLIPS_DIR (where to look) and DATA_DIR (the prefix the
    # worker strips when computing clip_path values stored in the DB).
    monkeypatch.setattr(worker, "CLIPS_DIR", d)
    monkeypatch.setattr(worker, "DATA_DIR", tmp_path)
    return d


def _make_file(path, age_seconds: float = 120.0) -> None:
    path.write_bytes(b"\xff\xd8\xff\xe0fake jpg payload")
    mtime = time.time() - age_seconds
    os.utime(path, (mtime, mtime))


def test_scan_creates_visit_for_new_jpg(db, clips_dir, monkeypatch):
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "RecS_20260520_001234.jpg")
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
    # 60 s < the 90 s MIN_FILE_AGE_SECONDS threshold (a 100 MB MP4 can still
    # be uploading at that point on residential bandwidth).
    _make_file(clips_dir / "partial.jpg", age_seconds=60)
    n = worker._scan_clips_dir()
    assert n == 0


def test_scan_skips_files_already_tracked(db, clips_dir, monkeypatch):
    """Running scan twice on the same file should only create one Visit."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "RecS_repeat.jpg")
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
    _make_file(clips_dir / "README.txt")
    _make_file(clips_dir / "RecS.jpg")
    assert worker._scan_clips_dir() == 1


def test_scan_handles_multiple_files_in_one_tick(db, clips_dir, monkeypatch):
    monkeypatch.setattr(worker, "SessionLocal", db)
    for i in range(4):
        _make_file(clips_dir / f"RecS_{i}.jpg")
    n = worker._scan_clips_dir()
    assert n == 4

    session = db()
    try:
        assert session.query(Visit).count() == 4
    finally:
        session.close()


def test_scan_picks_up_mp4_too(db, clips_dir, monkeypatch):
    """Video uploads from the webhook path also work even though scan is
    primarily for FTPS — they don't have a Visit row at scan time only if
    something else races; defensive coverage."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    _make_file(clips_dir / "motion.mp4")
    assert worker._scan_clips_dir() == 1


def test_capture_time_parsed_from_reolink_filename():
    """The Reolink filename embeds the UTC capture time; we must extract it
    so audio correlation in fuse.py uses photo-time, not worker-process-time."""
    from datetime import datetime
    parse = worker._capture_time_from_filename

    assert parse("Birdfeeder_00_20260520123029.jpg") == datetime(2026, 5, 20, 12, 30, 29)
    assert parse("Birdfeeder_00_20260520123029.mp4") == datetime(2026, 5, 20, 12, 30, 29)
    # No timestamp suffix — None so the caller falls back to file mtime.
    assert parse("RecS_random.jpg") is None
    assert parse("notimestamp.jpg") is None


def test_scan_uses_filename_timestamp_for_started_at(db, clips_dir, monkeypatch):
    """A visit row created from a Reolink upload should carry the camera's
    capture time, not the worker's `now`."""
    from datetime import datetime
    monkeypatch.setattr(worker, "SessionLocal", db)
    nested = clips_dir / "upload" / "2026" / "05" / "20"
    nested.mkdir(parents=True)
    _make_file(nested / "Birdfeeder_00_20260520123029.jpg")
    worker._scan_clips_dir()

    session = db()
    try:
        v = session.query(Visit).one()
        assert v.started_at == datetime(2026, 5, 20, 12, 30, 29)
    finally:
        session.close()


def test_skipfile_marks_visit_processed_without_retry(db, clips_dir, monkeypatch):
    """When process_visit raises SkipFile, the worker should set processed_at
    so the visit doesn't get re-queued forever. Failure mode this protects
    against: a 100 MB MP4 that hits MAX_VIDEO_BYTES, gets skipped, but
    without this would stay pending and retry every tick.
    """
    from pipeline.exceptions import SkipFile

    monkeypatch.setattr(worker, "SessionLocal", db)
    # The daylight gate isn't what this test exercises — pin it open so the
    # SkipFile path is what actually runs.
    monkeypatch.setattr(worker, "is_daylight", lambda _ts: True)
    # Seed a visit pointing at a real file so the worker queries it.
    _make_file(clips_dir / "huge.mp4")
    worker._scan_clips_dir()

    # Monkeypatch process_visit to raise SkipFile, mimicking the size-cap path.
    def fake_process_visit(visit, session):
        raise SkipFile("video too large to process: 99.9 MB > 15 MB cap")

    monkeypatch.setattr(worker, "process_visit", fake_process_visit)
    worker._process_pending()

    session = db()
    try:
        v = session.query(Visit).one()
        assert v.processed_at is not None  # marked done
        assert "skipped" in (v.processing_error or "")
        assert "video too large" in (v.processing_error or "")
    finally:
        session.close()

    # Run the worker again — the visit should no longer be queued because
    # processed_at is set.
    def boom(_visit, _session):
        raise AssertionError("worker should not re-process a skipped visit")

    monkeypatch.setattr(worker, "process_visit", boom)
    worker._process_pending()  # should be a no-op


def test_scan_recurses_into_subdirectories(db, clips_dir, monkeypatch):
    """Reolink uploads under upload/YYYY/MM/DD/ — the scan must descend."""
    monkeypatch.setattr(worker, "SessionLocal", db)
    nested = clips_dir / "upload" / "2026" / "05" / "20"
    nested.mkdir(parents=True)
    _make_file(nested / "Birdfeeder_00_20260520102329.jpg")
    _make_file(nested / "Birdfeeder_00_20260520102329.mp4")
    assert worker._scan_clips_dir() == 2

    session = db()
    try:
        paths = sorted(v.clip_path for v in session.query(Visit).all())
        assert paths == [
            "clips/upload/2026/05/20/Birdfeeder_00_20260520102329.jpg",
            "clips/upload/2026/05/20/Birdfeeder_00_20260520102329.mp4",
        ]
    finally:
        session.close()


def test_cleanup_old_frames_deletes_only_stale(tmp_path, monkeypatch):
    """Files older than FRAME_RETENTION_DAYS are deleted; newer ones survive."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    fresh = frames_dir / "v00000001_t0001.jpg"
    stale = frames_dir / "v00000002_t0001.jpg"
    fresh.write_bytes(b"fresh")
    stale.write_bytes(b"stale")
    # Backdate the stale one well past the cutoff.
    long_ago = time.time() - (worker.FRAME_RETENTION_DAYS + 1) * 86400
    os.utime(stale, (long_ago, long_ago))

    monkeypatch.setattr(worker, "FRAMES_DIR", frames_dir)
    deleted = worker._cleanup_old_frames()

    assert deleted == 1
    assert fresh.exists()
    assert not stale.exists()


def test_cleanup_old_frames_handles_missing_dir(tmp_path, monkeypatch):
    """No-op when the frames directory doesn't exist yet (fresh install)."""
    monkeypatch.setattr(worker, "FRAMES_DIR", tmp_path / "does_not_exist")
    assert worker._cleanup_old_frames() == 0
