"""APScheduler-driven worker that processes pending visits.

We claim work via the DB rather than an in-memory queue so that:
  - A restart of the API process doesn't lose pending clips.
  - Multiple worker instances could be added later (with row-level locking)
    without changing the ingest path.

Phase 2 runs the worker inside the same process as the FastAPI app. If
detection throughput becomes a bottleneck, split this out as a separate
container in docker-compose using the same image with a different CMD.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from db.models import Visit
from db.session import SessionLocal
from db.utils import utcnow
from ingest.haikubox import POLL_INTERVAL_SECONDS as HAIKUBOX_POLL_SECONDS
from ingest.haikubox import poll_once as poll_haikubox
from pipeline.frames import IMAGE_EXTS, VIDEO_EXTS
from pipeline.process import process_visit

log = logging.getLogger(__name__)

# How often to poll for new work. Reolink motion events fire at most every few
# seconds; 5 s keeps latency low without burning CPU.
POLL_INTERVAL_SECONDS = 5

# Cap concurrent visits processed per tick — one is plenty on a CPU-only VM
# because YOLO inference is already serial. This prevents a backlog tick from
# blocking the API process for too long.
MAX_VISITS_PER_TICK = 1

# Skip files that were modified in the last N seconds — they might still be
# uploading via SFTP. A second-long pause is plenty for a 1–2 MB snapshot.
MIN_FILE_AGE_SECONDS = 5

CLIPS_DIR = Path(__file__).resolve().parent.parent / "data" / "clips"
INGESTIBLE_EXTS = IMAGE_EXTS | VIDEO_EXTS


def _scan_clips_dir() -> int:
    """Create Visit rows for any clip files that landed via SFTP but don't
    have a corresponding row yet. Skips files that are still being written.

    The webhook ingest path creates its own Visit rows synchronously, so this
    scan only finds files that arrived via the SFTP container.
    """
    if not CLIPS_DIR.exists():
        return 0

    now = time.time()
    candidates: list[Path] = []
    for f in CLIPS_DIR.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in INGESTIBLE_EXTS:
            continue
        if now - f.stat().st_mtime < MIN_FILE_AGE_SECONDS:
            continue  # might still be uploading
        candidates.append(f)
    if not candidates:
        return 0

    new_count = 0
    db = SessionLocal()
    try:
        # One query to find which of these are already tracked.
        rel_paths = [f"clips/{f.name}" for f in candidates]
        existing = {
            row[0]
            for row in db.execute(select(Visit.clip_path).where(Visit.clip_path.in_(rel_paths))).all()
        }
        for f in candidates:
            rel = f"clips/{f.name}"
            if rel in existing:
                continue
            visit = Visit(started_at=utcnow(), clip_path=rel)
            db.add(visit)
            new_count += 1
        if new_count:
            db.commit()
            log.info("Filesystem scan: queued %d new clip(s) from data/clips/", new_count)
    finally:
        db.close()
    return new_count


def _process_pending() -> None:
    # Step 1: discover any clips that arrived via SFTP and have no Visit row.
    _scan_clips_dir()

    # Step 2: process the oldest pending visit.
    db = SessionLocal()
    try:
        pending = db.execute(
            select(Visit)
            .where(Visit.processed_at.is_(None))
            .order_by(Visit.started_at)
            .limit(MAX_VISITS_PER_TICK)
        ).scalars().all()

        for visit in pending:
            try:
                count = process_visit(visit, db)
                log.info("Processed visit %d (%d tracks)", visit.id, count)
            except Exception as exc:  # noqa: BLE001
                log.exception("Visit %d failed", visit.id)
                visit.processing_error = str(exc)[:500]
                # Leave processed_at NULL so we can retry — but bound retries
                # by giving up after the same error appears a few times.
                db.commit()
    finally:
        db.close()


def start_worker() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        _process_pending,
        "interval",
        seconds=POLL_INTERVAL_SECONDS,
        max_instances=1,
        coalesce=True,
        id="process_pending_visits",
    )
    scheduler.add_job(
        poll_haikubox,
        "interval",
        seconds=HAIKUBOX_POLL_SECONDS,
        max_instances=1,
        coalesce=True,
        id="poll_haikubox",
    )
    scheduler.start()
    log.info(
        "Workers started: pipeline every %ds, Haikubox poller every %ds",
        POLL_INTERVAL_SECONDS,
        HAIKUBOX_POLL_SECONDS,
    )
    return scheduler
