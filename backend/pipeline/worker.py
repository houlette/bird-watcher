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

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from db.models import Visit
from db.session import SessionLocal
from pipeline.process import process_visit

log = logging.getLogger(__name__)

# How often to poll for new work. Reolink motion events fire at most every few
# seconds; 5 s keeps latency low without burning CPU.
POLL_INTERVAL_SECONDS = 5

# Cap concurrent visits processed per tick — one is plenty on a CPU-only VM
# because YOLO inference is already serial. This prevents a backlog tick from
# blocking the API process for too long.
MAX_VISITS_PER_TICK = 1


def _process_pending() -> None:
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
    scheduler.start()
    log.info("Pipeline worker started (poll every %ds)", POLL_INTERVAL_SECONDS)
    return scheduler
