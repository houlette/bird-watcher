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
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from db.models import Visit
from db.session import SessionLocal
from db.utils import utcnow
from ingest.haikubox import POLL_INTERVAL_SECONDS as HAIKUBOX_POLL_SECONDS
from ingest.haikubox import poll_once as poll_haikubox
from pipeline.daylight import is_daylight
from pipeline.exceptions import SkipFile
from pipeline.frames import IMAGE_EXTS, VIDEO_EXTS
from pipeline.process import FRAMES_DIR, process_visit

log = logging.getLogger(__name__)

# How often to poll for new work. Reolink motion events fire at most every few
# seconds; 5 s keeps latency low without burning CPU.
POLL_INTERVAL_SECONDS = 5

# Cap concurrent visits processed per tick — one is plenty on a CPU-only VM
# because YOLO inference is already serial. This prevents a backlog tick from
# blocking the API process for too long.
MAX_VISITS_PER_TICK = 1

# Skip files that were modified in the last N seconds — they might still be
# uploading via FTPS. A 4K MP4 motion clip is 20–100 MB; over residential
# upload bandwidth (10–50 Mbps) that takes 16–80 s, so 90 s gives a healthy
# margin against picking up a partial upload. Snapshots (~1–2 MB) clear
# this comfortably.
MIN_FILE_AGE_SECONDS = 90

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
INGESTIBLE_EXTS = IMAGE_EXTS | VIDEO_EXTS

# Reolink encodes the capture time in the upload filename:
#   Birdfeeder_00_20260520123029.jpg → 2026-05-20 12:30:29 UTC
# Webhook ground truth confirms the timestamp is UTC (alarmTime: ...+0000).
# We parse this rather than using file mtime / worker time so the audio
# correlation window in fuse.py aligns with when the photo was actually
# taken — which can be hours earlier than when we process it if we're
# working through a backlog.
_CAPTURE_TIME_RE = re.compile(r"_(\d{14})\.[A-Za-z0-9]+$")


def _capture_time_from_filename(name: str) -> datetime | None:
    """Parse Reolink's `..._YYYYMMDDHHMMSS.ext` filename into a naive UTC datetime.

    Reject implausible timestamps (year < 2000) — the Reolink loses NTP sync
    occasionally and starts writing `19700220...` filenames. Returning None
    here makes the caller fall back to the file's mtime, which the FTP
    server stamps when the upload completes (correctly reflecting wall time
    even when the camera's own clock is wrong)."""
    m = _CAPTURE_TIME_RE.search(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    if dt.year < 2000:
        return None
    return dt


def _scan_clips_dir() -> int:
    """Create Visit rows for any clip files that landed via FTPS but don't
    have a corresponding row yet. Skips files that are still being written.

    Walks the clips/ tree recursively because Reolink uploads under
    date-stamped subdirectories: clips/upload/YYYY/MM/DD/Birdfeeder_...mp4.
    Visit.clip_path is stored relative to backend/data/ so the downstream
    pipeline can resolve it as DATA_DIR / clip_path.
    """
    if not CLIPS_DIR.exists():
        return 0

    now = time.time()
    candidates: list[Path] = []
    for f in CLIPS_DIR.rglob("*"):
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
        rel_paths = [str(f.relative_to(DATA_DIR)) for f in candidates]
        existing = {
            row[0]
            for row in db.execute(select(Visit.clip_path).where(Visit.clip_path.in_(rel_paths))).all()
        }
        for f in candidates:
            rel = str(f.relative_to(DATA_DIR))
            if rel in existing:
                continue
            # Prefer the camera-encoded capture time from the filename;
            # fall back to file mtime (the moment the FTP upload completed,
            # close enough on a healthy live system); last resort is now.
            started_at = _capture_time_from_filename(f.name)
            if started_at is None:
                started_at = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
            visit = Visit(started_at=started_at, clip_path=rel)
            db.add(visit)
            new_count += 1
        if new_count:
            db.commit()
            log.info("Filesystem scan: queued %d new clip(s) under data/clips/", new_count)
    finally:
        db.close()
    return new_count


def _process_pending() -> None:
    # Step 1: discover any clips that arrived via SFTP and have no Visit row.
    _scan_clips_dir()

    # Step 2: process the NEWEST pending visit first. Counter-intuitive for a
    # work queue, but during a backlog drain we want the user's feed to show
    # what the camera saw 30 seconds ago — not what it saw 5 days ago. Older
    # backlog still drains in the background between fresh arrivals; it just
    # waits its turn behind anything more recent.
    db = SessionLocal()
    try:
        pending = db.execute(
            select(Visit)
            .where(Visit.processed_at.is_(None))
            .order_by(Visit.started_at.desc())
            .limit(MAX_VISITS_PER_TICK)
        ).scalars().all()

        for visit in pending:
            # Skip visits captured outside daylight hours. At night the
            # Reolink switches to IR/grayscale (the species classifier
            # can't handle that) and birds aren't active anyway — so we
            # save the CPU and don't pollute the feed with shadow shapes.
            if visit.started_at and not is_daylight(visit.started_at):
                log.info("Visit %d captured outside daylight; skipping", visit.id)
                visit.processed_at = utcnow()
                visit.processing_error = "skipped: captured outside daylight hours"
                db.commit()
                continue
            try:
                count = process_visit(visit, db)
                log.info("Processed visit %d (%d tracks)", visit.id, count)
            except SkipFile as skip:
                # Permanent skip — file is too big / corrupt / unsupported.
                # Mark processed so we never re-queue it. The error message
                # is preserved for inspection via the DB or the feed UI.
                log.warning("Visit %d skipped: %s", visit.id, skip)
                visit.processed_at = utcnow()
                visit.processing_error = f"skipped: {skip}"[:500]
                db.commit()
            except Exception as exc:  # noqa: BLE001
                # Transient error — leave processed_at NULL so the next tick
                # retries. Record the latest error for visibility.
                log.exception("Visit %d failed", visit.id)
                visit.processing_error = str(exc)[:500]
                db.commit()
    finally:
        db.close()


# Source frames are saved per persisted track for a future YOLO fine-tune.
# We retain them long enough for the user's labeling to catch up to fresh
# detections (median lag is hours, occasional outliers run days). 14 days is
# generous without unbounded disk growth — ~1.5–2 GB/day in steady state.
FRAME_RETENTION_DAYS = 14
FRAME_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60

# Reolink MP4 / JPG uploads sit under CLIPS_DIR. Once a clip is processed
# (the Visit row's processed_at is set), the file is dead weight — every
# frame we'd want for future YOLO retraining was already extracted into
# FRAMES_DIR during process_visit. Holding longer just risks disk-full
# (we hit it twice already, 75 GB VM and ~10-40 GB of clips/day depending
# on bird activity). 1-day retention bounds growth without losing
# never-processed clips.
CLIP_RETENTION_HOURS = 24
CLIP_CLEANUP_INTERVAL_SECONDS = 60 * 60   # hourly, since 24h windows tighter than 14d


def _cleanup_old_frames() -> int:
    """Delete source-frame archive files older than FRAME_RETENTION_DAYS.

    Keeps things simple: time-based retention by file mtime, no DB join.
    If we later want to preserve labeled-detection frames indefinitely,
    extend this to skip files whose (visit_id, track_id) appears in
    Correction. For now, the assumption is "label within 14 days or lose it."
    """
    if not FRAMES_DIR.exists():
        return 0
    cutoff = time.time() - FRAME_RETENTION_DAYS * 86400
    deleted = 0
    for path in FRAMES_DIR.glob("v*_t*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except OSError:
            # File raced with another delete or got mode-changed; ignore.
            continue
    if deleted:
        log.info("Frame retention: deleted %d frame(s) older than %d days", deleted, FRAME_RETENTION_DAYS)
    return deleted


def _cleanup_old_clips() -> int:
    """Delete MP4 / JPG clips older than CLIP_RETENTION_HOURS.

    Runs hourly because the daily 7-15 GB ingestion rate makes a 24h window
    much tighter than the 14d frame retention window. Walks the recursive
    Reolink upload tree (clips/upload/YYYY/MM/DD/...)."""
    if not CLIPS_DIR.exists():
        return 0
    cutoff = time.time() - CLIP_RETENTION_HOURS * 3600
    deleted = 0
    bytes_freed = 0
    for path in CLIPS_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".mp4", ".jpg", ".jpeg"):
            continue
        try:
            st = path.stat()
            if st.st_mtime < cutoff:
                bytes_freed += st.st_size
                path.unlink()
                deleted += 1
        except OSError:
            continue
    if deleted:
        log.info("Clip retention: deleted %d clip(s) older than %dh (%.1f GB freed)",
                 deleted, CLIP_RETENTION_HOURS, bytes_freed / 1e9)
    return deleted


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
    scheduler.add_job(
        _cleanup_old_frames,
        "interval",
        seconds=FRAME_CLEANUP_INTERVAL_SECONDS,
        max_instances=1,
        coalesce=True,
        id="cleanup_old_frames",
        next_run_time=datetime.now(timezone.utc),  # run once at startup
    )
    scheduler.add_job(
        _cleanup_old_clips,
        "interval",
        seconds=CLIP_CLEANUP_INTERVAL_SECONDS,
        max_instances=1,
        coalesce=True,
        id="cleanup_old_clips",
        next_run_time=datetime.now(timezone.utc),  # run once at startup
    )
    scheduler.start()
    log.info(
        "Workers started: pipeline every %ds, Haikubox poller every %ds, frame cleanup daily, clip cleanup hourly",
        POLL_INTERVAL_SECONDS,
        HAIKUBOX_POLL_SECONDS,
    )
    return scheduler
