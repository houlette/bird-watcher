"""End-to-end smoke test of the BirdWatcher pipeline.

Feeds a video clip through the same code path that production uses for a
Reolink motion event: creates a Visit row, runs the worker logic against it,
then prints the resulting Detection rows. No HTTP server, no APScheduler —
this is purely an exercise of the pipeline.

Usage from inside the container:

    docker compose run --rm api python scripts/smoke_test.py [clip_path]

Defaults to backend/data/clips/smoke_test.webm.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke")


def main() -> int:
    from db.models import Detection, Visit
    from db.session import SessionLocal, init_db
    from db.utils import utcnow
    from pipeline.process import DATA_DIR, process_visit

    clip_arg = sys.argv[1] if len(sys.argv) > 1 else "clips/smoke_test.webm"
    clip_path = Path(clip_arg)
    if not clip_path.is_absolute():
        # Path is relative to the backend/data directory (same as what /ingest stores).
        absolute = DATA_DIR / clip_path
        if not absolute.exists():
            log.error("Clip not found: %s", absolute)
            return 1
        clip_rel = str(clip_path)
    else:
        # If an absolute path was given, copy or symlink-resolve later. For the
        # smoke test we just expect a relative path under backend/data.
        log.error("Pass a path relative to backend/data, e.g. clips/smoke_test.webm")
        return 1

    init_db()
    db = SessionLocal()
    try:
        log.info("Creating Visit pointing at %s", clip_rel)
        visit = Visit(started_at=utcnow(), clip_path=clip_rel)
        db.add(visit)
        db.commit()
        db.refresh(visit)

        log.info("Running pipeline (this loads YOLO + classifier on first call)…")
        t0 = utcnow()
        n_tracks = process_visit(visit, db)
        elapsed = (utcnow() - t0).total_seconds()
        log.info("Pipeline done: %d tracks in %.1fs", n_tracks, elapsed)

        # Refresh and print the resulting Detection rows.
        detections = (
            db.query(Detection)
            .filter(Detection.visit_id == visit.id)
            .order_by(Detection.id)
            .all()
        )
        if not detections:
            log.warning("No detections produced. Possible causes: no birds detected by YOLO, "
                        "very low confidence, or video too short.")
            return 0

        log.info("=" * 70)
        log.info("DETECTIONS")
        log.info("=" * 70)
        for d in detections:
            species_label = d.species.common_name if d.species else "Unidentified"
            log.info(
                "Track %d  →  %-30s  fused_conf=%.3f  audio_confirmed=%s",
                d.track_id, species_label, d.confidence, bool(d.audio_confirmed),
            )
            if d.raw_predictions:
                log.info("           top-5 (post-fusion):")
                for p in d.raw_predictions[:5]:
                    audio_flag = " ♪" if p.get("audio") else ""
                    log.info("             %-30s  p=%.3f%s", p["species"], p["p"], audio_flag)
            log.info("           crop saved at: data/%s", d.crop_path)
            log.info("-" * 70)
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
