"""Receive motion-event clips from the Reolink camera.

Phase 1: accept the upload and stash it on disk. Later phases will hand it to
the classification pipeline.

Reolink HTTP push: configured in the camera's web UI under
Settings → Surveillance → HTTP push, pointing at:
  https://birdwatcher.ryanhoulette.com/api/ingest/motion
"""
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from db.models import Visit
from db.session import get_db
from db.utils import utcnow

router = APIRouter()

CLIPS_DIR = Path(__file__).parent.parent / "data" / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/motion")
async def receive_motion(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    """Receive a motion-event clip from the camera and queue it for processing."""
    now = utcnow()
    filename = f"{now:%Y%m%d_%H%M%S_%f}_{file.filename}"
    dest = CLIPS_DIR / filename

    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)

    visit = Visit(started_at=now, clip_path=str(dest.relative_to(CLIPS_DIR.parent)))
    db.add(visit)
    db.commit()
    db.refresh(visit)

    # TODO Phase 2: enqueue visit.id for the classification worker
    return {"visit_id": visit.id, "filename": filename, "bytes": dest.stat().st_size}
