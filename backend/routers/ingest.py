"""Receive motion-event clips from the Reolink camera.

Reolink HTTP push: configured in the camera's web UI under
Settings → Surveillance → HTTP push, pointing at:
  https://birdwatcher.ryanhoulette.com/api/ingest/motion

The endpoint accepts:
  - GET or POST with no body: a connectivity ping (the camera's Test button),
    returns 200 OK with {"ok": true, "kind": "ping"}.
  - POST with one or more multipart file parts: a real motion event. We
    iterate over the form to find the first file regardless of its field
    name, so Reolink firmware that uses "file"/"image"/"video"/etc. all work.

Returning 200 to the bare ping is required because Reolink shows a generic
'error' if the Test button gets back anything else (originally we returned
422 because UploadFile was required — that broke the Test button without
breaking real motion events). 200-on-empty is harmless: nothing gets
persisted unless a file actually arrived.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile as StarletteUploadFile

from db.models import Visit
from db.session import get_db
from db.utils import utcnow

log = logging.getLogger(__name__)

router = APIRouter()

CLIPS_DIR = Path(__file__).parent.parent / "data" / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


async def _first_uploaded_file(request: Request) -> UploadFile | None:
    """Pull the first uploaded file out of the multipart form regardless of
    which field name Reolink used. Returns None if the request has no body
    or no file parts — that's the Test-button ping case."""
    content_type = (request.headers.get("content-type") or "").lower()
    if not content_type.startswith("multipart/"):
        return None
    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to parse multipart form: %s", exc)
        return None
    for value in form.values():
        if isinstance(value, StarletteUploadFile):
            return value
    return None


@router.post("/motion")
@router.get("/motion")
async def receive_motion(
    request: Request,
    db: Session = Depends(get_db),
) -> dict:
    upload = await _first_uploaded_file(request) if request.method == "POST" else None

    if upload is None:
        # Connectivity test — Reolink's Test button fires an empty request
        # to verify the URL responds. Also: some Reolink firmwares send
        # motion notifications without the clip body (intended to be paired
        # with FTP upload elsewhere). We log content-type, length, and a
        # body preview to make diagnosis easy.
        ct = request.headers.get("content-type", "")
        body_preview = ""
        if request.method == "POST" and not ct.startswith("multipart/"):
            try:
                body = await request.body()
                body_preview = body[:1024].decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                body_preview = f"<read failed: {exc}>"
        log.info(
            "ingest ping: %s from %s ct=%r len=%s body=%r",
            request.method,
            request.client.host if request.client else "?",
            request.headers.get("content-type"),
            request.headers.get("content-length"),
            body_preview,
        )
        return {"ok": True, "kind": "ping"}

    now = utcnow()
    safe_orig = (upload.filename or "clip.mp4").replace("/", "_").replace("\\", "_")
    filename = f"{now:%Y%m%d_%H%M%S_%f}_{safe_orig}"
    dest = CLIPS_DIR / filename

    with dest.open("wb") as out:
        while chunk := await upload.read(1 << 20):
            out.write(chunk)

    visit = Visit(started_at=now, clip_path=str(dest.relative_to(CLIPS_DIR.parent)))
    db.add(visit)
    db.commit()
    db.refresh(visit)

    log.info("ingest clip: visit %d, %d bytes -> %s", visit.id, dest.stat().st_size, filename)
    return {"ok": True, "kind": "clip", "visit_id": visit.id, "filename": filename, "bytes": dest.stat().st_size}
