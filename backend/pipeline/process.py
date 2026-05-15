"""End-to-end per-clip pipeline: extract → detect → track → persist.

Phase 2 stops at writing per-track crops + Detection rows with species=None.
Phase 3 will plug the species classifier in between `tracker.finalize()`
and `_persist_detections()`.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from db.models import Detection, Visit
from pipeline.detect import detect_birds
from pipeline.frames import extract_frames
from pipeline.track import Track, Tracker

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CROPS_DIR = DATA_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)


def process_visit(visit: Visit, db: Session) -> int:
    """Run detection + tracking on a visit's clip; return number of tracks found."""
    if not visit.clip_path:
        raise ValueError(f"Visit {visit.id} has no clip_path")

    clip_path = DATA_DIR / visit.clip_path
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip missing for visit {visit.id}: {clip_path}")

    tracker = Tracker()
    last_frame_image = None  # keep handy for cropping after the loop

    # Cache frame images by index so we can produce crops once tracks are known.
    # Memory budget: ~30 frames × 4K BGR ≈ 700 MB worst case — high but bounded
    # by clip length (5–10s × 3 fps = 15–30 frames). Acceptable on an 8 GB VM.
    frames_by_index: dict[int, "cv2.Mat"] = {}

    for frame in extract_frames(clip_path, target_fps=3.0):
        frames_by_index[frame.index] = frame.image
        last_frame_image = frame.image
        dets = detect_birds(frame.image, frame.index)
        tracker.update(frame.index, dets)

    if last_frame_image is None:
        # Empty/corrupted clip — mark processed so we don't retry forever.
        visit.processed_at = datetime.utcnow()
        visit.processing_error = "no frames decoded"
        db.commit()
        return 0

    tracks = tracker.finalize()
    log.info("visit %d: %d tracks", visit.id, len(tracks))

    for track in tracks:
        if not track.detections:
            continue
        crop_rel_path = _save_best_crop(track, frames_by_index, visit_id=visit.id)
        best = track.best_detection
        db.add(
            Detection(
                visit_id=visit.id,
                species_id=None,  # filled by Phase 3
                confidence=best.confidence,
                raw_predictions=[],
                audio_confirmed=False,
                crop_path=str(crop_rel_path),
                bbox=list(best.bbox),
                track_id=track.track_id,
            )
        )

    visit.processed_at = datetime.utcnow()
    visit.ended_at = datetime.utcnow()
    visit.processing_error = None
    db.commit()
    return len(tracks)


def _save_best_crop(track: Track, frames_by_index: dict, *, visit_id: int) -> Path:
    """Write the best-crop image for a track to disk; return its path relative to DATA_DIR."""
    best = track.best_detection
    frame_image = frames_by_index[best.frame_index]
    x, y, w, h = best.bbox

    # Expand the crop slightly so the bird isn't tight against the edges —
    # 15% padding on each side, clipped to frame bounds.
    fh, fw = frame_image.shape[:2]
    pad_w, pad_h = int(w * 0.15), int(h * 0.15)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(fw, x + w + pad_w)
    y1 = min(fh, y + h + pad_h)
    crop = frame_image[y0:y1, x0:x1]

    filename = f"v{visit_id:08d}_t{track.track_id:04d}.jpg"
    out_path = CROPS_DIR / filename
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path.relative_to(DATA_DIR)
