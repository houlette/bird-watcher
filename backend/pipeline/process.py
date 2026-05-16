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

from db.models import Detection, Species, Visit
from pipeline.classify import SpeciesPrediction, classify_bird
from pipeline.detect import detect_birds
from pipeline.frames import extract_frames
from pipeline.fuse import fuse
from pipeline.notify import dispatch_for_detection
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

    persisted = 0
    for track in tracks:
        if not track.detections:
            continue
        best = track.best_detection

        # Multi-frame voting: classify up to 3 crops from this track (best
        # by area×conf, plus 2 evenly-spaced others if the track has enough
        # frames). Average their softmax distributions before fusion.
        crops_to_classify = _select_voting_crops(track, frames_by_index)
        per_crop_predictions = [classify_bird(c) for c in crops_to_classify]

        # Drop empty per-crop predictions (the classifier's not_a_bird gate
        # returns an empty list for crops it doesn't think are birds). If
        # every crop in this track was rejected, the track was a false
        # positive from YOLO and we skip it entirely — no Detection, no crop.
        per_crop_predictions = [p for p in per_crop_predictions if p]
        if not per_crop_predictions:
            log.info("track %d: all crops rejected as not_a_bird, skipping", track.track_id)
            continue

        averaged = _average_predictions(per_crop_predictions)

        # Phase 4: fuse averaged visual predictions with audio + seasonal priors.
        fused = fuse(
            [(p.species, p.probability) for p in averaged],
            db=db,
            when=visit.started_at,
        )
        if not fused:
            continue

        # Build a lookup of base species -> representative raw plumage label
        # from the best (top-1) per-crop prediction set for that species.
        raw_labels = {p.species: p.raw_label for preds in per_crop_predictions for p in preds}

        top = fused[0]
        crop_rel_path, _ = _save_best_crop(track, frames_by_index, visit_id=visit.id)
        species_id = _resolve_species(db, top.species)

        detection = Detection(
            visit_id=visit.id,
            species_id=species_id,
            confidence=top.probability,
            raw_predictions=[
                {
                    "species": f.species,
                    "raw": raw_labels.get(f.species, ""),
                    "p": f.probability,
                    "audio": f.audio_confirmed,
                }
                for f in fused
            ],
            audio_confirmed=bool(top.audio_confirmed),
            crop_path=str(crop_rel_path),
            bbox=list(best.bbox),
            track_id=track.track_id,
        )
        db.add(detection)
        db.flush()  # populate detection.id + created_at before push dispatch
        persisted += 1

        # Phase 5: notify subscribers if this species hasn't been seen recently.
        # Failures here are non-fatal — push is a nice-to-have, not a blocker.
        try:
            dispatch_for_detection(db, detection)
        except Exception:  # noqa: BLE001
            log.exception("Push dispatch failed for detection %d", detection.id)
    log.info("visit %d: %d tracks persisted (after not_a_bird filter)", visit.id, persisted)

    visit.processed_at = datetime.utcnow()
    visit.ended_at = datetime.utcnow()
    visit.processing_error = None
    db.commit()
    return len(tracks)


def _save_best_crop(
    track: Track,
    frames_by_index: dict,
    *,
    visit_id: int,
) -> tuple[Path, "cv2.Mat"]:
    """Write the best-crop image for a track to disk and return (relative_path, crop_array).

    Returning the crop array as well as the path lets the caller hand the same
    image to the classifier without re-decoding it from JPEG.
    """
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
    return out_path.relative_to(DATA_DIR), crop


def _select_voting_crops(track: Track, frames_by_index: dict) -> list["cv2.Mat"]:
    """Pick up to 3 crops from a track for multi-frame voting.

    Always include the 'best' (area × confidence). For longer tracks add 2 more
    evenly spaced so the classifier sees different poses / angles. Crops are
    cut directly from the cached frame images so this is essentially free.
    """
    n = len(track.detections)
    if n == 0:
        return []
    indices = {track.detections.index(track.best_detection)}
    if n >= 3:
        indices.add(n // 3)
        indices.add((2 * n) // 3)
    elif n == 2:
        indices.add(0)
        indices.add(1)

    out: list["cv2.Mat"] = []
    for i in sorted(indices):
        det = track.detections[i]
        frame_image = frames_by_index.get(det.frame_index)
        if frame_image is None:
            continue
        x, y, w, h = det.bbox
        fh, fw = frame_image.shape[:2]
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(fw, x + w + pad_w)
        y1 = min(fh, y + h + pad_h)
        crop = frame_image[y0:y1, x0:x1]
        if crop.size > 0:
            out.append(crop)
    return out


def _average_predictions(per_crop: list[list[SpeciesPrediction]]) -> list[SpeciesPrediction]:
    """Average top-K predictions across multiple crops.

    Each crop emits its own top-5; we union all species across crops and
    average their probabilities (treating missing-from-top-5 as 0). This is
    a soft majority vote weighted by per-crop confidence.
    """
    if not per_crop:
        return []
    if len(per_crop) == 1:
        return per_crop[0]

    sums: dict[str, float] = {}
    for preds in per_crop:
        for p in preds:
            sums[p.species] = sums.get(p.species, 0.0) + p.probability
    n = len(per_crop)
    averaged = [SpeciesPrediction(species=sp, probability=total / n) for sp, total in sums.items()]
    averaged.sort(key=lambda p: p.probability, reverse=True)
    return averaged[:5]


def _resolve_species(db: Session, common_name: str) -> int:
    """Get-or-create a Species row by its common name and return its id.

    Classifier labels arrive with whatever casing the model uses; we normalize
    by trimming whitespace and storing the label as-is. The scientific_name
    field is left empty until Phase 6 (active learning) or a manual import.
    """
    name = common_name.strip()
    species = db.query(Species).filter(Species.common_name == name).one_or_none()
    if species:
        return species.id
    species = Species(common_name=name, scientific_name="", is_rare=False)
    db.add(species)
    db.flush()  # populate species.id before we attach it to a Detection
    return species.id
