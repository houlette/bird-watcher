"""End-to-end per-clip pipeline: extract → detect → track → classify → fuse → persist.

For every YOLO-detected track we always save the best crop and write a
Detection row, even if the species classifier couldn't place the bird in
our NA allow-list. Unidentified rows go into the feed as 'Unidentified'
so the user can tag them via the SpeciesPicker — those manual labels
are the highest-value examples for the eventual classifier fine-tune.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from db.models import Detection, Species, Visit
from db.utils import utcnow
from pipeline.classify import SpeciesPrediction, classify_bird
from pipeline.detect import detect_birds
from pipeline.exceptions import SkipFile
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
        # The file's gone — either it was deleted (manual cleanup, retention
        # eviction) or it never landed (interrupted upload, bug elsewhere).
        # Either way, it's not coming back; treat as a permanent skip so the
        # worker doesn't loop on this row forever.
        raise SkipFile(f"clip file missing on disk: {clip_path}")

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
        visit.processed_at = utcnow()
        visit.processing_error = "no frames decoded"
        db.commit()
        return 0

    tracks = tracker.finalize()
    log.info("visit %d: %d tracks", visit.id, len(tracks))

    persisted = 0
    for track in tracks:
        if not track.detections:
            continue

        # Rank this track's detections by area × confidence × sharpness so
        # the saved crop AND the crops we hand to the classifier are the
        # best-looking ones available across all frames. Motion blur and
        # mid-flight poses score low on the Laplacian-variance sharpness
        # term, letting in-focus perched frames win.
        ranked = _rank_detections(track, frames_by_index)
        if not ranked:
            continue
        best = ranked[0]

        # Always save the YOLO-detected crop. If the classifier later rejects
        # it, the row still goes into the feed as "Unidentified" so the user
        # can tag it via the picker (real species OR "Not a bird"). These
        # are the highest-value active-learning examples — YOLO sees a
        # bird shape but the classifier isn't confident, so an explicit
        # human label closes the loop.
        crop_rel_path, _ = _save_crop(best, frames_by_index, visit_id=visit.id, track_id=track.track_id)

        # Multi-frame voting: classify up to 3 crops from this track. Pick
        # the top 3 by sharpness-aware score so the classifier sees the
        # cleanest views, not arbitrary samples.
        crops_to_classify = [
            _extract_crop(d, frames_by_index) for d in ranked[:3]
        ]
        crops_to_classify = [c for c in crops_to_classify if c is not None and c.size > 0]
        per_crop_predictions = [classify_bird(c) for c in crops_to_classify]
        per_crop_predictions = [p for p in per_crop_predictions if p]

        if not per_crop_predictions:
            # Classifier rejected every crop. Persist with species_id=NULL;
            # the user can correct via the "Wrong species?" picker.
            log.info("track %d: classifier rejected; persisting as Unidentified", track.track_id)
            db.add(Detection(
                visit_id=visit.id,
                species_id=None,
                confidence=0.0,
                raw_predictions=[],
                audio_confirmed=False,
                crop_path=str(crop_rel_path),
                bbox=list(best.bbox),
                track_id=track.track_id,
            ))
            persisted += 1
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
    log.info("visit %d: %d tracks persisted (some may be Unidentified)", visit.id, persisted)

    visit.processed_at = utcnow()
    visit.ended_at = utcnow()
    visit.processing_error = None
    db.commit()
    return len(tracks)


def _laplacian_variance(image: "cv2.Mat") -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper.

    Standard "focus measure" from microscopy / astrophotography. A motion-
    blurred crop has low high-frequency content and scores low; an in-focus
    crop scores high. Typical values on our crops:
        very blurry  ~10
        in focus     ~100-500
        very sharp   1000+
    Content-dependent (a uniformly-colored bird scores lower than a heavily
    feathered one even at the same focus), so we only use it as a ranking
    signal, not an absolute threshold.
    """
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _extract_crop(det, frames_by_index: dict, padding: float = 0.15) -> "cv2.Mat | None":
    """Crop the padded bbox out of the relevant frame. Returns None if the
    frame isn't cached (shouldn't happen in normal flow)."""
    frame = frames_by_index.get(det.frame_index)
    if frame is None:
        return None
    x, y, w, h = det.bbox
    fh, fw = frame.shape[:2]
    pad_w, pad_h = int(w * padding), int(h * padding)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(fw, x + w + pad_w)
    y1 = min(fh, y + h + pad_h)
    return frame[y0:y1, x0:x1]


def _rank_detections(track: Track, frames_by_index: dict) -> list:
    """Sort the track's detections by area × confidence × sharpness, desc.

    The +1 on sharpness avoids zeroing-out a small but confident detection
    just because its crop happened to land at low Laplacian variance, and
    lets the ranking degrade gracefully if a frame is missing from cache
    (sharpness defaults to 0, so area × confidence still orders them).
    """
    scored: list[tuple[float, object]] = []
    for det in track.detections:
        crop = _extract_crop(det, frames_by_index)
        if crop is None or crop.size == 0:
            continue
        _, _, w, h = det.bbox
        sharpness = _laplacian_variance(crop)
        score = w * h * det.confidence * (sharpness + 1.0)
        scored.append((score, det))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


def _save_crop(
    det,
    frames_by_index: dict,
    *,
    visit_id: int,
    track_id: int,
) -> tuple[Path, "cv2.Mat"]:
    """Write the crop for `det` to disk and return (relative_path, crop_array).
    Returning the array lets the caller pass it straight to the classifier
    without re-decoding the JPEG."""
    crop = _extract_crop(det, frames_by_index)
    assert crop is not None and crop.size > 0
    filename = f"v{visit_id:08d}_t{track_id:04d}.jpg"
    out_path = CROPS_DIR / filename
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path.relative_to(DATA_DIR), crop


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
