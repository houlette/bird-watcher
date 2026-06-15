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
import numpy as np
from sqlalchemy.orm import Session

from db.models import NOT_A_BIRD_LABEL, Detection, Species, Visit
from db.utils import utcnow
from pipeline.binary_filter import (
    is_enabled as binary_filter_enabled,
    nab_probability,
)
from settings import settings
from pipeline.classify import SpeciesPrediction, classify_bird
from pipeline.detect import detect_birds
from pipeline.exceptions import SkipFile
from pipeline.frames import extract_frames
from pipeline.fuse import FusedPrediction, fuse
from pipeline.notify import dispatch_for_detection
from pipeline.scene_mask import filter_detections as _scene_mask_filter
from pipeline.track import Track, Tracker

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CROPS_DIR = DATA_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
# Full source frames for persisted tracks, preserved for a future YOLO
# fine-tune. Crops alone aren't enough because YOLO needs (image, bbox, label)
# triples — the bbox in image coordinates of the full frame. Disk usage is
# bounded by a daily 14-day retention task in pipeline.worker.
FRAMES_DIR = DATA_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


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

    # Idempotency guard. A visit only reaches here with processed_at NULL, but
    # that can mean "never run" OR "a prior attempt failed partway and left
    # detections behind." Clear any pre-existing detections for this visit so a
    # reprocess REPLACES rather than APPENDS — without this, every retry stacks
    # another identical crop into the feed (the 6–8× duplicate bug). Safe
    # because processed_at NULL means the user hasn't reviewed it yet, so no
    # Correction can reference these rows.
    #
    # CRITICAL: probe with a READ first and only DELETE (+ commit immediately)
    # when rows actually exist. A bulk DELETE opens a write transaction and
    # SQLite holds the write lock until commit; issuing it unconditionally here
    # — with the only commit at the end of the function — pinned the write lock
    # through the entire 1–2 min frame-extraction + YOLO phase, so every other
    # writer (user corrections, the Haikubox poller) failed with "database is
    # locked". The common never-reprocessed case takes the read path and
    # acquires no write lock until persistence.
    has_prior = (
        db.query(Detection.id).filter(Detection.visit_id == visit.id).first() is not None
    )
    if has_prior:
        cleared = (
            db.query(Detection)
            .filter(Detection.visit_id == visit.id)
            .delete(synchronize_session=False)
        )
        db.commit()  # release the write lock now; don't hold it through processing
        log.warning("visit %d: cleared %d detection(s) from a prior incomplete attempt", visit.id, cleared)

    tracker = Tracker()
    any_frame_decoded = False
    # Aggregate count of YOLO detections the scene mask suppressed
    # across all frames in this visit. Persisted on the Visit row at
    # the end so the Stats page can show this hidden funnel stage and
    # we can spot-check when real birds get filtered (e.g., a
    # woodpecker on the lilac near the hummingbird feeder).
    scene_mask_suppressed = 0

    # We sample at 3 fps (vs the source's 20 fps) — every ~7th frame.
    # Combined with the 10 s clip-duration cap in frames.py, this gives the
    # sharpness ranker ~30 candidates per visit total, and bounds per-visit
    # YOLO cost to about 2 minutes on the CAX21 — enough to keep up with
    # ~30 arrivals/hr without the queue growing. Was briefly 4 then 6 fps;
    # both saturated the CPU budget against 24 s clips.
    #
    # We DON'T cache full frames — instead each detection's crop is extracted
    # at YOLO time and stored on the BirdDetection itself (see detect.py).
    # At 4K BGR a single frame is ~25 MB; a typical bird crop is ~120 KB.
    # Caching crops scales with bird-count, not frame-count, so we can raise
    # target_fps without OOM risk.
    for frame in extract_frames(clip_path, target_fps=3.0):
        any_frame_decoded = True
        dets = detect_birds(frame.image, frame.index)
        # Scene-mask: drop YOLO detections in regions the user has
        # repeatedly labeled as Not-a-bird (hummingbird feeder, etc.).
        # Detections with strong YOLO confidence override the mask, so
        # an actual bird at the feeder still gets through.
        dets, this_frame_suppressed = _scene_mask_filter(dets)
        scene_mask_suppressed += this_frame_suppressed
        for d in dets:
            d.crop = _extract_crop_from_image(d, frame.image)
        tracker.update(frame.index, dets)
        # frame.image goes out of scope at the next iteration; the ~25 MB
        # allocation is freed before the next frame is decoded.

    if not any_frame_decoded:
        # Empty/corrupted clip — mark processed so we don't retry forever.
        visit.processed_at = utcnow()
        visit.processing_error = "no frames decoded"
        visit.scene_mask_suppressed = scene_mask_suppressed
        db.commit()
        return 0

    tracks = tracker.finalize()
    log.info("visit %d: %d tracks", visit.id, len(tracks))

    # ----------------------------------------------------------------------
    # The work is split into three phases so the SQLite *write* lock is held
    # for only the few milliseconds it takes to INSERT the rows — not the
    # seconds of model inference + video decode that dominate a visit. Holding
    # the write lock across that compute is what starved user corrections and
    # the Haikubox poller with "database is locked".
    #
    #   Phase 1 — compute (reads only): rank, save crops to disk, classify,
    #             fuse, binary-filter; accumulate plain per-track records.
    #   Phase 2 — persist (the ONLY write transaction): resolve species,
    #             INSERT all detections + the visit update, commit once.
    #   Phase 3 — side effects after commit (no lock held): save source
    #             frames (disk) and dispatch push notifications.
    # ----------------------------------------------------------------------

    # Maps track_id -> the sampled frame index of the chosen best detection;
    # consumed by the single clip re-decode in Phase 3.
    frames_to_save: dict[int, int] = {}
    # Per-track records built in Phase 1. Each is a dict of Detection kwargs
    # plus a transient "species_name" (resolved to an id in Phase 2; the
    # get-or-create is a write, so it's deferred out of the compute phase).
    pending: list[dict] = []

    # ---- Phase 1: compute everything; NO writes (reads via fuse() only). ----
    for track in tracks:
        if not track.detections:
            continue

        # Rank this track's detections by area × confidence × sharpness so
        # the saved crop AND the crops we hand to the classifier are the
        # best-looking ones available across all frames. Motion blur and
        # mid-flight poses score low on the Laplacian-variance sharpness
        # term, letting in-focus perched frames win.
        ranked = _rank_detections(track)
        if not ranked:
            continue
        best = ranked[0]

        # Snapshot every per-frame bbox in the track for downstream
        # smoothing analysis (the size-prior may benefit from a
        # track-median bbox instead of the single best frame's noisy one
        # — see scripts/depth/compare_track_smoothed.py once we have
        # enough labeled data with this column populated). All bboxes are
        # in full-frame 4K coords, same shape as Detection.bbox.
        track_bboxes_for_db = [list(d.bbox) for d in track.detections]

        # Always save the YOLO-detected crop. If the classifier later rejects
        # it, the row still goes into the feed as "Unidentified" so the user
        # can tag it via the picker (real species OR "Not a bird" / "Unknown
        # bird"). These are the highest-value active-learning examples — YOLO
        # sees a bird shape but the classifier isn't confident, so an
        # explicit human label closes the loop.
        crop_rel_path = _save_crop(best, visit_id=visit.id, track_id=track.track_id)

        # Hand the classifier the top-K crops. Two behaviors switched by the
        # _USE_MULTI_FRAME_FUSION constant:
        #
        #   - Fused (Step 2): phase-correlation-align the top-3 crops and
        #     average to one denoised composite, then classify ONCE. Saves
        #     ~2× classifier latency per track and gives the model a cleaner
        #     image to work with — net win on perched-bird tracks where
        #     alignment succeeds; reverts to single-anchor when it doesn't.
        #
        #   - Legacy (vote): classify each of the top-3 independently and
        #     soft-average the per-crop top-5 distributions. Three model
        #     calls per track.
        candidate_crops = [d.crop for d in ranked[:3] if d.crop is not None and d.crop.size > 0]
        # `fused_crop_image` is the multi-frame-aligned composite. Stored
        # separately because the variable name `fused` is reassigned below
        # to the prediction-fusion result from pipeline.fuse.fuse(), and
        # the binary-filter code further down needs the IMAGE, not the
        # prediction list. Without this rename the binary filter blew up
        # with "'list' object has no attribute 'size'" because it
        # received the list of FusedPrediction tuples instead of the
        # numpy crop.
        fused_crop_image: "cv2.Mat | None" = None
        if _USE_MULTI_FRAME_FUSION:
            fused_crop_image = _fuse_crops(candidate_crops)
            preds = classify_bird(fused_crop_image) if fused_crop_image is not None else []
            per_crop_predictions = [preds] if preds else []
        else:
            per_crop_predictions = [classify_bird(c) for c in candidate_crops]
            per_crop_predictions = [p for p in per_crop_predictions if p]

        area_px, brightness, sharpness = _crop_quality(best.crop, list(best.bbox))

        if not per_crop_predictions:
            # Classifier rejected every crop. Persist with species_id=NULL;
            # the user can correct via the "Wrong species?" picker.
            log.info("track %d: classifier rejected; persisting as Unidentified", track.track_id)
            pending.append({
                "species_name": None,
                "confidence": 0.0,
                "yolo_confidence": float(best.confidence),
                "raw_predictions": [],
                "audio_confirmed": False,
                "crop_path": str(crop_rel_path),
                "bbox": list(best.bbox),
                "track_bboxes": track_bboxes_for_db,
                "track_id": track.track_id,
                "crop_area_px": area_px,
                "brightness": brightness,
                "sharpness": sharpness,
            })
            frames_to_save[track.track_id] = best.frame_index
            continue

        averaged = _average_predictions(per_crop_predictions)

        # Fuse averaged visual predictions with audio + seasonal + size priors.
        # `best.bbox` is the saved crop's bbox in full-frame 4K coords; the size
        # prior reads max(w, h) from it. When no size_priors.json is present
        # (fresh DB / haven't calibrated yet), the prior is a no-op. fuse()
        # issues DB *reads* only — no write lock.
        fused = fuse(
            [(p.species, p.probability) for p in averaged],
            db=db,
            when=visit.started_at,
            bbox=tuple(best.bbox),
        )
        if not fused:
            continue

        # Build a lookup of base species -> representative raw plumage label
        # from the best (top-1) per-crop prediction set for that species.
        raw_labels = {p.species: p.raw_label for preds in per_crop_predictions for p in preds}

        top = fused[0]

        # Records P(NAB) iff the binary filter overrides below; stored on the
        # Detection so the kill cohort can be audited for precision.
        nab_override_p = None

        # Binary post-filter. The species classifier is fooled by leaves /
        # ivy / debris with bird-shaped silhouettes; the binary head was
        # trained specifically on this yard's false positives. When it
        # disagrees confidently, override to NAB. We score the fused crop
        # (or best candidate when fusion is off) so the binary head sees
        # the same pixels the species classifier did.
        if binary_filter_enabled() and top.species != NOT_A_BIRD_LABEL:
            filter_crop = fused_crop_image if _USE_MULTI_FRAME_FUSION else best.crop
            nab_p = nab_probability(filter_crop) if filter_crop is not None else None
            if nab_p is not None and nab_p >= settings.bird_binary_nab_threshold:
                log.info(
                    "track %d: binary filter override → NAB (was %s @ %.2f; NAB P=%.2f)",
                    track.track_id, top.species, top.probability, nab_p,
                )
                nab_override_p = nab_p
                # Replace the top entry's species with NAB but keep the
                # original raw_predictions intact for transparency —
                # users can see what was overridden via the feed card.
                # Must be a FusedPrediction (not SpeciesPrediction): `top`
                # is consumed below as a fused result — top.audio_confirmed
                # is read when building the Detection row, and
                # SpeciesPrediction has neither that field nor accepts it as
                # a kwarg (it raised TypeError here every time the binary
                # filter fired, crashing the visit).
                top = FusedPrediction(
                    species=NOT_A_BIRD_LABEL,
                    probability=nab_p,
                    audio_confirmed=False,
                    seasonal_boost=1.0,
                )

        pending.append({
            "species_name": top.species,
            "confidence": top.probability,
            "yolo_confidence": float(best.confidence),
            "raw_predictions": [
                {
                    "species": f.species,
                    "raw": raw_labels.get(f.species, ""),
                    "p": f.probability,
                    "audio": f.audio_confirmed,
                }
                for f in fused
            ],
            "audio_confirmed": bool(top.audio_confirmed),
            "crop_path": str(crop_rel_path),
            "bbox": list(best.bbox),
            "track_bboxes": track_bboxes_for_db,
            "crop_area_px": area_px,
            "brightness": brightness,
            "sharpness": sharpness,
            "track_id": track.track_id,
            "nab_override_p": nab_override_p,
        })
        frames_to_save[track.track_id] = best.frame_index

    # ---- Phase 2: persist in one short write transaction. ----
    # Drop any read snapshot fuse() left open during the (seconds-long) Phase 1
    # compute. Otherwise the INSERTs below would try to upgrade a stale
    # snapshot and could fail with SQLITE_BUSY_SNAPSHOT — which busy_timeout
    # does NOT retry — if the API committed a write meanwhile. Phase 1 wrote
    # nothing, so rollback discards only the read transaction.
    db.rollback()
    species_id_cache: dict[str, int] = {}
    detections: list[Detection] = []
    for rec in pending:
        name = rec.pop("species_name")
        if name is None:
            species_id = None
        else:
            species_id = species_id_cache.get(name)
            if species_id is None:
                species_id = _resolve_species(db, name)
                species_id_cache[name] = species_id
        detection = Detection(visit_id=visit.id, species_id=species_id, **rec)
        db.add(detection)
        detections.append(detection)

    visit.processed_at = utcnow()
    visit.ended_at = utcnow()
    visit.processing_error = None
    visit.scene_mask_suppressed = scene_mask_suppressed
    db.commit()
    log.info("visit %d: %d tracks persisted (some may be Unidentified)", visit.id, len(detections))

    # ---- Phase 3: side effects AFTER commit — no write lock held. ----
    # Save the source frame for each persisted track via a single clip
    # re-decode. These frames are the ingredient for a future YOLO
    # false-positive fine-tune; combined with Detection.bbox and the
    # user-applied Correction.correct_species_id, they form the (image, bbox,
    # label) triples the fine-tune will need.
    try:
        _save_source_frames(clip_path, frames_to_save, visit_id=visit.id)
    except Exception:  # noqa: BLE001
        # Non-fatal: missing source frames just means this visit's tracks
        # won't be available for YOLO fine-tune. Detection rows still land.
        log.exception("Failed to save source frames for visit %d", visit.id)

    # Notify subscribers of species not seen recently. Best-effort: failures
    # are non-fatal, and each dispatch runs its own tiny transaction (it may
    # prune dead subscriptions). Done after the main commit so the detections
    # are durable first and the write lock isn't held across push network I/O.
    for detection in detections:
        try:
            dispatch_for_detection(db, detection)
        except Exception:  # noqa: BLE001
            log.exception("Push dispatch failed for detection %d", detection.id)

    return len(tracks)


def _crop_quality(crop_bgr: "cv2.Mat", bbox: list) -> tuple[int | None, float | None, float | None]:
    """Compute (crop_area_px, brightness, sharpness) for the saved crop.

    Persisted on the Detection row so the feed can filter / sort by
    quality without re-loading every JPEG. Returns (None, None, None)
    for empty / unreadable crops.

    - crop_area_px: bbox area from the stored YOLO bbox (NOT the padded
      saved-crop area, so this is "how big was the bird itself"). Drives
      a "too small" filter at e.g. < 6400 px (~80×80).
    - brightness: mean of grayscale-converted crop, 0-255. Drives a
      "too dark" filter at e.g. < 30.
    - sharpness: Laplacian variance — same metric `_rank_detections`
      uses to score crop quality within a track. Drives a "too blurry"
      filter at e.g. < 30. Content-dependent so use as a rough signal.
    """
    if crop_bgr is None or crop_bgr.size == 0 or len(bbox) < 4:
        return None, None, None
    w, h = bbox[2], bbox[3]
    area = int(w * h)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return area, brightness, sharpness


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


# Step 2 (multi-frame fusion) toggle. The classifier-time helper below uses
# this to switch between the current behavior (classify the top-K crops
# independently, average the predictions) and the fused behavior (align +
# average the crops, classify the composite once). Sweep harness monkey-
# patches this at A/B time; production state is True unless reverted.
_USE_MULTI_FRAME_FUSION = True
# Reject phase-correlation alignments below this peak strength — below this
# the bird's pose has likely changed too much to fuse usefully and we drop
# the misaligned candidate. Tuned by hand on a few perched-vs-flying tracks;
# higher = stricter = falls back to single-crop more often.
_FUSION_MIN_CORR_PEAK = 0.10
# Pixel size we resize each crop to before fusion. Matches the classifier's
# input (260) plus a bit of headroom for the warpAffine border replication.
_FUSION_RESIZE_PX = 260


def _fuse_crops(crops: list) -> "cv2.Mat | None":
    """Align top-K sharpness-ranked crops via phase correlation and average.

    Inputs:
      - `crops`: list of BGR uint8 ndarrays, ordered best-first by the
        sharpness ranker. Caller filters out empties.

    Returns:
      - A fused BGR uint8 ndarray sized (_FUSION_RESIZE_PX × _FUSION_RESIZE_PX)
        when at least 2 crops align well; OR
      - The (resized) anchor when only the anchor remains after alignment-
        quality rejection; OR
      - None only if `crops` is empty.

    The anchor is the first (sharpest) crop. For each non-anchor crop we:
      1. Resize to the anchor's size so phase correlation has a common grid.
      2. Run cv2.phaseCorrelate to find the (dx, dy) translation that
         maximizes cross-correlation. Skip if the peak response is below
         _FUSION_MIN_CORR_PEAK — that means pose or content changed too
         much between the two frames for averaging to help (bird flapped
         a wing, turned its head, etc.).
      3. Translate the candidate by (-dx, -dy) so it aligns with the anchor.
      4. Stack and pixel-wise-average all aligned crops.

    The denoising win comes from the law-of-large-numbers reduction in
    sensor noise once N reasonably-aligned views are averaged (~ sqrt(N)
    SNR improvement). For dim crops where the classifier currently has
    low in-range probability mass, that extra signal is what tips it
    from rejected to accepted.
    """
    if not crops:
        return None
    target = (_FUSION_RESIZE_PX, _FUSION_RESIZE_PX)
    anchor = cv2.resize(crops[0], target)
    if len(crops) == 1:
        return anchor

    anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY).astype(np.float32)
    aligned = [anchor]
    for cand in crops[1:]:
        resized = cv2.resize(cand, target)
        try:
            cand_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(anchor_gray, cand_gray)
        except cv2.error:
            continue
        if response < _FUSION_MIN_CORR_PEAK:
            continue
        # Shift cand by (-dx, -dy) so it lines up with anchor.
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        warped = cv2.warpAffine(resized, M, target, borderMode=cv2.BORDER_REPLICATE)
        aligned.append(warped)

    if len(aligned) == 1:
        return anchor   # all candidates rejected; just use the anchor
    return np.mean(np.stack(aligned), axis=0).astype(np.uint8)


# Cap on how lopsided the saved crop is allowed to get after padding.
# YOLO sometimes returns very flat-wide or flat-tall boxes around
# partially-occluded birds (a head poking up over a perch, a tail
# sticking out behind ivy) — symmetric padding doesn't fix that, so the
# resulting crop looks like a useless strip in the feed AND deprives the
# classifier of the rest of the bird. After applying the standard 30 %
# pad, we expand the *short* axis further until aspect ≤ this cap (in
# either direction). Squarer crops give both human and model more to
# work with at near-zero CPU cost.
_CROP_ASPECT_CAP = 1.6


def _extract_crop_from_image(det, image: "cv2.Mat", padding: float = 0.30) -> "cv2.Mat":
    """Crop the padded bbox out of a frame image.

    Called once per detection at YOLO time so the full-resolution frame can
    be released immediately afterward (the cropped result is stored on
    det.crop — see process_visit). Padding bleeds context outside the YOLO
    box so:
      - The classifier sees surrounding plumage / perch context.
      - Tightly-fit YOLO bboxes (which often clip wings/tails) still show
        a visually complete bird in the saved feed crop.
      - Even if NMM in detect.py misses a tile-split fragment, the extra
        30 % around the (partial) detection often catches the rest of the
        bird in the image data we save to disk.

    Aspect-ratio cap: if the padded crop is more than _CROP_ASPECT_CAP×
    longer on one axis than the other, the short axis is extended
    symmetrically until that ratio is hit (clamped at the frame edge).
    This rescues partial-bird detections that would otherwise save as
    unusable horizontal/vertical strips.
    """
    x, y, w, h = det.bbox
    fh, fw = image.shape[:2]
    pad_w, pad_h = int(w * padding), int(h * padding)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(fw, x + w + pad_w)
    y1 = min(fh, y + h + pad_h)

    # Aspect-cap pass: pad the *short* axis until the crop is at most
    # _CROP_ASPECT_CAP×1 in either direction. Clipped at frame edges, so
    # an extreme bbox near the border just gets as much extra context as
    # the frame allows — never worse than the unpadded behavior.
    cw, ch = x1 - x0, y1 - y0
    if cw and ch:
        if cw / ch > _CROP_ASPECT_CAP:
            extra = int((cw / _CROP_ASPECT_CAP - ch) / 2)
            y0 = max(0, y0 - extra)
            y1 = min(fh, y1 + extra)
        elif ch / cw > _CROP_ASPECT_CAP:
            extra = int((ch / _CROP_ASPECT_CAP - cw) / 2)
            x0 = max(0, x0 - extra)
            x1 = min(fw, x1 + extra)

    return image[y0:y1, x0:x1]


def _rank_detections(track: Track) -> list:
    """Sort the track's detections by area × confidence × sharpness, desc.

    Each detection already carries its pre-extracted crop on `det.crop`
    (populated at YOLO time in process_visit). The +1 on sharpness avoids
    zeroing-out a small but confident detection just because its crop
    happened to land at low Laplacian variance.
    """
    scored: list[tuple[float, object]] = []
    for det in track.detections:
        crop = det.crop
        if crop is None or crop.size == 0:
            continue
        _, _, w, h = det.bbox
        sharpness = _laplacian_variance(crop)
        score = w * h * det.confidence * (sharpness + 1.0)
        scored.append((score, det))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored]


def _save_source_frames(
    clip_path: Path,
    frame_index_by_track: dict[int, int],
    *,
    visit_id: int,
    target_fps: float = 3.0,
) -> None:
    """Re-decode the clip and write out the source frame for each persisted track.

    Done as a post-processing step (rather than during the YOLO loop) because
    we otherwise have to keep all sampled frames in RAM through the track-
    finalize phase — that was the OOM trigger we just fixed. The cost of one
    extra clip decode per visit is ~5–10 % wall time, paid once per visit.

    The sampled frame index stored on `BirdDetection.frame_index` is the index
    after subsampling (3 fps from a 20 fps source = every 7th frame), so we
    multiply by `step` to recover the source-stream index, then seek there.

    No-op if the clip can't be reopened (file was deleted mid-process) — the
    Detection rows still land; only the frame archive is missing.
    """
    if not frame_index_by_track:
        return
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        log.warning("Couldn't reopen %s for source-frame extraction", clip_path)
        return
    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / target_fps)))
        for track_id, sampled_idx in frame_index_by_track.items():
            src_idx = sampled_idx * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
            ok, frame = cap.read()
            if not ok:
                log.debug(
                    "visit %d track %d: source frame %d unreadable; skipping",
                    visit_id, track_id, src_idx,
                )
                continue
            out_path = FRAMES_DIR / f"v{visit_id:08d}_t{track_id:04d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    finally:
        cap.release()


# Display-side image polish, applied to the crop we write to disk for the
# user-visible feed. Decoupled from the classifier's own pre-processing
# (which has its own CLAHE in classify.py) so we can dial display
# aesthetics without affecting model accuracy.
#
#   - CLAHE on the L channel of LAB to lift shadowed feather detail without
#     blowing out highlights or shifting color. Same setup as the classifier
#     uses; ~1 ms.
#   - Unsharp mask was tried at 0.5 → 0.3 → 0.15 amount and consistently
#     read as "crunchy" on feather edges. Disabled — the win wasn't worth
#     the artifact. The constant and helper paths are retained in case
#     we want to re-introduce a smarter sharpener later (e.g., edge-aware
#     or only-when-blurry), but the live path skips it entirely when
#     amount == 0.
_DISPLAY_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_UNSHARP_AMOUNT = 0.0
_UNSHARP_BLUR_KSIZE = (0, 0)         # auto-compute from sigma
_UNSHARP_BLUR_SIGMA = 1.5


def _polish_for_display(bgr: np.ndarray) -> np.ndarray:
    """Lighting-normalize (CLAHE) the user-visible feed crop. Sharpening
    is short-circuited when _UNSHARP_AMOUNT == 0 (no GaussianBlur /
    addWeighted cost)."""
    # CLAHE
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _DISPLAY_CLAHE.apply(l)
    eq = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    if _UNSHARP_AMOUNT <= 0:
        return eq
    # Unsharp mask: out = eq + amount * (eq - blur(eq))
    blurred = cv2.GaussianBlur(eq, _UNSHARP_BLUR_KSIZE, _UNSHARP_BLUR_SIGMA)
    sharpened = cv2.addWeighted(eq, 1 + _UNSHARP_AMOUNT, blurred, -_UNSHARP_AMOUNT, 0)
    return sharpened


def _save_crop(det, *, visit_id: int, track_id: int) -> Path:
    """Write a display-polished `det.crop` to disk and return the path
    relative to DATA_DIR.

    The polish (CLAHE + unsharp mask) is applied here, on the way to disk
    — the un-polished `det.crop` ndarray is what the classifier still
    receives (via its own preprocessing path in pipeline.classify). This
    keeps the user-facing image quality lever independent from the
    classifier's input distribution.
    """
    crop = det.crop
    assert crop is not None and crop.size > 0, "crop must be populated at detection time"
    polished = _polish_for_display(crop)
    filename = f"v{visit_id:08d}_t{track_id:04d}.jpg"
    out_path = CROPS_DIR / filename
    cv2.imwrite(str(out_path), polished, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path.relative_to(DATA_DIR)


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
