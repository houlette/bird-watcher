"""Replay harness — run `process_visit()` with arbitrary configuration overrides.

The trick is that production process_visit reads many module-level constants
and calls a handful of functions whose behavior we want to vary. We achieve
overrides by monkeypatching the live modules before each replay (and
restoring afterward). This avoids any production-code shims.

Public API:

    with apply_config(cfg):
        result = replay_visit(clip_path, ground_truth_visit_id, db)

Returns a ReplayResult describing the detections the pipeline produced for
that clip under that config.
"""
from __future__ import annotations

import contextlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# We have to ensure the parent backend dir is on sys.path so `pipeline.*`
# imports resolve when this is run as a script.
_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from db.models import Detection, Visit
from db.utils import utcnow

log = logging.getLogger(__name__)


@dataclass
class ReplayDetection:
    """One detection produced by a replayed pipeline run."""
    bbox: tuple[int, int, int, int]
    yolo_confidence: float
    fused_confidence: float
    species: str | None         # None == Unidentified
    track_id: int
    frame_index: int            # the chosen best frame_index for this track
    crop_path: str


@dataclass
class ReplayResult:
    visit_id: int                # ground-truth visit id this replay corresponds to
    clip_path: str
    config_label: str
    detections: list[ReplayDetection] = field(default_factory=list)
    error: str | None = None


# ────────────────────────────────────────────────────────────────────────
# Config application via monkeypatching
# ────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def apply_config(cfg: dict[str, Any]):
    """Apply `cfg` by monkeypatching the live pipeline modules. Restores
    everything on exit."""
    # Lazy import so the modules' import-time side effects (model loads, etc.)
    # don't fire at script import time.
    from pipeline import detect, frames, classify, scene_mask, track, process

    # Snapshot of values we'll restore on exit.
    saved: list[tuple[Any, str, Any]] = []

    def patch(obj, attr, value):
        if hasattr(obj, attr):
            saved.append((obj, attr, getattr(obj, attr)))
        else:
            saved.append((obj, attr, _MISSING))
        setattr(obj, attr, value)

    # ─── Simple constant overrides ────────────────────────────────────────
    for key, mod in (
        ("MAX_PROCESS_DURATION_SECONDS", frames),
        ("BIRD_CONFIDENCE_THRESHOLD", detect),
        ("TILE_PX", detect),
        ("TILE_OVERLAP_PX", detect),
        ("NMS_IOU", detect),
        ("TILE_SEAM_GAP_PX", detect),
        ("TILE_SEAM_OVERLAP_FRAC", detect),
        ("IN_RANGE_THRESHOLD", classify),
        ("GRID_PX", scene_mask),
        ("MIN_NABS_PER_CELL", scene_mask),
        ("OVERRIDE_YOLO_CONFIDENCE", scene_mask),
        ("LOOKBACK_DAYS", scene_mask),
        ("MATCH_IOU_THRESHOLD", track),
        ("MAX_MISSED_FRAMES", track),
    ):
        if key in cfg:
            patch(mod, key, cfg[key])

    # ─── target_fps: literal arg passed into extract_frames at the call site ─
    if "target_fps" in cfg:
        orig_extract = process.extract_frames
        target = cfg["target_fps"]
        def patched_extract(clip_path, target_fps=None):
            return orig_extract(clip_path, target_fps=target)
        patch(process, "extract_frames", patched_extract)

    # ─── crop padding: default-arg on _extract_crop_from_image ───────────
    if "crop_padding" in cfg:
        orig_crop = process._extract_crop_from_image
        pad = cfg["crop_padding"]
        def patched_crop(det, image, padding=None):
            return orig_crop(det, image, padding=pad)
        patch(process, "_extract_crop_from_image", patched_crop)

    # ─── YOLO model swap ──────────────────────────────────────────────────
    # detect._get_model is a singleton; resetting `_model` to None forces it
    # to reload on the next call. YOLO_WEIGHTS_FILE is read at call time, so
    # patching it tells the reload which weights to use. The function's
    # default `weights_path` arg still points at the original models/ dir
    # which won't exist locally → `_get_model` falls back to the patched
    # YOLO_WEIGHTS_FILE name and ultralytics auto-downloads it.
    if "yolo_model_name" in cfg:
        patch(detect, "_model", None)
        patch(detect, "YOLO_WEIGHTS_FILE", cfg["yolo_model_name"])

    # ─── Scene-mask toggle ────────────────────────────────────────────────
    if "scene_mask_enabled" in cfg and not cfg["scene_mask_enabled"]:
        # Identity filter: pass detections through unchanged.
        patch(process, "_scene_mask_filter", lambda dets, hot_zones=None: dets)

    # ─── NMM toggle (fall back to plain NMS-on-IoU-only) ─────────────────
    if "nmm_enabled" in cfg and not cfg["nmm_enabled"]:
        # The merge half of NMM happens in detect._nmm. Replace it with the
        # original strict NMS so duplicates are dropped, not unioned, and
        # tile-seam fragments are kept separate (which is the buggy old
        # behavior — that's the point of running this arm).
        from pipeline.detect import BirdDetection, _iou
        def plain_nms(dets, iou_thresh):
            if not dets:
                return []
            sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
            keep = []
            for d in sorted_dets:
                if any(_iou(d.bbox, k.bbox) >= iou_thresh for k in keep):
                    continue
                keep.append(d)
            return keep
        patch(detect, "_nmm", plain_nms)

    try:
        yield
    finally:
        for obj, attr, value in reversed(saved):
            if value is _MISSING:
                delattr(obj, attr)
            else:
                setattr(obj, attr, value)


_MISSING = object()


# ────────────────────────────────────────────────────────────────────────
# Replay a single visit
# ────────────────────────────────────────────────────────────────────────

def replay_visit(
    clip_path: Path,
    ground_truth_visit_id: int,
    started_at,
    db,                                # SQLAlchemy session for a SCRATCH DB
    config_label: str,
) -> ReplayResult:
    """Run process_visit on `clip_path` under whatever monkeypatched config is
    currently active. Returns the produced detections.

    `db` should be a session against a scratch DB that already has Species
    and (optionally) NAB-bearing rows seeded — scene_mask reads from it via
    SessionLocal, so we'd ordinarily need to monkeypatch SessionLocal too,
    but for the replay we want scene_mask to see the *same* historic NABs
    the production code does, so the simpler pattern is to use the pulled
    production DB and add this replay's outputs to it under a synthetic
    Visit, then query/strip them on the way out.
    """
    from pipeline.process import process_visit

    # Insert a scratch Visit row that points at the clip.
    visit = Visit(
        clip_path=str(clip_path),
        started_at=started_at,
        processed_at=None,
    )
    db.add(visit)
    db.flush()
    scratch_visit_id = visit.id

    result = ReplayResult(
        visit_id=ground_truth_visit_id,
        clip_path=str(clip_path),
        config_label=config_label,
    )

    try:
        process_visit(visit, db)
    except Exception as exc:  # noqa: BLE001
        result.error = repr(exc)
        log.exception("Replay failed for %s", clip_path)
        # Clean up the scratch visit and any partial detections.
        db.query(Detection).filter(Detection.visit_id == scratch_visit_id).delete()
        db.delete(visit)
        db.commit()
        return result

    # Read produced detections back out.
    rows = db.query(Detection).filter(Detection.visit_id == scratch_visit_id).all()
    for d in rows:
        # The species relationship may need a refresh.
        sp_name = d.species.common_name if d.species else None
        bbox = tuple(d.bbox) if d.bbox else (0, 0, 0, 0)
        result.detections.append(ReplayDetection(
            bbox=bbox,
            yolo_confidence=float(d.yolo_confidence or 0.0),
            fused_confidence=float(d.confidence or 0.0),
            species=sp_name,
            track_id=int(d.track_id),
            frame_index=-1,    # not stored on Detection; left -1 for now
            crop_path=d.crop_path,
        ))

    # Tear down: remove this replay's detections + visit so we don't pollute
    # the DB between runs.
    db.query(Detection).filter(Detection.visit_id == scratch_visit_id).delete()
    db.delete(visit)
    db.commit()

    return result
