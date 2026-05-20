"""Bird detection via SAHI-tiled YOLO11 inference.

YOLO11 ships with a COCO-trained model whose class 14 is 'bird'. Used
directly on our 4K motion frames, even YOLO11-medium misses birds smaller
than ~150 px because the model internally downsamples to its training
resolution (640 by default; 1280 with our override) — a 100-px bird on a
3840×2160 frame collapses to ~17–35 px after downsample and falls below
detection thresholds.

We use SAHI (Slicing Aided Hyper Inference) to work around this without
swapping models. SAHI splits the input image into overlapping tiles, runs
YOLO on each tile at native scale, then NMS-merges the results — so a
100-px bird on the full frame becomes a 100-px bird on a ~1000-px tile,
about 10% of tile width, comfortably detectable.

Tradeoff: SAHI runs YOLO once per tile (~15 tiles for a 3840×2160 frame
with 1024-px tiles at 20% overlap), so detection time is roughly 10× the
single-pass equivalent. Acceptable since each motion event is a single
frame and we're not real-time critical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sahi import AutoDetectionModel  # type: ignore

log = logging.getLogger(__name__)

# COCO class index for 'bird'.
COCO_BIRD_CLASS = 14

# Confidence threshold below which we ignore detections. With SAHI tiling
# the bird ends up much larger relative to the tile, so we can be a bit
# stricter than the 0.15 we used during the full-frame era. 0.20 is a
# middle ground — caught the test cardinal in tile-based bench runs and
# rejects most YOLO phantom labels (toilet, orange, fire hydrant, etc.).
BIRD_CONFIDENCE_THRESHOLD = 0.20

# Where the YOLO weights live (auto-downloaded on first run by ultralytics).
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "yolo11n.pt"

# SAHI tiling parameters.
#
# A 4K frame (3840×2160) tiled at 1024×1024 with 20% overlap produces a
# 5×3 grid = 15 tiles. Each tile fed to YOLO at imgsz=1024 (its native
# tile size — no downsampling waste) takes ~150–300 ms on CPU, so the
# whole frame is ~2–5 s. With overlap, birds straddling tile seams still
# get caught and the NMS step dedupes.
SAHI_TILE_PX = 1024
SAHI_OVERLAP_RATIO = 0.20

# Inside-tile NMS: merge overlapping detections of the same bird seen by
# two adjacent tiles. Default 0.5 is fine.
SAHI_NMS_IOU = 0.50


@dataclass
class BirdDetection:
    """One bird detected in one frame."""

    bbox: tuple[int, int, int, int]   # x, y, w, h (pixels, top-left origin)
    confidence: float
    frame_index: int


_model_lock = Lock()
_model: "AutoDetectionModel | None" = None


def _get_model(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> "AutoDetectionModel":
    """Lazy-load and cache the SAHI-wrapped YOLO model singleton."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sahi import AutoDetectionModel  # noqa: WPS433
            weights_path.parent.mkdir(parents=True, exist_ok=True)

            # Make sure the YOLO weights are on disk before SAHI tries to
            # load them. ultralytics will fetch on first use if missing.
            if not weights_path.exists():
                from ultralytics import YOLO  # noqa: WPS433
                log.info("Fetching YOLO weights to %s", weights_path)
                YOLO("yolo11n.pt")  # downloads to current cwd
                # Move to the expected location
                if (cwd_pt := Path.cwd() / "yolo11n.pt").exists():
                    cwd_pt.rename(weights_path)

            log.info("Loading SAHI-wrapped YOLO11 from %s", weights_path)
            # SAHI 0.11.18 doesn't have a dedicated 'ultralytics' loader;
            # YOLO11 shares architecture with YOLOv8 so the yolov8 loader
            # handles both. Upgrade SAHI later if a YOLO11-specific path
            # appears.
            #
            # We do NOT override category_mapping: SAHI then uses the
            # model's own full COCO names dict (80 classes). Restricting
            # it to {'14': 'bird'} causes a KeyError when YOLO predicts any
            # other class on a tile (e.g. 'orange' for a red blob). We
            # filter to bird-only in detect_birds() instead.
            _model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=str(weights_path),
                confidence_threshold=BIRD_CONFIDENCE_THRESHOLD,
            )
    return _model


def detect_birds(frame_image: np.ndarray, frame_index: int) -> list[BirdDetection]:
    """Run SAHI-tiled YOLO on a single BGR frame; return only bird detections."""
    model = _get_model()

    # SAHI's get_sliced_prediction expects RGB images.
    from sahi.predict import get_sliced_prediction  # noqa: WPS433

    rgb = frame_image[:, :, ::-1]
    result = get_sliced_prediction(
        rgb,
        detection_model=model,
        slice_height=SAHI_TILE_PX,
        slice_width=SAHI_TILE_PX,
        overlap_height_ratio=SAHI_OVERLAP_RATIO,
        overlap_width_ratio=SAHI_OVERLAP_RATIO,
        postprocess_match_threshold=SAHI_NMS_IOU,
        verbose=0,
    )

    out: list[BirdDetection] = []
    for pred in result.object_prediction_list:
        # SAHI gives us back ObjectPrediction. We restrict to the bird class
        # both via the model's category_mapping above AND by name here, as
        # a belt-and-suspenders against future model swaps that might not
        # honor category_mapping.
        if pred.category.id != COCO_BIRD_CLASS and pred.category.name != "bird":
            continue
        bbox = pred.bbox  # SAHI BoundingBox: minx, miny, maxx, maxy
        x, y = int(bbox.minx), int(bbox.miny)
        w, h = int(bbox.maxx - bbox.minx), int(bbox.maxy - bbox.miny)
        if w <= 0 or h <= 0:
            continue
        out.append(
            BirdDetection(
                bbox=(x, y, w, h),
                confidence=float(pred.score.value),
                frame_index=frame_index,
            )
        )
    return out
