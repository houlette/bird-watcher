"""Bird detection via tiled YOLO11 inference.

YOLO11 ships with a COCO-trained model whose class 14 is 'bird'. Used
directly on our 4K motion frames, even YOLO11-medium misses birds smaller
than ~150 px because the model internally downsamples to its training
resolution — a 100-px bird on a 3840×2160 frame collapses to ~17–35 px
after downsample and falls below detection thresholds.

To work around this we slice each frame into overlapping ~1024-px tiles,
run YOLO on each tile (so a 100-px bird is now ~10% of tile width, easy
to detect), translate detections back to full-image coordinates, and
NMS-merge to dedupe birds that straddle tile seams. This is what SAHI
does; we implemented it inline rather than depend on SAHI because their
0.11.x category_mapping handling broke for our use case and the manual
code is ~50 lines.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO  # type: ignore

log = logging.getLogger(__name__)

COCO_BIRD_CLASS = 14

# Confidence threshold. With tiling, birds-on-tile score much higher than
# birds-on-downsampled-full-frame, so we can be moderately strict.
BIRD_CONFIDENCE_THRESHOLD = 0.20

# Tiling parameters. A 3840×2160 frame tiled at 1024 with 20% overlap
# gives a 5×3 grid = 15 tiles. Each tile run at imgsz=1024 takes
# ~150-300 ms on CPU; full-frame detection ~2-5 s. Overlap lets us catch
# birds spanning two tiles; NMS then dedupes.
TILE_PX = 1024
TILE_OVERLAP_PX = int(TILE_PX * 0.20)

# NMS IoU threshold for merging cross-tile duplicates.
NMS_IOU = 0.50

# Which YOLO11 variant to use. Sized from nano (smallest/fastest, lowest
# accuracy) to extra-large. We sit at 's' (small) — a 2× param bump over
# nano that materially reduces false positives on bird-shaped non-birds
# (hummingbird feeders, leaves), in exchange for ~2-3× tile inference
# time. 'm' is still feasible on the 4-vCPU box (~20s/frame) if we want
# to push further later. Override via YOLO_WEIGHTS env var.
import os
YOLO_WEIGHTS_FILE = os.getenv("YOLO_WEIGHTS", "yolo11s.pt")
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / YOLO_WEIGHTS_FILE


@dataclass
class BirdDetection:
    """One bird detected in one frame."""

    bbox: tuple[int, int, int, int]   # x, y, w, h (pixels, top-left origin)
    confidence: float
    frame_index: int


_model_lock = Lock()
_model: "YOLO | None" = None


def _get_model(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> "YOLO":
    """Lazy-load and cache the YOLO model singleton."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO  # noqa: WPS433
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            # If the file is missing on disk, ultralytics fetches it from
            # its GitHub release by passing just the model name.
            _model = YOLO(str(weights_path) if weights_path.exists() else YOLO_WEIGHTS_FILE)
    return _model


def _tile_offsets(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Generate (x, y, w, h) tile rectangles covering width × height.

    Tiles are TILE_PX × TILE_PX with TILE_OVERLAP_PX overlap between
    neighbors. Edge tiles get clipped to the frame so the last column /
    row may be narrower than TILE_PX.
    """
    step = TILE_PX - TILE_OVERLAP_PX
    rects: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            w = min(TILE_PX, width - x)
            h = min(TILE_PX, height - y)
            rects.append((x, y, w, h))
            if x + TILE_PX >= width:
                break
            x += step
        if y + TILE_PX >= height:
            break
        y += step
    return rects


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two xywh boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[BirdDetection], iou_thresh: float) -> list[BirdDetection]:
    """Non-maximum suppression: drop lower-confidence duplicates of the same bird."""
    if not dets:
        return []
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list[BirdDetection] = []
    for d in sorted_dets:
        if any(_iou(d.bbox, k.bbox) >= iou_thresh for k in keep):
            continue
        keep.append(d)
    return keep


def detect_birds(frame_image: np.ndarray, frame_index: int) -> list[BirdDetection]:
    """Tiled YOLO bird detection on a single BGR frame."""
    model = _get_model()
    height, width = frame_image.shape[:2]

    raw: list[BirdDetection] = []
    for tile_x, tile_y, tile_w, tile_h in _tile_offsets(width, height):
        tile = frame_image[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]
        results = model.predict(
            tile,
            classes=[COCO_BIRD_CLASS],
            conf=BIRD_CONFIDENCE_THRESHOLD,
            imgsz=TILE_PX,  # tiles are already tile-sized; no waste downsample
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs, strict=True):
            # Translate tile-local coords back to full-frame coords.
            x = int(x1) + tile_x
            y = int(y1) + tile_y
            w = int(x2 - x1)
            h = int(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            raw.append(
                BirdDetection(
                    bbox=(x, y, w, h),
                    confidence=float(conf),
                    frame_index=frame_index,
                )
            )

    return _nms(raw, NMS_IOU)
