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
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ultralytics import YOLO  # type: ignore

log = logging.getLogger(__name__)

COCO_BIRD_CLASS = 14

# Confidence threshold. With tiling, birds-on-tile score much higher than
# birds-on-downsampled-full-frame, so we can be moderately strict. Bumped
# from 0.20 → 0.35 after the sweep showed YOLO11n at 0.35 produces ~3×
# fewer detections (32 → 10 per 15-clip dataset) with no measurable loss
# on the FP-suppression axis. The dropped detections are mostly the
# "blurry blob" novel-FP class polluting the feed.
BIRD_CONFIDENCE_THRESHOLD = 0.35

# Tiling parameters. A 3840×2160 frame tiled at 1024 with 20% overlap
# gives a 5×3 grid = 15 tiles. Each tile run at imgsz=1024 takes
# ~150-300 ms on CPU; full-frame detection ~2-5 s. Overlap lets us catch
# birds spanning two tiles; NMS then dedupes.
TILE_PX = 1024
# 20% overlap (≈205 px) catches most small birds fully inside a single tile.
# Larger birds that straddle tile seams emit two half-bird detections; the
# _nmm pass below merges those back into one. We previously bumped this to
# 40% but the resulting +1 tile per row pushed the API container over its
# 4 GB memory limit during peak feeder activity (OOM-killed 2026-05-26),
# because PyTorch's intermediate-tensor pool grows roughly with the number
# of consecutive YOLO calls and we'd raised that count by 2.4×. NMM gives
# us the half-bird fix at no extra CPU/memory cost, so the structural
# overlap bump isn't necessary.
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
    """One bird detected in one frame.

    `crop` is populated by `process.py` right after detection runs, so the
    full-resolution source frame can be released before the next frame is
    decoded. Carrying the crop on the detection itself lets the tracker
    pass detections downstream without us also having to thread a frame
    cache through every function — and the crop is tiny (~120 KB at
    typical bird sizes) vs ~25 MB for a cached 4K frame.
    """

    bbox: tuple[int, int, int, int]   # x, y, w, h (pixels, top-left origin)
    confidence: float
    frame_index: int
    crop: Any = field(default=None, repr=False, compare=False)  # H×W×3 BGR ndarray once populated


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


def _box_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The smallest xywh box that contains both `a` and `b`."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


# A pair of boxes is treated as "two halves of the same bird split across a
# tile seam" if the gap along one axis is ≤ this many pixels AND they overlap
# substantially on the perpendicular axis (see _is_tile_fragment_pair). 20 px
# absorbs YOLO's per-tile bbox imprecision near the seam without merging two
# birds perched 30+ px apart on the same branch.
TILE_SEAM_GAP_PX = 20
TILE_SEAM_OVERLAP_FRAC = 0.5


def _is_tile_fragment_pair(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True if `a` and `b` look like two fragments of one bird across a tile seam.

    Tile-seam fragments share an edge: their gap on one axis is tiny while
    they line up almost completely on the perpendicular axis (same bird's
    top/bottom or left/right). Two distinct birds perched near each other
    have a real gap on both axes OR a misalignment on the perpendicular.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    # Vertical seam (boxes side-by-side, small x-gap, large y-overlap).
    x_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    if x_gap <= TILE_SEAM_GAP_PX and y_overlap >= TILE_SEAM_OVERLAP_FRAC * min(ah, bh):
        return True
    # Horizontal seam (boxes stacked, small y-gap, large x-overlap).
    y_gap = max(0, max(ay, by) - min(ay + ah, by + bh))
    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    if y_gap <= TILE_SEAM_GAP_PX and x_overlap >= TILE_SEAM_OVERLAP_FRAC * min(aw, bw):
        return True
    return False


def _nmm(dets: list[BirdDetection], iou_thresh: float) -> list[BirdDetection]:
    """Non-Maximum Merging: drops or expands rather than just suppresses.

    Two cases handled:
      1. Overlapping duplicates (IoU ≥ iou_thresh): a fully-detected bird in
         one tile + a partial detection of the same bird in the adjacent
         tile's overlap zone. We merge into the union bbox (instead of
         dropping the partial outright) so the kept box represents the
         largest extent both tiles agreed on. Keeps the higher confidence.
      2. Tile-seam fragments (zero/near-zero IoU but spatially adjacent
         and aligned, see _is_tile_fragment_pair): two halves of one bird
         that each tile saw partially. Merge into the union bbox.

    Iterates in confidence-desc order; the surviving box for a merged pair
    is therefore the higher-confidence one with its bbox grown to contain
    both, which is exactly what we want for the crop downstream.
    """
    if not dets:
        return []
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list[BirdDetection] = []
    for d in sorted_dets:
        merged = False
        for i, k in enumerate(keep):
            if _iou(d.bbox, k.bbox) >= iou_thresh or _is_tile_fragment_pair(d.bbox, k.bbox):
                keep[i] = BirdDetection(
                    bbox=_box_union(k.bbox, d.bbox),
                    confidence=k.confidence,   # higher of the two (k was kept first)
                    frame_index=k.frame_index,
                )
                merged = True
                break
        if not merged:
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

    return _nmm(raw, NMS_IOU)
