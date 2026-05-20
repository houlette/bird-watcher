"""YOLO11-nano bird detection.

YOLO11 ships with a COCO-trained model whose class 14 is 'bird'. That's
plenty for the first-pass "is there a bird in this frame" filter — it will
also reject squirrels (class 21 — 'sheep' / not a class), people, cars,
shadows, branches, etc., because they don't classify as 'bird'.

The expensive thing is loading the model, so we keep a process-wide singleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing torch at module import time
    from ultralytics import YOLO  # type: ignore

# COCO class index for 'bird'. Ultralytics ships this mapping with the model;
# we hard-code to make the intent explicit and to avoid a startup-time lookup.
COCO_BIRD_CLASS = 14

# Confidence threshold below which we ignore detections. Lowered from 0.30
# after real-world testing: a male cardinal at ~100 px on a 4K frame was
# scoring around 0.18-0.22 and getting dropped. 0.15 catches small/distant
# birds at the cost of more phantom detections — but those get re-filtered
# downstream by classify_bird's not_a_bird gate against the NA allow-list,
# so cheap to be lenient here.
BIRD_CONFIDENCE_THRESHOLD = 0.15

# YOLO downsamples to imgsz×imgsz before inference. Default 640 on our 4K
# frames means a 100-px bird collapses to ~17 px after downsample —
# borderline detectable. 1280 preserves enough detail to actually see them,
# at ~4× the CPU cost (acceptable; we're not real-time critical and
# processing one frame per JPG takes well under a second either way).
YOLO_IMGSZ = 1280

# Where the YOLO weights live (auto-downloaded on first run by ultralytics).
DEFAULT_WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "yolo11n.pt"


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
            # Import here so unit tests for tracking can run without torch.
            from ultralytics import YOLO  # noqa: WPS433
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            # If the file is missing, ultralytics fetches it automatically.
            _model = YOLO(str(weights_path) if weights_path.exists() else "yolo11n.pt")
    return _model


def detect_birds(frame_image: np.ndarray, frame_index: int) -> list[BirdDetection]:
    """Run YOLO on a single BGR frame and return only bird detections."""
    model = _get_model()
    results = model.predict(
        frame_image,
        classes=[COCO_BIRD_CLASS],
        conf=BIRD_CONFIDENCE_THRESHOLD,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    out: list[BirdDetection] = []
    # boxes.xyxy is a tensor of shape (N, 4); .conf is (N,)
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), conf in zip(xyxy, confs, strict=True):
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        if w <= 0 or h <= 0:
            continue
        out.append(BirdDetection(bbox=(x, y, w, h), confidence=float(conf), frame_index=frame_index))
    return out
