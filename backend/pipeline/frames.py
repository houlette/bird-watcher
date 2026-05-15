"""Extract sampled frames from a motion-event clip.

A typical Reolink motion clip is 5–10 seconds at 15–25 fps. For the
detection/classification pipeline we don't need every frame; sampling at
3 fps captures bird poses across a visit without burning CPU on near-duplicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    index: int          # frame index within the sampled sequence, starting at 0
    timestamp: float    # seconds from start of clip
    image: np.ndarray   # H x W x 3 BGR (OpenCV convention)


def extract_frames(clip_path: Path, target_fps: float = 3.0) -> Iterator[Frame]:
    """Yield decoded frames sampled at approximately `target_fps`.

    Falls back to yielding every frame if the source FPS is lower than the
    target. Raises ValueError if the clip cannot be opened.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open clip: {clip_path}")

    try:
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / target_fps)))

        out_idx = 0
        src_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_idx % step == 0:
                yield Frame(index=out_idx, timestamp=src_idx / src_fps, image=frame)
                out_idx += 1
            src_idx += 1
    finally:
        cap.release()
