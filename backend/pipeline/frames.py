"""Frame extraction from clips OR snapshots.

Two input shapes the pipeline has to handle:

  - **Video clip** (.mp4, .webm, etc.): a typical Reolink motion clip is
    5–10 s at 15–25 fps. We sample at ~3 fps to capture bird pose changes
    without burning CPU on near-duplicates.
  - **Snapshot** (.jpg, .jpeg, .png): Reolink's motion-triggered FTP/SFTP
    upload sends still images, not video (Reolink doesn't support
    motion-triggered MP4 uploads — see DEVELOPING.md). We yield a single
    Frame; the rest of the pipeline treats that as a one-frame visit.

In either case the caller gets the same `Frame` stream and doesn't care
which underlying format produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Frame:
    index: int          # frame index within the sampled sequence, starting at 0
    timestamp: float    # seconds from start of clip
    image: np.ndarray   # H x W x 3 BGR (OpenCV convention)


def extract_frames(clip_path: Path, target_fps: float = 3.0) -> Iterator[Frame]:
    """Yield Frames decoded from `clip_path`.

    Dispatches on file extension. Video files are sampled at ~`target_fps`;
    image files yield a single Frame. Raises ValueError if the file can't
    be decoded.
    """
    ext = clip_path.suffix.lower()

    if ext in IMAGE_EXTS:
        image = cv2.imread(str(clip_path))
        if image is None:
            raise ValueError(f"Could not decode image: {clip_path}")
        yield Frame(index=0, timestamp=0.0, image=image)
        return

    # Default to video decoding for .mp4/.webm and any unrecognized extension —
    # better to try and fail loudly than to silently skip.
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
