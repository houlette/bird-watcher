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

from pipeline.exceptions import SkipFile

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Skip videos larger than this. process_visit holds every sampled frame in
# a dict keyed by frame index so the classifier can read crops from any of
# them — and a 30-second 4K clip at 3 fps means ~90 frames × ~25 MB each
# (decoded BGR uint8) ≈ 2 GB of RAM per visit. With multiple frames worth
# of YOLO + classifier overhead, that easily OOMs a small VM.
#
# 15 MB cap covers typical motion-event JPGs and short MP4 clips comfortably
# but rejects the 25-100 MB clips Reolink produces on long motion events.
# A future refactor that streams frames through processing (rather than
# caching them in a dict) can lift this cap.
MAX_VIDEO_BYTES = 15 * 1024 * 1024


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

    # Video path — enforce the size cap before even opening the file so we
    # never start a decode that's doomed to OOM us.
    size = clip_path.stat().st_size
    if size > MAX_VIDEO_BYTES:
        raise SkipFile(
            f"video too large to process: {size / 1e6:.1f} MB > {MAX_VIDEO_BYTES / 1e6:.0f} MB cap"
        )

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
