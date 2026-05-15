"""Simple IoU-based multi-object tracker.

For backyard feeders the per-bird tracking problem is gentle: birds move in
small jumps between frames, and at 3 fps individual birds rarely cross paths
in a way that confuses simple overlap-based association. A full Kalman/SORT
implementation is overkill; this is a greedy IoU matcher with a few frames of
"keep alive" to bridge brief mis-detections.

The tracker is deliberately stateless w.r.t. ML: it consumes BirdDetection
records (from `detect.py`) but knows nothing about YOLO, so it can be unit-
tested without torch installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pipeline.detect import BirdDetection

# IoU above this counts as a match between an existing track and a new detection.
MATCH_IOU_THRESHOLD = 0.30

# Number of consecutive frames with no match before a track is considered closed.
# At 3 fps this is ~1 second of unseen, which comfortably bridges a bird
# turning its back or being briefly occluded.
MAX_MISSED_FRAMES = 3


@dataclass
class Track:
    """A single bird followed across frames."""

    track_id: int
    detections: list[BirdDetection] = field(default_factory=list)
    missed_frames: int = 0
    closed: bool = False

    @property
    def last_bbox(self) -> tuple[int, int, int, int]:
        return self.detections[-1].bbox

    @property
    def best_detection(self) -> BirdDetection:
        """Detection with the largest area * confidence — the 'best crop' for classification."""

        def score(d: BirdDetection) -> float:
            _, _, w, h = d.bbox
            return w * h * d.confidence

        return max(self.detections, key=score)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two axis-aligned bounding boxes given as (x, y, w, h)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU tracker — one instance per clip."""

    def __init__(self) -> None:
        self._tracks: list[Track] = []
        self._next_track_id = 1

    def update(self, frame_index: int, detections: Iterable[BirdDetection]) -> None:
        """Associate this frame's detections with existing tracks."""
        dets = list(detections)
        active = [t for t in self._tracks if not t.closed]
        unmatched_det_idxs = set(range(len(dets)))
        unmatched_track_idxs = set(range(len(active)))

        # Greedy matching: iterate pairs by descending IoU and take the best each step.
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(active):
            for di, det in enumerate(dets):
                score = iou(track.last_bbox, det.bbox)
                if score >= MATCH_IOU_THRESHOLD:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        for _, ti, di in pairs:
            if ti in unmatched_track_idxs and di in unmatched_det_idxs:
                active[ti].detections.append(dets[di])
                active[ti].missed_frames = 0
                unmatched_track_idxs.discard(ti)
                unmatched_det_idxs.discard(di)

        # Unmatched detections start new tracks.
        for di in unmatched_det_idxs:
            new = Track(track_id=self._next_track_id, detections=[dets[di]])
            self._next_track_id += 1
            self._tracks.append(new)

        # Unmatched existing tracks age; close them if they've gone too long.
        for ti in unmatched_track_idxs:
            t = active[ti]
            t.missed_frames += 1
            if t.missed_frames > MAX_MISSED_FRAMES:
                t.closed = True

        _ = frame_index  # accepted for future use (e.g. drift-based heuristics)

    def finalize(self) -> list[Track]:
        """Close any remaining open tracks and return all tracks seen."""
        for t in self._tracks:
            t.closed = True
        return list(self._tracks)
