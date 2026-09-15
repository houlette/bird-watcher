"""Gate 1.5a: the shipped detect_birds, PyTorch against OpenVINO, on production pixels.

Runs inside the deployed api image so it exercises pipeline.detect exactly as
the worker will. Frames are the sampled frames of recent clips at which
production's tracks held a box (from Detection.track_frames), so the sample is
made of frames with birds or junk in them rather than empty scenery. It keeps
adding clips until PyTorch has produced at least MIN_BOXES merged boxes.

Pass (DETECTOR_PLAN.md): at least 98% of PyTorch's merged boxes matched by an
OpenVINO box at IoU 0.5, total box counts within 2%, and at least MIN_BOXES
PyTorch boxes in the sample.

Usage (on the VM):
    docker compose run --rm --no-deps -e YOLO_OPENVINO_STREAMS=1 -e YOLO_OPENVINO_THREADS=1 \\
        api python scripts/detector/parity_integration.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from db.models import Detection, Visit  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline.detect import detect_birds  # noqa: E402
from pipeline.frames import extract_frames  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
MIN_BOXES = 100
MAX_CLIPS = 60
FRAMES_PER_CLIP = 3


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def main() -> int:
    torch.set_num_threads(1)
    rng = random.Random(0)
    db = SessionLocal()
    rows = (
        db.query(Visit.id, Visit.clip_path, Detection.track_frames)
        .join(Detection, Detection.visit_id == Visit.id)
        .filter(Visit.clip_path.like("%.mp4"), Detection.track_frames.isnot(None))
        .order_by(Visit.id.desc())
        .limit(2000)
        .all()
    )
    by_visit: dict[int, tuple[str, set[int]]] = {}
    for vid, path, frames in rows:
        if (DATA / path).exists():
            by_visit.setdefault(vid, (path, set()))[1].update(frames or [])
    visits = sorted(by_visit)
    rng.shuffle(visits)
    print(f"{len(visits)} recent clips with tracked boxes still on disk", flush=True)

    n_torch = n_ov = n_matched = n_frames = n_clips = 0
    for vid in visits[:MAX_CLIPS]:
        if n_torch >= MIN_BOXES:
            break
        path, indices = by_visit[vid]
        wanted = set(rng.sample(sorted(indices), min(FRAMES_PER_CLIP, len(indices))))
        n_clips += 1
        for frame in extract_frames(DATA / path, target_fps=3.0):
            if frame.index not in wanted:
                continue
            ref = detect_birds(frame.image, frame.index, backend="torch")
            stats: dict = {}
            got = detect_birds(frame.image, frame.index, stats=stats, backend="openvino")
            if stats.get("backend") != "openvino":
                print("OpenVINO backend did not load; see the log above.")
                return 2
            n_frames += 1
            n_torch += len(ref)
            n_ov += len(got)
            used: set[int] = set()
            for r in ref:
                best = max(((iou(r.bbox, g.bbox), i) for i, g in enumerate(got) if i not in used), default=(0.0, None))
                if best[0] >= 0.5:
                    n_matched += 1
                    used.add(best[1])
        print(f"  clip {n_clips} (visit {vid}): torch boxes so far {n_torch}, openvino {n_ov}, matched {n_matched}", flush=True)

    share = n_matched / n_torch if n_torch else 0.0
    diff = (n_ov - n_torch) / n_torch if n_torch else 0.0
    ok = n_torch >= MIN_BOXES and share >= 0.98 and abs(diff) <= 0.02
    print(f"\n{n_frames} frames from {n_clips} clips: PyTorch {n_torch} boxes, OpenVINO {n_ov} ({diff:+.1%}), "
          f"matched {n_matched} ({share:.1%})  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
