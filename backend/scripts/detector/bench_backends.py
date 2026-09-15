"""Per-tile YOLO cost by inference backend, one configuration per process.

Same frames, same tile slices (views, as detect_birds takes them), same predict
arguments as production. Usage: bench_backends.py CLIP CONFIG REPS
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
clip, config, reps = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
EXPORT = Path("/export")

ov_threads = {"ov_static_3": 3}.get(config)
if ov_threads:
    import openvino as ov

    _orig = ov.Core.compile_model

    def compile_with_threads(self, model, device_name=None, config=None, **kw):
        config = dict(config or {}, INFERENCE_NUM_THREADS=ov_threads)
        return _orig(self, model, device_name, config, **kw)

    ov.Core.compile_model = compile_with_threads

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from pipeline.detect import BIRD_CONFIDENCE_THRESHOLD, COCO_BIRD_CLASS, TILE_PX, _tile_offsets  # noqa: E402
from pipeline.frames import extract_frames  # noqa: E402

path = {
    "torch_3": EXPORT / "yolo11s.pt",
    "torch_4": EXPORT / "yolo11s.pt",
    "ov_static_4": EXPORT / "yolo11s_ov_static_openvino_model",
    "ov_static_3": EXPORT / "yolo11s_ov_static_openvino_model",
    "ov_dynamic_4": EXPORT / "yolo11s_ov_dynamic_openvino_model",
    "onnx_4": EXPORT / "yolo11s_onnx_static.onnx",
}[config]
model = YOLO(str(path), task="detect")

frames = []
for f in extract_frames(clip, target_fps=3.0):
    frames.append(f.image)
    if len(frames) >= 2:
        break
h, w = frames[0].shape[:2]
rects = _tile_offsets(w, h)
kw = dict(classes=[COCO_BIRD_CLASS], conf=BIRD_CONFIDENCE_THRESHOLD, imgsz=TILE_PX, verbose=False)


def run():
    boxes = 0
    for img in frames:
        for x, y, tw, th in rects:
            r = model.predict(img[y:y + th, x:x + tw], **kw)[0]
            boxes += 0 if r.boxes is None else len(r.boxes)
    return boxes


run()  # warm-up; also where Ultralytics sets torch threads
if config.startswith("torch"):
    torch.set_num_threads(int(config[-1]))
run()
times = []
for _ in range(reps):
    t0 = time.perf_counter()
    boxes = run()
    times.append((time.perf_counter() - t0) / (len(frames) * len(rects)))
import resource  # noqa: E402
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"{config:<13} s/tile {' '.join(f'{t:.3f}' for t in times)}  boxes {boxes}  torch_threads {torch.get_num_threads()}  peak_rss_mb {peak_mb:.0f}", flush=True)
