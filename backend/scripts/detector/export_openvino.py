"""Export yolo11s to OpenVINO and ONNX (FP32) and sanity-check one frame.

Runs in a throwaway container with openvino and onnx pip-installed; writes to
/export. Each export gets its own copy of the weights so the output folders do
not collide.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
from ultralytics import YOLO  # noqa: E402

from pipeline.detect import BIRD_CONFIDENCE_THRESHOLD, COCO_BIRD_CLASS, TILE_PX, _tile_offsets  # noqa: E402
from pipeline.frames import extract_frames  # noqa: E402

EXPORT = Path("/export")
SRC = EXPORT / "yolo11s.pt"
variants = {
    "ov_static": dict(format="openvino", imgsz=TILE_PX, dynamic=False, half=False),
    "ov_dynamic": dict(format="openvino", imgsz=TILE_PX, dynamic=True, half=False),
    "onnx_static": dict(format="onnx", imgsz=TILE_PX, dynamic=False, simplify=True),
}
for name, kw in variants.items():
    stem = EXPORT / f"yolo11s_{name}.pt"
    shutil.copy(SRC, stem)
    t0 = time.perf_counter()
    out = YOLO(str(stem)).export(**kw)
    print(f"{name}: exported to {out} in {time.perf_counter() - t0:.0f}s", flush=True)

clip = Path(sys.argv[1])
frame = next(iter(extract_frames(clip, target_fps=3.0))).image
h, w = frame.shape[:2]
rects = _tile_offsets(w, h)
kw = dict(classes=[COCO_BIRD_CLASS], conf=BIRD_CONFIDENCE_THRESHOLD, imgsz=TILE_PX, verbose=False)
models = {
    "torch": YOLO(str(SRC)),
    "ov_static": YOLO(str(EXPORT / "yolo11s_ov_static_openvino_model"), task="detect"),
    "ov_dynamic": YOLO(str(EXPORT / "yolo11s_ov_dynamic_openvino_model"), task="detect"),
    "onnx_static": YOLO(str(EXPORT / "yolo11s_onnx_static.onnx"), task="detect"),
}
for name, m in models.items():
    boxes = 0
    t0 = time.perf_counter()
    for x, y, tw, th in rects:
        r = m.predict(frame[y:y + th, x:x + tw], **kw)[0]
        boxes += 0 if r.boxes is None else len(r.boxes)
    print(f"sanity {name}: {boxes} boxes on one frame, {time.perf_counter() - t0:.1f}s (contended, not a benchmark)", flush=True)
