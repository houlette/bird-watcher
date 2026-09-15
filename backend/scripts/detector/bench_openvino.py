"""OpenVINO called directly: sync per tile, or all of a frame's tiles in parallel.

Usage:
  bench_ov2.py CLIP check                       direct path vs Ultralytics' OV predict
  bench_ov2.py CLIP sync  MODEL THREADS REPS
  bench_ov2.py CLIP async MODEL STREAMS THREADS REPS
MODEL is fp32, int8 or nano. Pre- and post-processing copy Ultralytics'
static-shape path (square letterbox, NMS iou 0.7, bird class only).
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")
import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils import ops  # noqa: E402

from pipeline.detect import BIRD_CONFIDENCE_THRESHOLD, COCO_BIRD_CLASS, TILE_PX, _tile_offsets  # noqa: E402
from pipeline.frames import extract_frames  # noqa: E402

EXPORT = Path("/export")
MODELS = {
    "fp32": EXPORT / "yolo11s_ov_static_openvino_model" / "yolo11s_ov_static.xml",
    "int8": EXPORT / "yolo11s_ov_int8_openvino_model" / "yolo11s_ov_int8.xml",
    "nano": EXPORT / "yolo11n_openvino_model" / "yolo11n.xml",
    "fp32dyn": EXPORT / "yolo11s_ov_dynamic_openvino_model" / "yolo11s_ov_dynamic.xml",
    "int8dyn": EXPORT / "yolo11s_ov_int8dyn_openvino_model" / "yolo11s_ov_int8dyn.xml",
    "nanodyn": EXPORT / "yolo11n_dyn_openvino_model" / "yolo11n_dyn.xml",
}
clip, mode = Path(sys.argv[1]), sys.argv[2]
frames = []
N_FRAMES = int(os.environ.get("BENCH_FRAMES", "2"))
for f in extract_frames(clip, target_fps=3.0):
    frames.append(f.image)
    if len(frames) >= N_FRAMES:
        break
rects = _tile_offsets(frames[0].shape[1], frames[0].shape[0])
letterbox = LetterBox(new_shape=(TILE_PX, TILE_PX), auto=False, stride=32)


def preprocess(tile):
    im = letterbox(image=tile)
    return np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0


def postprocess(raw, tile_shape):
    det = ops.non_max_suppression(torch.from_numpy(raw), BIRD_CONFIDENCE_THRESHOLD, 0.7, classes=[COCO_BIRD_CLASS])[0]
    if len(det):
        det[:, :4] = ops.scale_boxes((TILE_PX, TILE_PX), det[:, :4], tile_shape)
    return det[:, :5].numpy()


def tiles_of(img):
    return [img[y:y + th, x:x + tw] for x, y, tw, th in rects]


core = ov.Core()

if mode == "check":
    from ultralytics import YOLO

    compiled = core.compile_model(MODELS["fp32"], "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    ref = YOLO(str(MODELS["fp32"].parent), task="detect")
    kw = dict(classes=[COCO_BIRD_CLASS], conf=BIRD_CONFIDENCE_THRESHOLD, imgsz=TILE_PX, verbose=False)
    worst, n_ref, n_mine = 0.0, 0, 0
    for img in frames:
        for tile in tiles_of(img):
            mine = postprocess(compiled(preprocess(tile))[0], tile.shape)
            r = ref.predict(tile, **kw)[0].boxes
            theirs = np.concatenate([r.xyxy.numpy(), r.conf.numpy()[:, None]], 1) if len(r) else np.zeros((0, 5))
            n_ref += len(theirs)
            n_mine += len(mine)
            if len(mine) == len(theirs) and len(mine):
                worst = max(worst, float(np.abs(np.sort(mine, 0) - np.sort(theirs, 0)).max()))
            elif len(mine) != len(theirs):
                worst = float("inf")
    print(f"check: ultralytics {n_ref} boxes, direct {n_mine} boxes, worst coordinate/conf difference {worst:.4f}")
    sys.exit(0)

name = sys.argv[3]
if mode == "asyncrect":
    # Tiles keep their own size, padded only to a multiple of 32, exactly as
    # Ultralytics letterboxes them for the PyTorch model production runs.
    streams, threads, reps = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    m = core.read_model(MODELS[name])
    m.reshape({m.inputs[0].get_any_name(): ov.PartialShape([1, 3, -1, -1])})
    compiled = core.compile_model(m, "CPU", {
        "PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": streams, "INFERENCE_NUM_THREADS": threads,
    })
    queue = ov.AsyncInferQueue(compiled, streams)
    outputs = {}
    queue.set_callback(lambda req, i: outputs.__setitem__(i, req.get_output_tensor(0).data.copy()))
    rect_box = LetterBox(new_shape=(TILE_PX, TILE_PX), auto=True, stride=32)
    label = f"asyncrect {name} {streams}s{threads}t"

    def run_frame(img):
        ts = tiles_of(img)
        outputs.clear()
        shapes = []
        for i, t in enumerate(ts):
            im = rect_box(image=t)
            shapes.append(im.shape[:2])
            queue.start_async(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0, userdata=i)
        queue.wait_all()
        n = 0
        for i, t in enumerate(ts):
            det = ops.non_max_suppression(torch.from_numpy(outputs[i]), BIRD_CONFIDENCE_THRESHOLD, 0.7, classes=[COCO_BIRD_CLASS])[0]
            n += len(det)
        return n
elif mode == "sync":
    threads, reps = int(sys.argv[4]), int(sys.argv[5])
    compiled = core.compile_model(MODELS[name], "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads})
    label = f"sync  {name} {threads}t"

    def run_frame(img):
        return sum(len(postprocess(compiled(preprocess(t))[0], t.shape)) for t in tiles_of(img))
else:
    streams, threads, reps = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    compiled = core.compile_model(MODELS[name], "CPU", {
        "PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": streams, "INFERENCE_NUM_THREADS": threads,
    })
    queue = ov.AsyncInferQueue(compiled, streams)
    outputs = {}
    queue.set_callback(lambda req, i: outputs.__setitem__(i, req.get_output_tensor(0).data.copy()))
    label = f"async {name} {streams}s{threads}t"

    def run_frame(img):
        ts = tiles_of(img)
        outputs.clear()
        for i, t in enumerate(ts):
            queue.start_async(preprocess(t), userdata=i)
        queue.wait_all()
        return sum(len(postprocess(outputs[i], t.shape)) for i, t in enumerate(ts))

for img in frames:
    run_frame(img)  # warm-up
times, boxes = [], 0
for _ in range(reps):
    t0 = time.perf_counter()
    boxes = sum(run_frame(img) for img in frames)
    times.append((time.perf_counter() - t0) / (len(frames) * len(rects)))
import resource  # noqa: E402
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"{label:<22} s/tile {' '.join(f'{t:.3f}' for t in times)}  boxes {boxes}  peak_rss_mb {peak_mb:.0f}", flush=True)
