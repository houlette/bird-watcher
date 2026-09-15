"""Does a faster detector still find the birds the current one finds?

Ground truth is the user's own label: detections whose latest user correction
names a bird (or Not a bird, for the junk set), on the saved source frame the
detection came from, with the stored box. Each model runs production's tiled
detection on the tiles that overlap that box (a box can only come from tiles
that overlap it), then production's cross-tile merge (_nmm). A box counts as
found at IoU >= 0.5.

Caveats: saved frames are JPEG q85 while production scores the decoded video
frame, so absolute found rates here are not production's; the comparison
between models on identical pixels is what this measures. Frames are the best
frame of tracks production already found, so this measures what a candidate
loses relative to the current detector, not birds both miss.

Usage: quality_eval.py N_BIRD N_JUNK MODELS...   (MODELS from torch fp32 int8 nano)
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/app")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from db.models import NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL, Detection, Visit  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline.detect import (  # noqa: E402
    BIRD_CONFIDENCE_THRESHOLD, COCO_BIRD_CLASS, NMS_IOU, TILE_PX, BirdDetection, _nmm, _tile_offsets,
)
from scripts.train import heldout  # noqa: E402

EXPORT = Path("/export")
FRAMES = Path("/app/data/frames")
n_bird, n_junk, model_names = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
torch.set_num_threads(1)
rng = random.Random(0)


# ---- sample: one frame per visit, round-robin across capture days ----
db = SessionLocal()
labels = heldout.latest_labels(db, sources=heldout.USER_SOURCES)
calib = set((EXPORT / "calibration_frames.txt").read_text().split())
names = set(os.listdir(FRAMES))
by_kind_day = {"bird": defaultdict(list), "junk": defaultdict(list)}
seen_visits = set()
rows = db.query(Detection.id, Detection.visit_id, Detection.track_id, Detection.bbox, Visit.started_at).join(Visit).all()
rng.shuffle(rows)
for det_id, visit_id, track_id, bbox, started in rows:
    name = f"v{visit_id:08d}_t{track_id:04d}.jpg"
    lab = labels.get(det_id)
    if lab is None or name not in names or name in calib or visit_id in seen_visits or lab[2] == POOR_QUALITY_LABEL:
        continue
    kind = "junk" if lab[2] == NOT_A_BIRD_LABEL else "bird"
    by_kind_day[kind][started.date().isoformat()].append((det_id, name, bbox))
    seen_visits.add(visit_id)


def round_robin(day_lists, n):
    days = sorted(day_lists)
    out, i = [], 0
    while len(out) < n and any(day_lists[d] for d in days):
        d = days[i % len(days)]
        if day_lists[d]:
            out.append(day_lists[d].pop())
        i += 1
    return out


sample = [("bird", *s) for s in round_robin(by_kind_day["bird"], n_bird)] + \
         [("junk", *s) for s in round_robin(by_kind_day["junk"], n_junk)]
n_days = len({d for k in by_kind_day for d in by_kind_day[k]})
print(f"sample: {sum(1 for s in sample if s[0] == 'bird')} bird, {sum(1 for s in sample if s[0] == 'junk')} junk frames", flush=True)


# ---- models ----
def torch_predictor():
    from ultralytics import YOLO
    m = YOLO(str(EXPORT / "yolo11s.pt"))
    kw = dict(classes=[COCO_BIRD_CLASS], conf=BIRD_CONFIDENCE_THRESHOLD, imgsz=TILE_PX, verbose=False)

    def predict(tile):
        r = m.predict(tile, **kw)[0].boxes
        return [] if r is None else list(zip(r.xyxy.numpy().tolist(), r.conf.numpy().tolist()))
    return predict


def ov_predictor(xml):
    """Dynamic-shape IR fed rectangular tiles, letterboxed exactly as
    Ultralytics does for the PyTorch model production runs."""
    import openvino as ov
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    core = ov.Core()
    m = core.read_model(xml)
    m.reshape({m.inputs[0].get_any_name(): ov.PartialShape([1, 3, -1, -1])})
    compiled = core.compile_model(m, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 1})
    lb = LetterBox(new_shape=(TILE_PX, TILE_PX), auto=True, stride=32)

    def predict(tile):
        im = lb(image=tile)
        x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
        det = ops.non_max_suppression(torch.from_numpy(compiled(x)[0]), BIRD_CONFIDENCE_THRESHOLD, 0.7, classes=[COCO_BIRD_CLASS])[0]
        if len(det):
            det[:, :4] = ops.scale_boxes(im.shape[:2], det[:, :4], tile.shape)
        return list(zip(det[:, :4].numpy().tolist(), det[:, 4].numpy().tolist()))
    return predict


factories = {
    "torch": torch_predictor,
    "fp32": lambda: ov_predictor(EXPORT / "yolo11s_ov_dynamic_openvino_model" / "yolo11s_ov_dynamic.xml"),
    "int8": lambda: ov_predictor(EXPORT / "yolo11s_ov_int8dyn_openvino_model" / "yolo11s_ov_int8dyn.xml"),
    "nano": lambda: ov_predictor(EXPORT / "yolo11n_dyn_openvino_model" / "yolo11n_dyn.xml"),
}
models = {name: factories[name]() for name in model_names}


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


# ---- run ----
results = []
t_start = time.time()
tile_time = defaultdict(float)
tile_count = 0
for i, (kind, det_id, name, gt) in enumerate(sample, 1):
    img = cv2.imread(str(FRAMES / name))
    if img is None:
        continue
    gx, gy, gw, gh = gt
    rects = [r for r in _tile_offsets(img.shape[1], img.shape[0])
             if r[0] < gx + gw and gx < r[0] + r[2] and r[1] < gy + gh and gy < r[1] + r[3]]
    tile_count += len(rects)
    rec = {"kind": kind, "det_id": det_id, "frame": name, "gt": gt, "tiles": len(rects)}
    for mname, predict in models.items():
        raw = []
        t0 = time.perf_counter()
        for x, y, tw, th in rects:
            for (x1, y1, x2, y2), conf in predict(img[y:y + th, x:x + tw]):
                w, h = int(x2 - x1), int(y2 - y1)
                if w > 0 and h > 0:
                    raw.append(BirdDetection(bbox=(int(x1) + x, int(y1) + y, w, h), confidence=float(conf), frame_index=0))
        tile_time[mname] += time.perf_counter() - t0
        merged = _nmm(raw, NMS_IOU)
        best = max(((iou(d.bbox, gt), d.confidence) for d in merged), default=(0.0, None))
        rec[mname] = {"iou": round(best[0], 3), "conf": None if best[1] is None else round(best[1], 3), "boxes": len(merged)}
    results.append(rec)
    if i % 25 == 0:
        print(f"  {i}/{len(sample)} frames, {time.time() - t_start:.0f}s", flush=True)

(EXPORT / f"quality_{'_'.join(model_names)}.json").write_text(json.dumps(results))


# ---- report ----
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - m) / d, (c + m) / d)


found = lambda r, m: r[m]["iou"] >= 0.5  # noqa: E731
print(f"\n{len(results)} frames, {tile_count} tiles; single-thread model time per tile: "
      + ", ".join(f"{m} {tile_time[m] / max(tile_count, 1):.2f}s" for m in models))
for kind in ("bird", "junk"):
    rs = [r for r in results if r["kind"] == kind]
    print(f"\n{kind}: {len(rs)} frames")
    for m in models:
        k = sum(found(r, m) for r in rs)
        lo, hi = wilson(k, len(rs))
        print(f"  {m:<6} finds the labelled box in {k}/{len(rs)} ({k / max(len(rs), 1):.1%}, 95% CI {lo:.1%}-{hi:.1%})")
    if "torch" in models:
        base = [r for r in rs if found(r, "torch")]
        for m in models:
            if m == "torch":
                continue
            lost = [r for r in base if not found(r, m)]
            gained = sum(1 for r in rs if found(r, m) and not found(r, "torch"))
            lo, hi = wilson(len(lost), len(base))
            both = [r for r in base if found(r, m)]
            deltas = sorted(r[m]["conf"] - r["torch"]["conf"] for r in both)
            flips = sum(1 for r in both if (r[m]["conf"] >= 0.65) != (r["torch"]["conf"] >= 0.65))
            med = deltas[len(deltas) // 2] if deltas else float("nan")
            print(f"  {m:<6} vs torch: lost {len(lost)}/{len(base)} torch finds ({len(lost) / max(len(base), 1):.1%}, 95% CI {lo:.1%}-{hi:.1%}), "
                  f"gained {gained}; conf delta median {med:+.3f} p10 {deltas[len(deltas) // 10] if deltas else float('nan'):+.3f} "
                  f"p90 {deltas[9 * len(deltas) // 10] if deltas else float('nan'):+.3f}; crosses 0.65 on {flips}")
            if lost:
                print(f"         lost: {[r['frame'] for r in lost][:12]}")
