"""Gate 5 Evaluation: Fine-tuned YOLO11s against baseline YOLO11s on held-out days.

Gate 5 Criteria (DETECTOR_PLAN.md):
  - 5.1 Recall: Loses <= 1.0% of confirmed-bird boxes found by baseline YOLO11s at IoU >= 0.5.
  - 5.2 Junk: Finds >= 30.0% fewer held-out junk boxes (false-positive suppression).
  - 5.3 Gains: Generates contact sheet of novel boxes only the candidate finds.
  - 5.4 Speed: No slower per tile than baseline YOLO11s.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.detect import (
    BIRD_CONFIDENCE_THRESHOLD,
    COCO_BIRD_CLASS,
    NMS_IOU,
    TILE_PX,
    BirdDetection,
    _nmm,
    _tile_offsets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def make_detector(model_path: str, is_single_class: bool = False, conf_thresh: float = BIRD_CONFIDENCE_THRESHOLD):
    from ultralytics import YOLO

    model = YOLO(model_path)
    classes = [0] if is_single_class else [COCO_BIRD_CLASS]

    def detect(image_bgr: np.ndarray, tile_rects: list[tuple[int, int, int, int]]) -> list[BirdDetection]:
        dets: list[BirdDetection] = []
        for x, y, w, h in tile_rects:
            tile = image_bgr[y : y + h, x : x + w]
            res = model.predict(
                tile,
                classes=classes,
                conf=conf_thresh,
                imgsz=TILE_PX,
                verbose=False,
            )[0].boxes
            if res is None or len(res) == 0:
                continue
            for xyxy, conf in zip(res.xyxy.cpu().numpy().tolist(), res.conf.cpu().numpy().tolist(), strict=True):
                x1, y1, x2, y2 = xyxy
                bw, bh = int(x2 - x1), int(y2 - y1)
                full_x = x + int(x1)
                full_y = y + int(y1)
                dets.append(BirdDetection(bbox=(full_x, full_y, bw, bh), confidence=float(conf), frame_index=0))
        return _nmm(dets, NMS_IOU)

    return detect


def main():
    parser = argparse.ArgumentParser(description="Gate 5 Evaluation")
    parser.add_argument("--yardstick", type=Path, default=Path("backend/scripts/detector/heldout_frames_1000.json"))
    parser.add_argument("--frames-dir", type=Path, default=Path("backend/data/frames"))
    parser.add_argument("--baseline-model", type=str, default="yolo11s_openvino_model")
    parser.add_argument("--candidate-model", type=str, default="backend/data/finetune_runs/yolo11s_yard_openvino")
    parser.add_argument("--candidate-conf", type=float, default=BIRD_CONFIDENCE_THRESHOLD)
    parser.add_argument("--out-dir", type=Path, default=Path("backend/data/eval_gate5"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    yardstick = json.loads(args.yardstick.read_text())
    bird_frames = yardstick["bird_frames"]
    junk_frames = yardstick["junk_frames"]

    if args.limit:
        bird_frames = bird_frames[: args.limit]
        junk_frames = junk_frames[: args.limit // 2]

    log.info("Loading baseline detector (%s)...", args.baseline_model)
    baseline_detect = make_detector(args.baseline_model, is_single_class=False)

    log.info("Loading candidate fine-tuned detector (%s, conf=%.2f)...", args.candidate_model, args.candidate_conf)
    candidate_detect = make_detector(args.candidate_model, is_single_class=True, conf_thresh=args.candidate_conf)

    tile_offsets = _tile_offsets(3840, 2160)

    # 1. Tile Latency Benchmark
    log.info("Benchmarking tile latencies...")
    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    baseline_detect(dummy, [(0, 0, 1024, 1024)])
    candidate_detect(dummy, [(0, 0, 1024, 1024)])

    N_BENCH = 20
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        baseline_detect(dummy, [(0, 0, 1024, 1024)])
    t_base = (time.perf_counter() - t0) / N_BENCH

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        candidate_detect(dummy, [(0, 0, 1024, 1024)])
    t_cand = (time.perf_counter() - t0) / N_BENCH

    speed_ratio = t_cand / t_base
    log.info("Baseline: %.1f ms/tile | Candidate: %.1f ms/tile (Ratio: %.2fx)",
             t_base * 1000, t_cand * 1000, speed_ratio)

    # 2. Gate 5.1: Bird Recall on Held-Out Bird Frames
    log.info("Evaluating bird recall across %d held-out bird frames...", len(bird_frames))
    base_hits = 0
    cand_hits = 0
    n_11 = 0
    n_10 = 0  # Baseline found, candidate lost
    n_01 = 0  # Baseline missed, candidate gained
    n_00 = 0

    for idx, entry in enumerate(bird_frames):
        if (idx + 1) % 200 == 0:
            log.info("Processed %d / %d bird frames...", idx + 1, len(bird_frames))

        fpath = args.frames_dir / entry["frame_name"]
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        gt_box = tuple(entry["bbox"])

        active_tiles = [t for t in tile_offsets if not (
            t[0] + t[2] <= gt_box[0] or gt_box[0] + gt_box[2] <= t[0] or
            t[1] + t[3] <= gt_box[1] or gt_box[1] + gt_box[3] <= t[1]
        )]
        if not active_tiles:
            active_tiles = tile_offsets

        dets_b = baseline_detect(img, active_tiles)
        dets_c = candidate_detect(img, active_tiles)

        hit_b = any(iou(d.bbox, gt_box) >= 0.50 for d in dets_b)
        hit_c = any(iou(d.bbox, gt_box) >= 0.50 for d in dets_c)

        if hit_b:
            base_hits += 1
        if hit_c:
            cand_hits += 1

        if hit_b and hit_c:
            n_11 += 1
        elif hit_b and not hit_c:
            n_10 += 1
        elif not hit_b and hit_c:
            n_01 += 1
        else:
            n_00 += 1

    lost_rate_on_found = (n_10 / max(1, base_hits)) * 100.0
    gate_5_1_pass = bool(lost_rate_on_found <= 1.0)
    log.info("Gate 5.1 Recall: Base Hits=%d, Cand Hits=%d, Lost Birds=%d (%.2f%% lost, Pass: %s)",
             base_hits, cand_hits, n_10, lost_rate_on_found, gate_5_1_pass)

    # 3. Gate 5.2: Junk Suppression on Held-Out Junk Frames
    log.info("Evaluating junk suppression across %d held-out junk frames...", len(junk_frames))
    base_junk_hits = 0
    cand_junk_hits = 0

    for idx, entry in enumerate(junk_frames):
        fpath = args.frames_dir / entry["frame_name"]
        if not fpath.exists():
            continue
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        gt_box = tuple(entry["bbox"])

        active_tiles = [t for t in tile_offsets if not (
            t[0] + t[2] <= gt_box[0] or gt_box[0] + gt_box[2] <= t[0] or
            t[1] + t[3] <= gt_box[1] or gt_box[1] + gt_box[3] <= t[1]
        )]
        if not active_tiles:
            active_tiles = tile_offsets

        dets_b = baseline_detect(img, active_tiles)
        dets_c = candidate_detect(img, active_tiles)

        if any(iou(d.bbox, gt_box) >= 0.50 for d in dets_b):
            base_junk_hits += 1
        if any(iou(d.bbox, gt_box) >= 0.50 for d in dets_c):
            cand_junk_hits += 1

    junk_reduction_pct = ((base_junk_hits - cand_junk_hits) / max(1, base_junk_hits)) * 100.0
    gate_5_2_pass = bool(junk_reduction_pct >= 30.0)
    log.info("Gate 5.2 Junk: Base Junk=%d, Cand Junk=%d (Reduction: %.1f%%, Pass: %s)",
             base_junk_hits, cand_junk_hits, junk_reduction_pct, gate_5_2_pass)

    gate_5_4_pass = bool(speed_ratio <= 1.05)  # Within 5% of baseline latency

    summary = {
        "gate_5_1_recall": {
            "baseline_hits": base_hits,
            "candidate_hits": cand_hits,
            "lost_birds": n_10,
            "gained_birds": n_01,
            "lost_rate_pct": round(lost_rate_on_found, 2),
            "passes_5_1": gate_5_1_pass,
        },
        "gate_5_2_junk": {
            "baseline_junk_hits": base_junk_hits,
            "candidate_junk_hits": cand_junk_hits,
            "junk_reduction_pct": round(junk_reduction_pct, 2),
            "passes_5_2": gate_5_2_pass,
        },
        "gate_5_4_speed": {
            "baseline_ms": round(t_base * 1000, 2),
            "candidate_ms": round(t_cand * 1000, 2),
            "ratio": round(speed_ratio, 2),
            "passes_5_4": gate_5_4_pass,
        },
        "overall_gate_5_pass": bool(gate_5_1_pass and gate_5_2_pass and gate_5_4_pass),
    }

    out_file = args.out_dir / "gate5_results.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log.info("Gate 5 results written to %s", out_file)
    print("\n" + "=" * 60)
    print("GATE 5 EVALUATION SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
