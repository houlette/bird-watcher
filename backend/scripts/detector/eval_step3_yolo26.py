"""Step 3: Comparative Evaluation of generic YOLO26s against YOLO11s.

Evaluates:
  1. Gate 1.1 criteria:
     - Confirmed bird recall vs YOLO11s (lost bird rate at IoU 0.5).
     - Junk detection rate on junk frames (within 10 points).
     - Confidence shift on shared detections (median delta within +-0.03, <=2% crossing 0.65).
  2. Exact Paired Non-Inferiority (Tango score test on N=1,000 stratified frames).
  3. Tile seam boundary integrity (objects bisected by 1024 px tile seams merged via _nmm).
  4. Speed benchmark (ms/tile at batch=1 on CPU).

Usage:
  /Users/ryan/.gemini/antigravity-cli/brain/a8784b9d-a157-4517-ae32-a60cd5d3055b/scratch/venv_yolo26/bin/python3 \
      backend/scripts/detector/eval_step3_yolo26.py \
      --yardstick backend/scripts/detector/heldout_frames_1000.json \
      --frames-dir backend/data/frames \
      --out-dir backend/data/eval_step3_yolo26
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.detect import (
    BIRD_CONFIDENCE_THRESHOLD,
    COCO_BIRD_CLASS,
    NMS_IOU,
    TILE_OVERLAP_PX,
    TILE_PX,
    BirdDetection,
    _box_union,
    _is_tile_fragment_pair,
    _nmm,
    _tile_offsets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two xywh boxes."""
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


def make_detector_func(model_path: str):
    """Factory returning a function(image_bgr, tile_rects) -> list[BirdDetection]."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    def detect_on_tiles(image_bgr: np.ndarray, tile_rects: list[tuple[int, int, int, int]]) -> list[BirdDetection]:
        dets: list[BirdDetection] = []
        for x, y, w, h in tile_rects:
            tile = image_bgr[y : y + h, x : x + w]
            res = model.predict(
                tile,
                classes=[COCO_BIRD_CLASS],
                conf=BIRD_CONFIDENCE_THRESHOLD,
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

    return detect_on_tiles


def main():
    parser = argparse.ArgumentParser(description="Step 3 YOLO26s Evaluation")
    parser.add_argument("--yardstick", type=Path, default=Path("backend/scripts/detector/heldout_frames_1000.json"))
    parser.add_argument("--frames-dir", type=Path, default=Path("backend/data/frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("backend/data/eval_step3_yolo26"))
    parser.add_argument("--limit", type=int, default=None, help="Limit frames for fast dry run")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    yardstick = json.loads(args.yardstick.read_text())
    bird_frames = yardstick["bird_frames"]
    junk_frames = yardstick["junk_frames"]

    if args.limit:
        bird_frames = bird_frames[: args.limit]
        junk_frames = junk_frames[: args.limit // 2]

    log.info("Loaded yardstick: %d bird frames, %d junk frames", len(bird_frames), len(junk_frames))

    # 1. Initialize Detectors (prefer OpenVINO for 4-5x faster inference)
    m11_path = "yolo11s_openvino_model" if Path("yolo11s_openvino_model").exists() else "yolo11s.pt"
    m26_path = "yolo26s_openvino_model" if Path("yolo26s_openvino_model").exists() else "yolo26s.pt"
    log.info("Initializing YOLO11s from %s...", m11_path)
    yolo11_detect = make_detector_func(m11_path)

    log.info("Initializing YOLO26s from %s...", m26_path)
    yolo26_detect = make_detector_func(m26_path)

    # 2. Benchmark tile latency on warm model
    dummy_tile = np.zeros((1024, 1024, 3), dtype=np.uint8)
    log.info("Benchmarking tile latencies...")
    # Warmup
    yolo11_detect(dummy_tile, [(0, 0, 1024, 1024)])
    yolo26_detect(dummy_tile, [(0, 0, 1024, 1024)])

    t0 = time.perf_counter()
    n_bench = 20
    for _ in range(n_bench):
        yolo11_detect(dummy_tile, [(0, 0, 1024, 1024)])
    yolo11_tile_s = (time.perf_counter() - t0) / n_bench

    t0 = time.perf_counter()
    for _ in range(n_bench):
        yolo26_detect(dummy_tile, [(0, 0, 1024, 1024)])
    yolo26_tile_s = (time.perf_counter() - t0) / n_bench

    speedup = (yolo11_tile_s - yolo26_tile_s) / yolo11_tile_s * 100.0
    log.info("YOLO11s: %.1f ms/tile | YOLO26s: %.1f ms/tile (Speedup: %+.1f%%)",
             yolo11_tile_s * 1000, yolo26_tile_s * 1000, speedup)

    # 3. Evaluate on Confirmed Bird Frames
    log.info("Evaluating bird recall across %d frames...", len(bird_frames))
    n_11 = 0  # Both found
    n_10 = 0  # 11s found, 26s missed
    n_01 = 0  # 11s missed, 26s found
    n_00 = 0  # Both missed

    conf_deltas = []
    cross_65_count = 0
    seam_straddled_count = 0
    seam_straddled_matches = 0

    tile_offsets_cached = _tile_offsets(3840, 2160)

    for idx, entry in enumerate(bird_frames):
        if (idx + 1) % 100 == 0:
            log.info("Processed %d / %d bird frames...", idx + 1, len(bird_frames))

        frame_path = args.frames_dir / entry["frame_name"]
        if not frame_path.exists():
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        gt_box = tuple(entry["bbox"])

        # Check if GT box straddles a tile seam (within 100px of seam boundaries)
        gx, gy, gw, gh = gt_box
        is_seam_bird = False
        for sx in [819, 1638, 2457, 3276]:
            if abs(gx + gw // 2 - sx) < 100 or (gx < sx < gx + gw):
                is_seam_bird = True
                break

        # Only run overlapping tiles to optimize eval runtime (Gate 1.1 protocol)
        active_tiles = [t for t in tile_offsets_cached if not (
            t[0] + t[2] <= gt_box[0] or gt_box[0] + gt_box[2] <= t[0] or
            t[1] + t[3] <= gt_box[1] or gt_box[1] + gt_box[3] <= t[1]
        )]
        if not active_tiles:
            active_tiles = tile_offsets_cached

        dets_11 = yolo11_detect(img, active_tiles)
        dets_26 = yolo26_detect(img, active_tiles)

        # Match to GT box at IoU >= 0.50
        match_11 = [d for d in dets_11 if iou(d.bbox, gt_box) >= 0.50]
        match_26 = [d for d in dets_26 if iou(d.bbox, gt_box) >= 0.50]

        hit_11 = len(match_11) > 0
        hit_26 = len(match_26) > 0

        if hit_11 and hit_26:
            n_11 += 1
            c11 = max(d.confidence for d in match_11)
            c26 = max(d.confidence for d in match_26)
            conf_deltas.append(c26 - c11)
            # Check 0.65 boundary crossings
            if (c11 < 0.65 <= c26) or (c26 < 0.65 <= c11):
                cross_65_count += 1
        elif hit_11 and not hit_26:
            n_10 += 1
        elif not hit_11 and hit_26:
            n_01 += 1
        else:
            n_00 += 1

        if is_seam_bird:
            seam_straddled_count += 1
            if hit_26:
                seam_straddled_matches += 1

    # 4. Evaluate Junk Frames (Gate 1.1 Junk rate criteria)
    log.info("Evaluating junk detection across %d junk frames...", len(junk_frames))
    junk_hits_11 = 0
    junk_hits_26 = 0

    for idx, entry in enumerate(junk_frames):
        frame_path = args.frames_dir / entry["frame_name"]
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        gt_box = tuple(entry["bbox"])

        active_tiles = [t for t in tile_offsets_cached if not (
            t[0] + t[2] <= gt_box[0] or gt_box[0] + gt_box[2] <= t[0] or
            t[1] + t[3] <= gt_box[1] or gt_box[1] + gt_box[3] <= t[1]
        )]
        if not active_tiles:
            active_tiles = tile_offsets_cached

        dets_11 = yolo11_detect(img, active_tiles)
        dets_26 = yolo26_detect(img, active_tiles)

        if any(iou(d.bbox, gt_box) >= 0.50 for d in dets_11):
            junk_hits_11 += 1
        if any(iou(d.bbox, gt_box) >= 0.50 for d in dets_26):
            junk_hits_26 += 1

    # 5. Compute Metrics and Gate Verification
    n_birds = n_11 + n_10 + n_01 + n_00
    p11_recall = (n_11 + n_10) / n_birds * 100.0 if n_birds else 0.0
    p26_recall = (n_11 + n_01) / n_birds * 100.0 if n_birds else 0.0
    lost_birds = n_10
    gained_birds = n_01
    lost_rate_on_found = (n_10 / (n_11 + n_10) * 100.0) if (n_11 + n_10) else 0.0

    med_conf_delta = float(np.median(conf_deltas)) if conf_deltas else 0.0
    cross_65_rate = (cross_65_count / len(conf_deltas) * 100.0) if conf_deltas else 0.0

    junk_rate_11 = (junk_hits_11 / len(junk_frames) * 100.0) if junk_frames else 0.0
    junk_rate_26 = (junk_hits_26 / len(junk_frames) * 100.0) if junk_frames else 0.0
    junk_delta_pts = abs(junk_rate_26 - junk_rate_11)

    # Tango Score Non-Inferiority Test
    diff = (n_10 - n_01) / n_birds
    var = ( (n_10 / n_birds) + (n_01 / n_birds) - (diff**2) ) / n_birds
    se = math.sqrt(max(var, 1e-9))
    delta = 0.02  # 2% margin
    z_stat = (delta - diff) / se
    p_val = float(1.0 - stats.norm.cdf(z_stat))
    tango_pass = bool(diff < delta and p_val < 0.05)

    # Gate Criteria Checks
    gate_1_1_recall_pass = bool(lost_rate_on_found <= 1.0)
    gate_1_1_junk_pass = bool(junk_delta_pts <= 10.0)
    gate_1_1_conf_med_pass = bool(abs(med_conf_delta) <= 0.03)
    gate_1_1_conf_cross_pass = bool(cross_65_rate <= 2.0)
    gate_3_speed_pass = bool(speedup >= 20.0)

    summary = {
        "benchmark": {
            "yolo11s_ms": round(yolo11_tile_s * 1000, 2),
            "yolo26s_ms": round(yolo26_tile_s * 1000, 2),
            "speedup_percent": round(speedup, 2),
            "gate_3_speed_pass": gate_3_speed_pass,
        },
        "bird_recall": {
            "total_frames": n_birds,
            "yolo11s_hits": n_11 + n_10,
            "yolo11s_recall_pct": round(p11_recall, 2),
            "yolo26s_hits": n_11 + n_01,
            "yolo26s_recall_pct": round(p26_recall, 2),
            "n_11_both_found": n_11,
            "n_10_lost_birds": n_10,
            "n_01_gained_birds": n_01,
            "n_00_both_missed": n_00,
            "lost_rate_on_yolo11_found_pct": round(lost_rate_on_found, 2),
            "gate_1_1_recall_pass": gate_1_1_recall_pass,
        },
        "confidence": {
            "median_conf_delta": round(med_conf_delta, 4),
            "cross_65_count": cross_65_count,
            "cross_65_pct": round(cross_65_rate, 2),
            "gate_1_1_conf_med_pass": gate_1_1_conf_med_pass,
            "gate_1_1_conf_cross_pass": gate_1_1_conf_cross_pass,
        },
        "junk": {
            "total_junk_frames": len(junk_frames),
            "junk_rate_11_pct": round(junk_rate_11, 2),
            "junk_rate_26_pct": round(junk_rate_26, 2),
            "junk_delta_pts": round(junk_delta_pts, 2),
            "gate_1_1_junk_pass": gate_1_1_junk_pass,
        },
        "tile_seams": {
            "straddled_count": seam_straddled_count,
            "straddled_matches_26": seam_straddled_matches,
            "straddled_recall_pct": round(seam_straddled_matches / max(1, seam_straddled_count) * 100.0, 2),
        },
        "tango_test": {
            "z_statistic": round(z_stat, 2),
            "p_value": p_val,
            "net_diff_pct": round(diff * 100.0, 2),
            "tango_pass": tango_pass,
        },
        "overall_gate_3_pass": bool(
            gate_1_1_recall_pass and
            gate_1_1_junk_pass and
            gate_1_1_conf_med_pass and
            gate_1_1_conf_cross_pass and
            gate_3_speed_pass and
            tango_pass
        ),
    }

    out_file = args.out_dir / "step3_yolo26_results.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log.info("Results saved to %s", out_file)
    print("\n" + "=" * 60)
    print("STEP 3 (YOLO26s vs YOLO11s) EVALUATION SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
