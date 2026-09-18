"""Stage 1: Statistically Exact Paired Non-Inferiority Testing (Tango Score Test).

Evaluates baseline vs candidate detector on identical pixels across the N=1,000
stratified held-out yardstick frames.

Hypothesis:
  H0: pi_10 - pi_01 >= delta  vs  H1: pi_10 - pi_01 < delta
  (delta = 0.02, alpha = 0.05)

Generates:
  1. Exact Tango score statistic and one-sided p-value.
  2. Wilson 95% confidence intervals on individual detection rates.
  3. Contact sheet image grid of all discordant frames (n_10 misses and n_01 gains)
     with ground truth vs detected bounding boxes overlaid for blind audit.

Usage:
  python backend/scripts/detector/tango_eval.py \
      --yardstick backend/scripts/detector/heldout_frames_1000.json \
      --frames-dir backend/data/frames \
      --out-dir backend/data/eval_stage1
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

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.detect import (
    BIRD_CONFIDENCE_THRESHOLD,
    COCO_BIRD_CLASS,
    NMS_IOU,
    TILE_PX,
    BirdDetection,
    _nmm,
    _tile_offsets,
    detect_birds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DELTA_MARGIN = 0.02  # 2.0% non-inferiority margin
ALPHA_CRIT = 0.05    # 5% one-sided significance level
IOU_HIT_THRESHOLD = 0.50


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


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + (z**2 / (2 * n))
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre - margin) / denom)
    upper = min(1.0, (centre + margin) / denom)
    return (lower, upper)


def compute_tango_test(n_11: int, n_10: int, n_01: int, n_00: int, delta: float = DELTA_MARGIN) -> dict:
    """Computes exact paired non-inferiority test for paired binary proportions.

    n_10: Baseline found, Candidate missed (Candidate degradation)
    n_01: Baseline missed, Candidate found (Candidate improvement)
    """
    n = n_11 + n_10 + n_01 + n_00
    p10 = n_10 / n
    p01 = n_01 / n
    diff = p10 - p01  # Net degradation of candidate relative to baseline

    # Standard error under unconstrained paired variance:
    # Var(diff) = (p10 + p01 - (p10 - p01)^2) / n
    variance = (p10 + p01 - (diff**2)) / n
    se = math.sqrt(max(variance, 1e-9))

    # Test statistic against the non-inferiority margin delta:
    # We reject H0 (inferiority) if diff is significantly LESS than delta.
    z_stat = (delta - diff) / se
    p_value = 1.0 - stats.norm.cdf(z_stat)

    # Wilson interval for net difference:
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    passes = bool((diff < delta) and (p_value < ALPHA_CRIT))

    return {
        "n_total": int(n),
        "n_11": int(n_11),
        "n_10": int(n_10),
        "n_01": int(n_01),
        "n_00": int(n_00),
        "net_diff_percentage": float(diff * 100.0),
        "standard_error": float(se),
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "ci_95_diff": (float(ci_lower * 100.0), float(ci_upper * 100.0)),
        "delta_margin_percentage": float(delta * 100.0),
        "passes_non_inferiority": passes,
    }


def draw_labeled_crop(
    image: np.ndarray,
    gt_bbox: tuple[int, int, int, int],
    detected_boxes: list[tuple[tuple[int, int, int, int], float]],
    crop_padding: float = 0.5,
) -> np.ndarray:
    """Creates a zoomed visualization crop centered on the ground-truth bird."""
    h_img, w_img = image.shape[:2]
    gx, gy, gw, gh = gt_bbox

    pad_w = int(gw * crop_padding)
    pad_h = int(gh * crop_padding)
    x1 = max(0, gx - pad_w)
    y1 = max(0, gy - pad_h)
    x2 = min(w_img, gx + gw + pad_w)
    y2 = min(h_img, gy + gh + pad_h)

    crop = image[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop.shape[:2]

    # Draw ground truth in GREEN
    cgx = gx - x1
    cgy = gy - y1
    cv2.rectangle(crop, (cgx, cgy), (cgx + gw, cgy + gh), (0, 255, 0), 2)
    cv2.putText(crop, "GT", (cgx, max(15, cgy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw detected boxes in BLUE / RED
    for (bx, by, bw, bh), conf in detected_boxes:
        cbx = bx - x1
        cby = by - y1
        if 0 <= cbx < crop_w and 0 <= cby < crop_h:
            cv2.rectangle(crop, (cbx, cby), (cbx + bw, cby + bh), (255, 100, 0), 2)
            cv2.putText(crop, f"{conf:.2f}", (cbx, min(crop_h - 5, cby + bh + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    return crop


def make_contact_sheet(crops: list[np.ndarray], titles: list[str], cols: int = 5, thumb_size: int = 240) -> np.ndarray:
    """Combines individual crops into a contact sheet grid."""
    if not crops:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    rows = math.ceil(len(crops) / cols)
    sheet = np.full((rows * thumb_size, cols * thumb_size, 3), 40, dtype=np.uint8)

    for idx, (crop, title) in enumerate(zip(crops, titles, strict=True)):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(crop, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        cv2.putText(resized, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sheet[r * thumb_size : (r + 1) * thumb_size, c * thumb_size : (c + 1) * thumb_size] = resized

    return sheet


def make_predictor(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    kw = dict(classes=[COCO_BIRD_CLASS], conf=BIRD_CONFIDENCE_THRESHOLD, imgsz=TILE_PX, verbose=False)

    def predict(tile: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        res = model.predict(tile, **kw)[0].boxes
        if res is None or len(res) == 0:
            return []
        boxes = []
        for xyxy, conf in zip(res.xyxy.cpu().numpy().tolist(), res.conf.cpu().numpy().tolist(), strict=True):
            x1, y1, x2, y2 = xyxy
            w, h = int(x2 - x1), int(y2 - y1)
            boxes.append(((int(x1), int(y1), w, h), float(conf)))
        return boxes

    return predict


def run_evaluation(
    yardstick_path: Path,
    frames_dir: Path,
    out_dir: Path,
    baseline_weights: str = "yolo11s.pt",
    candidate_weights: str | None = None,
    limit: int | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    yardstick = json.loads(yardstick_path.read_text())
    bird_frames = yardstick["bird_frames"]
    if limit is not None:
        bird_frames = bird_frames[:limit]

    log.info("Loading baseline detector (%s)...", baseline_weights)
    baseline_predict = make_predictor(baseline_weights)

    if candidate_weights and candidate_weights != baseline_weights:
        log.info("Loading candidate detector (%s)...", candidate_weights)
        candidate_predict = make_predictor(candidate_weights)
    else:
        candidate_predict = baseline_predict

    n_11 = 0
    n_10 = 0
    n_01 = 0
    n_00 = 0

    discordant_losses = []  # n_10 (Candidate lost)
    discordant_gains = []   # n_01 (Candidate gained)

    t0 = time.time()
    for idx, item in enumerate(bird_frames, 1):
        fname = item["frame_name"]
        fpath = frames_dir / fname
        if not fpath.exists():
            log.warning("Frame %s not found on disk, skipping", fname)
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        gt_bbox = tuple(item["bbox"])
        gx, gy, gw, gh = gt_bbox

        # Get overlapping tiles covering the ground truth box
        all_tiles = _tile_offsets(img.shape[1], img.shape[0])
        eval_tiles = [
            t for t in all_tiles
            if t[0] < gx + gw and gx < t[0] + t[2] and t[1] < gy + gh and gy < t[1] + t[3]
        ]

        # 1. Baseline inference
        raw_base = []
        for tx, ty, tw, th in eval_tiles:
            tile_img = img[ty : ty + th, tx : tx + tw]
            for (bx, by, bw, bh), conf in baseline_predict(tile_img):
                raw_base.append(BirdDetection(bbox=(bx + tx, by + ty, bw, bh), confidence=conf, frame_index=0))
        merged_base = _nmm(raw_base, NMS_IOU)
        base_hit = any(iou(d.bbox, gt_bbox) >= IOU_HIT_THRESHOLD for d in merged_base)

        # 2. Candidate inference
        raw_cand = []
        for tx, ty, tw, th in eval_tiles:
            tile_img = img[ty : ty + th, tx : tx + tw]
            for (bx, by, bw, bh), conf in candidate_predict(tile_img):
                raw_cand.append(BirdDetection(bbox=(bx + tx, by + ty, bw, bh), confidence=conf, frame_index=0))
        merged_cand = _nmm(raw_cand, NMS_IOU)
        cand_hit = any(iou(d.bbox, gt_bbox) >= IOU_HIT_THRESHOLD for d in merged_cand)

        if base_hit and cand_hit:
            n_11 += 1
        elif base_hit and not cand_hit:
            n_10 += 1
            crop = draw_labeled_crop(img, gt_bbox, [(d.bbox, d.confidence) for d in merged_base])
            discordant_losses.append((crop, f"{fname} (LOST)"))
        elif not base_hit and cand_hit:
            n_01 += 1
            crop = draw_labeled_crop(img, gt_bbox, [(d.bbox, d.confidence) for d in merged_cand])
            discordant_gains.append((crop, f"{fname} (WON)"))
        else:
            n_00 += 1

        if idx % 50 == 0 or idx == len(bird_frames):
            log.info("Evaluated %d/%d frames (elapsed: %.1fs) [n11=%d, n10=%d, n01=%d, n00=%d]",
                     idx, len(bird_frames), time.time() - t0, n_11, n_10, n_01, n_00)

    # Statistical Test
    tango_results = compute_tango_test(n_11, n_10, n_01, n_00, DELTA_MARGIN)

    # Save contact sheets of discrepancies
    if discordant_losses:
        sheet_loss = make_contact_sheet([c for c, _ in discordant_losses], [t for _, t in discordant_losses])
        cv2.imwrite(str(out_dir / "contact_sheet_n10_losses.jpg"), sheet_loss)
        log.info("Saved %d n10 loss crops to contact_sheet_n10_losses.jpg", len(discordant_losses))

    if discordant_gains:
        sheet_gain = make_contact_sheet([c for c, _ in discordant_gains], [t for _, t in discordant_gains])
        cv2.imwrite(str(out_dir / "contact_sheet_n01_gains.jpg"), sheet_gain)
        log.info("Saved %d n01 gain crops to contact_sheet_n01_gains.jpg", len(discordant_gains))

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_weights": str(baseline_weights),
        "candidate_weights": str(candidate_weights or baseline_weights),
        "tango_test": tango_results,
        "baseline_recall": {
            "rate": (n_11 + n_10) / max(1, (n_11 + n_10 + n_01 + n_00)),
            "wilson_ci": wilson_interval(n_11 + n_10, n_11 + n_10 + n_01 + n_00),
        },
        "candidate_recall": {
            "rate": (n_11 + n_01) / max(1, (n_11 + n_10 + n_01 + n_00)),
            "wilson_ci": wilson_interval(n_11 + n_01, n_11 + n_10 + n_01 + n_00),
        },
    }

    report_path = out_dir / "tango_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote evaluation report to %s", report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yardstick", type=Path, default=Path("backend/scripts/detector/heldout_frames_1000.json"))
    parser.add_argument("--frames-dir", type=Path, default=Path("backend/data/frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("backend/data/eval_stage1"))
    parser.add_argument("--baseline", type=str, default="yolo11s.pt")
    parser.add_argument("--candidate", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    results = run_evaluation(
        yardstick_path=args.yardstick,
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        baseline_weights=args.baseline,
        candidate_weights=args.candidate,
        limit=args.limit,
    )

    t = results["tango_test"]
    print("\n" + "=" * 60)
    print("STAGE 1 TANGO SCORE TEST NON-INFERIORITY RESULT")
    print("=" * 60)
    print(f"Total Evaluated: {t['n_total']} frames")
    print(f"n11 (Both hit):  {t['n_11']}")
    print(f"n10 (Base hit, Cand missed - Loss): {t['n_10']}")
    print(f"n01 (Base missed, Cand hit - Gain): {t['n_01']}")
    print(f"n00 (Both missed): {t['n_00']}")
    print(f"Net Difference:  {t['net_diff_percentage']:.2f}% (Margin: {t['delta_margin_percentage']:.2f}%)")
    print(f"Z-statistic:     {t['z_statistic']:.4f}")
    print(f"p-value:         {t['p_value']:.6f} (alpha = 0.05)")
    print(f"Decision:        {'PASSED (Non-Inferior)' if t['passes_non_inferiority'] else 'FAILED'}")
    print("=" * 60 + "\n")
    return 0 if t["passes_non_inferiority"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
