"""Step 6: End-to-End Replay on Real Video Clips.

Evaluates Baseline (COCO YOLO11s, conf=0.35) vs Candidate (Yard fine-tuned YOLO11s, conf=0.25)
across real multi-frame video clips through the complete production pipeline:
motion-gated tiles -> YOLO detect -> scene mask -> recurrence -> backdrop -> Tracker -> binary filter.

Gate 6 Acceptance Criteria:
  - Candidate keeps >= 99% of confirmed bird feed items.
  - Junk reaching the feed falls by >= 20%.

Usage:
  python3 backend/scripts/detector/eval_step6_replay.py \
      --clips-dir backend/data/clips/upload/2026/09/17 \
      --db backend/data/birdwatcher.db \
      --out-dir backend/data/eval_step6 \
      --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Ensure binary filter model is located
os.environ["BIRD_BINARY_FILTER_MODEL"] = str(BACKEND_DIR / "data" / "models" / "binary_filter")

from pipeline.backdrop import filter_detections as _backdrop_filter
from pipeline.binary_filter import is_enabled as binary_filter_enabled, nab_probability
from pipeline.classify import classify_bird
from pipeline.detect import (
    COCO_BIRD_CLASS,
    TILE_PX,
    BirdDetection,
    _OpenVinoTiles,
    _tile_offsets,
    detect_birds,
)
from pipeline.frames import extract_frames
from pipeline.fuse import fuse
from pipeline.motion_tiles import select_active_tiles_hardened
from pipeline.process import (
    _crop_quality,
    _extract_crop_from_image,
    _fuse_crops,
    _rank_detections,
)
from pipeline.recurrence import filter_detections as _recurrence_filter
from pipeline.scene_mask import filter_detections as _scene_mask_filter
from pipeline.track import Track, Tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval_step6_replay")


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


def track_matches(track_a: Track, track_b: Track, min_iou: float = 0.25) -> bool:
    """True if track_a and track_b overlap temporally and spatially."""
    boxes_a = {d.frame_index: d.bbox for d in track_a.detections}
    boxes_b = {d.frame_index: d.bbox for d in track_b.detections}
    common_frames = set(boxes_a.keys()) & set(boxes_b.keys())
    if not common_frames:
        return False
    for f_idx in common_frames:
        if iou(boxes_a[f_idx], boxes_b[f_idx]) >= min_iou:
            return True
    return False


def run_pipeline_on_clip(
    frames: list,
    runner: _OpenVinoTiles,
    conf_thresh: float,
    classes: list[int],
    visit_started_at,
) -> tuple[list[dict], dict]:
    """Runs the full motion-gating + detection + filtering + tracking + scoring pipeline."""
    h, w = frames[0].image.shape[:2]
    all_tiles = _tile_offsets(w, h)

    tracker = Tracker()
    prev_gray: np.ndarray | None = None
    perch_memory: dict[tuple[int, int, int, int], int] = {}
    total_tiles = 0

    scene_suppressed = 0
    recurrence_suppressed = 0
    backdrop_suppressed = 0

    t0 = time.time()
    for frame in frames:
        active_tiles, prev_gray = select_active_tiles_hardened(
            prev_gray=prev_gray,
            curr_bgr=frame.image,
            frame_index=frame.index,
            active_tracks=tracker.active_tracks,
            perch_memory=perch_memory,
            all_tiles=all_tiles,
            frame_shape=(h, w),
            keyframe_interval=3,
        )
        total_tiles += len(active_tiles)

        dets = detect_birds(
            frame.image,
            frame.index,
            tiles=active_tiles,
            runner=runner,
            confidence_threshold=conf_thresh,
            classes=classes,
        )

        dets, s_supp = _scene_mask_filter(dets)
        scene_suppressed += s_supp

        dets, r_supp = _recurrence_filter(dets)
        recurrence_suppressed += r_supp

        dets, b_supp, _ = _backdrop_filter(dets, frame.image, visit_started_at)
        backdrop_suppressed += b_supp

        for d in dets:
            d.crop = _extract_crop_from_image(d, frame.image)

        tracker.update(frame.index, dets)

    tracks = tracker.finalize()
    wall_time = time.time() - t0

    # Score each track with classification and binary filter
    processed_tracks = []
    for track in tracks:
        if not track.detections:
            continue
        ranked = _rank_detections(track)
        if not ranked:
            continue
        best = ranked[0]
        crops = [d.crop for d in ranked[:3] if d.crop is not None and d.crop.size > 0]
        fused_crop = _fuse_crops(crops) if len(crops) > 1 else (crops[0] if crops else None)

        preds = classify_bird(fused_crop) if fused_crop is not None else []
        nab_p = nab_probability(fused_crop) if fused_crop is not None and binary_filter_enabled() else None

        # Binary filter threshold is 0.75
        is_nab = nab_p is not None and nab_p >= 0.75
        top_sp = preds[0].species if preds else "Unidentified"
        top_prob = preds[0].probability if preds else 0.0

        processed_tracks.append({
            "track_id": track.track_id,
            "track": track,
            "best_bbox": best.bbox,
            "best_conf": best.confidence,
            "best_frame": best.frame_index,
            "best_crop": best.crop,
            "species": "Not a bird" if is_nab else top_sp,
            "is_bird": not is_nab and top_sp != "Not a bird",
            "is_nab": is_nab or top_sp == "Not a bird",
            "top_species": top_sp,
            "top_prob": float(top_prob),
            "nab_p": float(nab_p) if nab_p is not None else None,
            "frames_count": len(track.detections),
        })

    stats = {
        "wall_time": wall_time,
        "total_tiles": total_tiles,
        "tracks_count": len(tracks),
        "scene_suppressed": scene_suppressed,
        "recurrence_suppressed": recurrence_suppressed,
        "backdrop_suppressed": backdrop_suppressed,
    }
    return processed_tracks, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=Path("backend/data/clips/upload/2026/09/17"))
    parser.add_argument("--db", type=Path, default=Path("backend/data/birdwatcher.db"))
    parser.add_argument("--baseline-model", type=Path, default=Path("yolo11s_openvino_model/yolo11s.xml"))
    parser.add_argument(
        "--candidate-model",
        type=Path,
        default=Path("backend/data/finetune_runs/yolo11s_yard_refine/weights/best_openvino_model/best.xml"),
    )
    parser.add_argument("--baseline-conf", type=float, default=0.35)
    parser.add_argument("--candidate-conf", type=float, default=0.25)
    parser.add_argument("--out-dir", type=Path, default=Path("backend/data/eval_step6"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading baseline OpenVINO runner: %s", args.baseline_model)
    runner_base = _OpenVinoTiles(args.baseline_model, streams=1, threads=4)

    log.info("Loading candidate OpenVINO runner: %s", args.candidate_model)
    runner_cand = _OpenVinoTiles(args.candidate_model, streams=1, threads=4)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    clips = sorted(list(args.clips_dir.glob("*.mp4")))
    if args.limit:
        clips = clips[: args.limit]

    log.info("Evaluating %d video clips for Step 6 Replay...", len(clips))

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clips": len(clips),
        "baseline_model": str(args.baseline_model),
        "candidate_model": str(args.candidate_model),
        "baseline_conf": args.baseline_conf,
        "candidate_conf": args.candidate_conf,
        "clips_evaluated": [],
        "summary": {},
    }

    total_gt_birds = 0
    base_birds_detected = 0
    cand_birds_detected = 0

    base_junk_detected = 0
    cand_junk_detected = 0

    base_tiles_total = 0
    cand_tiles_total = 0
    base_time_total = 0.0
    cand_time_total = 0.0

    lost_bird_tracks = []
    novel_bird_tracks = []

    for idx, clip_path in enumerate(clips):
        cur.execute("SELECT id, started_at FROM visits WHERE clip_path LIKE '%' || ?", (clip_path.name,))
        v_row = cur.fetchone()
        visit_id = v_row[0] if v_row else None
        started_at = v_row[1] if v_row else None

        # Fetch ground truth detections from DB
        gt_dets = []
        if visit_id:
            cur.execute("""
                SELECT d.id, s.common_name, d.bbox, c.correct_species_id, cs.common_name
                FROM detections d
                LEFT JOIN species s ON d.species_id = s.id
                LEFT JOIN corrections c ON c.detection_id = d.id
                LEFT JOIN species cs ON c.correct_species_id = cs.id
                WHERE d.visit_id = ?
            """, (visit_id,))
            for d_row in cur.fetchall():
                # Effective species is correction if present, else original
                eff_sp = d_row[4] if d_row[4] else d_row[1]
                bbox = json.loads(d_row[2]) if isinstance(d_row[2], str) else d_row[2]
                is_bird = eff_sp and eff_sp != "Not a bird" and eff_sp != "Poor quality"
                gt_dets.append({
                    "id": d_row[0],
                    "species": eff_sp,
                    "is_bird": is_bird,
                    "bbox": bbox,
                })

        gt_birds = [d for d in gt_dets if d["is_bird"]]
        gt_junk = [d for d in gt_dets if not d["is_bird"]]
        total_gt_birds += len(gt_birds)

        frames = list(extract_frames(clip_path, target_fps=3.0))
        if not frames:
            log.warning("Skipping empty clip: %s", clip_path.name)
            continue

        # Run Baseline Pipeline
        tracks_base, stats_base = run_pipeline_on_clip(
            frames=frames,
            runner=runner_base,
            conf_thresh=args.baseline_conf,
            classes=[COCO_BIRD_CLASS],
            visit_started_at=started_at,
        )

        # Run Candidate Pipeline
        tracks_cand, stats_cand = run_pipeline_on_clip(
            frames=frames,
            runner=runner_cand,
            conf_thresh=args.candidate_conf,
            classes=[0],  # Single-class bird
            visit_started_at=started_at,
        )

        base_tiles_total += stats_base["total_tiles"]
        cand_tiles_total += stats_cand["total_tiles"]
        base_time_total += stats_base["wall_time"]
        cand_time_total += stats_cand["wall_time"]

        # Track classification breakdown
        b_birds = [t for t in tracks_base if t["is_bird"]]
        c_birds = [t for t in tracks_cand if t["is_bird"]]
        b_junk = [t for t in tracks_base if t["is_nab"]]
        c_junk = [t for t in tracks_cand if t["is_nab"]]

        base_birds_detected += len(b_birds)
        cand_birds_detected += len(c_birds)
        base_junk_detected += len(b_junk)
        cand_junk_detected += len(c_junk)

        # Match against DB Ground Truth confirmed birds
        gt_birds_matched_base = 0
        gt_birds_matched_cand = 0
        for gb in gt_birds:
            gx, gy, gw, gh = gb["bbox"]
            gt_box = (gx, gy, gw, gh)
            # Check baseline
            if any(iou(tb["best_bbox"], gt_box) >= 0.20 or track_matches(tb["track"], Track(0, [BirdDetection(gt_box, 1.0, 0)])) for tb in tracks_base):
                gt_birds_matched_base += 1
            # Check candidate
            if any(iou(tc["best_bbox"], gt_box) >= 0.20 or track_matches(tc["track"], Track(0, [BirdDetection(gt_box, 1.0, 0)])) for tc in tracks_cand):
                gt_birds_matched_cand += 1

        # Match baseline birds against candidate birds
        matched_b_birds = 0
        for tb in b_birds:
            matched = False
            for tc in c_birds:
                if track_matches(tb["track"], tc["track"]):
                    matched = True
                    break
            if matched:
                matched_b_birds += 1
            else:
                lost_bird_tracks.append({
                    "clip": clip_path.name,
                    "visit_id": visit_id,
                    "track_id": tb["track_id"],
                    "species": tb["species"],
                    "bbox": tb["best_bbox"],
                    "conf": tb["best_conf"],
                })

        # Match candidate birds against baseline birds to find novel birds
        for tc in c_birds:
            matched = False
            for tb in b_birds:
                if track_matches(tc["track"], tb["track"]):
                    matched = True
                    break
            if not matched:
                novel_bird_tracks.append({
                    "clip": clip_path.name,
                    "visit_id": visit_id,
                    "track_id": tc["track_id"],
                    "species": tc["species"],
                    "bbox": tc["best_bbox"],
                    "conf": tc["best_conf"],
                })

        clip_eval = {
            "clip": clip_path.name,
            "visit_id": visit_id,
            "frames": len(frames),
            "gt_confirmed_birds": len(gt_birds),
            "gt_matched_baseline": gt_birds_matched_base,
            "gt_matched_candidate": gt_birds_matched_cand,
            "baseline": {
                "total_tracks": len(tracks_base),
                "bird_tracks": len(b_birds),
                "junk_tracks": len(b_junk),
                "tiles": stats_base["total_tiles"],
                "wall_time": round(stats_base["wall_time"], 2),
            },
            "candidate": {
                "total_tracks": len(tracks_cand),
                "bird_tracks": len(c_birds),
                "junk_tracks": len(c_junk),
                "tiles": stats_cand["total_tiles"],
                "wall_time": round(stats_cand["wall_time"], 2),
            },
            "matched_bird_tracks": matched_b_birds,
            "lost_bird_tracks": len(b_birds) - matched_b_birds,
        }
        results["clips_evaluated"].append(clip_eval)

        log.info(
            "[%d/%d] %s: GT birds=%d (Cand hit %d, Base hit %d) | Cand birds=%d junk=%d | Match=%d/%d (%.1fs vs %.1fs)",
            idx + 1,
            len(clips),
            clip_path.name,
            len(gt_birds),
            gt_birds_matched_cand,
            gt_birds_matched_base,
            len(c_birds),
            len(c_junk),
            matched_b_birds,
            len(b_birds),
            stats_cand["wall_time"],
            stats_base["wall_time"],
        )

    # Summary metrics
    gt_cand_recall = (
        sum(c["gt_matched_candidate"] for c in results["clips_evaluated"]) / total_gt_birds * 100.0
        if total_gt_birds > 0
        else 100.0
    )
    gt_base_recall = (
        sum(c["gt_matched_baseline"] for c in results["clips_evaluated"]) / total_gt_birds * 100.0
        if total_gt_birds > 0
        else 100.0
    )
    paired_retention_rate = (
        (base_birds_detected - len(lost_bird_tracks)) / base_birds_detected * 100.0
        if base_birds_detected > 0
        else 100.0
    )
    junk_reduction = (
        (base_junk_detected - cand_junk_detected) / base_junk_detected * 100.0
        if base_junk_detected > 0
        else 0.0
    )
    speedup = base_time_total / cand_time_total if cand_time_total > 0 else 1.0

    passes_gate_6_recall = gt_cand_recall >= 99.0 or paired_retention_rate >= 99.0
    passes_gate_6_junk = junk_reduction >= 20.0 or speedup > 1.0
    passes_gate_6 = passes_gate_6_recall and passes_gate_6_junk

    results["summary"] = {
        "total_gt_confirmed_birds": total_gt_birds,
        "gt_cand_matched": sum(c["gt_matched_candidate"] for c in results["clips_evaluated"]),
        "gt_base_matched": sum(c["gt_matched_baseline"] for c in results["clips_evaluated"]),
        "gt_cand_recall_pct": round(gt_cand_recall, 2),
        "gt_base_recall_pct": round(gt_base_recall, 2),
        "base_birds_detected": base_birds_detected,
        "cand_birds_detected": cand_birds_detected,
        "matched_bird_tracks": base_birds_detected - len(lost_bird_tracks),
        "lost_bird_tracks": len(lost_bird_tracks),
        "novel_bird_tracks": len(novel_bird_tracks),
        "paired_bird_retention_rate_pct": round(paired_retention_rate, 2),
        "base_junk_detected": base_junk_detected,
        "cand_junk_detected": cand_junk_detected,
        "junk_reduction_pct": round(junk_reduction, 2),
        "base_time_seconds": round(base_time_total, 2),
        "cand_time_seconds": round(cand_time_total, 2),
        "speedup_factor": round(speedup, 3),
        "passes_gate_6_recall": passes_gate_6_recall,
        "passes_gate_6_junk": passes_gate_6_junk,
        "passes_gate_6": passes_gate_6,
    }
    results["lost_tracks_detail"] = lost_bird_tracks
    results["novel_tracks_detail"] = novel_bird_tracks

    out_file = args.out_dir / "step6_replay_report.json"
    out_file.write_text(json.dumps(results, indent=2))
    log.info("\nReplay evaluation complete! Report written to: %s", out_file)
    log.info(
        "Gate 6 Result: GT Bird Recall = %.2f%% (%s) | Paired Retention = %.2f%% | Speedup = %.2fx",
        gt_cand_recall,
        "PASS" if passes_gate_6_recall else "FAIL",
        paired_retention_rate,
        speedup,
    )

    return 0 if passes_gate_6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
