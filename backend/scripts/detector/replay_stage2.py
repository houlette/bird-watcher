"""Stage 2: Offline 1,000-Visit Video Replay on Development Mac.

Evaluates Baseline (brute-force 15 tiles/frame) vs Candidate (hardened Phase 0
motion-gated tiles) on full video clips (.mp4) using identical frame decodes and Tracker logic.

Gate 2 Acceptance Criterion:
  Allowed misses k <= 12 tracks out of 1,000 bird visits (>= 98.8% net recall).
  Exact Type I error <= 3.85%, Statistical Power = 93.6%.

Usage:
  python3 backend/scripts/detector/replay_stage2.py \
      --clips-dir backend/data/clips \
      --out-dir backend/data/eval_stage2 \
      --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

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
from pipeline.frames import extract_frames
from pipeline.motion_tiles import select_active_tiles_hardened
from pipeline.track import Track, Tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replay_stage2")

IOU_MATCH_THRESHOLD = 0.40
MAX_MISSED_GATE = 12  # Out of 1,000 visits


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


def track_matches(track_a: Track, track_b: Track, min_iou: float = IOU_MATCH_THRESHOLD) -> bool:
    """Returns True if track_a and track_b overlap temporally and spatially."""
    boxes_a = {d.frame_index: d.bbox for d in track_a.detections}
    boxes_b = {d.frame_index: d.bbox for d in track_b.detections}
    common_frames = set(boxes_a.keys()) & set(boxes_b.keys())
    if not common_frames:
        return False
    for f_idx in common_frames:
        if iou(boxes_a[f_idx], boxes_b[f_idx]) >= min_iou:
            return True
    return False


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


def replay_clip(
    clip_path: Path,
) -> dict:
    """Replays a single video clip through both Baseline and Candidate pipelines."""
    # 1. Decode all frames once so both pipelines see identical pixels
    frames = list(extract_frames(clip_path, target_fps=3.0))
    if not frames:
        return {"clip": clip_path.name, "error": "no_frames", "tracks_base": 0, "tracks_cand": 0}

    h, w = frames[0].image.shape[:2]
    all_tiles = _tile_offsets(w, h)

    # --- Run Baseline (Full 15-tile brute force) ---
    tracker_base = Tracker()
    tiles_base_count = 0
    t0_base = time.time()
    for frame in frames:
        dets = detect_birds(frame.image, frame.index, tiles=all_tiles)
        tiles_base_count += len(all_tiles)
        tracker_base.update(frame.index, dets)
    tracks_base = tracker_base.finalize()
    time_base = time.time() - t0_base

    # --- Run Candidate (Hardened Phase 0 Motion-gated tiles) ---
    tracker_cand = Tracker()
    tiles_cand_count = 0
    prev_gray: np.ndarray | None = None
    perch_memory: dict[tuple[int, int, int, int], int] = {}
    t0_cand = time.time()
    for frame in frames:
        active_tiles, prev_gray = select_active_tiles_hardened(
            prev_gray=prev_gray,
            curr_bgr=frame.image,
            frame_index=frame.index,
            active_tracks=tracker_cand.active_tracks,
            perch_memory=perch_memory,
            all_tiles=all_tiles,
            frame_shape=(h, w),
            keyframe_interval=3,
        )
        dets = detect_birds(frame.image, frame.index, tiles=active_tiles)
        tiles_cand_count += len(active_tiles)
        tracker_cand.update(frame.index, dets)
    tracks_cand = tracker_cand.finalize()
    time_cand = time.time() - t0_cand

    # Match tracks
    matched_base = set()
    matched_cand = set()
    for idx_b, tb in enumerate(tracks_base):
        for idx_c, tc in enumerate(tracks_cand):
            if idx_c in matched_cand:
                continue
            if track_matches(tb, tc):
                matched_base.add(idx_b)
                matched_cand.add(idx_c)
                break

    unmatched_base = [tb for idx_b, tb in enumerate(tracks_base) if idx_b not in matched_base]
    unmatched_cand = [tc for idx_c, tc in enumerate(tracks_cand) if idx_c not in matched_cand]

    # Capture visual crops for any lost tracks
    lost_crops = []
    frames_by_idx = {f.index: f.image for f in frames}
    for tb in unmatched_base:
        if not tb.detections:
            continue
        best_d = tb.best_detection
        img = frames_by_idx.get(best_d.frame_index)
        if img is not None:
            bx, by, bw, bh = best_d.bbox
            crop = img[by : by + bh, bx : bx + bw]
            if crop.size > 0:
                lost_crops.append((crop, f"{clip_path.stem} t{tb.track_id} (LOST)"))

    return {
        "clip": clip_path.name,
        "frames_count": len(frames),
        "tracks_base_count": len(tracks_base),
        "tracks_cand_count": len(tracks_cand),
        "matched_count": len(matched_base),
        "lost_tracks_count": len(unmatched_base),
        "gained_tracks_count": len(unmatched_cand),
        "tiles_base": tiles_base_count,
        "tiles_cand": tiles_cand_count,
        "time_base": time_base,
        "time_cand": time_cand,
        "lost_crops": lost_crops,
    }


def run_stage2_replay(
    clips_dir: Path,
    out_dir: Path,
    limit: int | None = None,
) -> dict:
    """Executes Stage 2 video replay across all clips in clips_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_clips = sorted(list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.webm")))
    if not all_clips:
        # Check subdirectories (e.g. upload/)
        all_clips = sorted(list(clips_dir.glob("**/*.mp4")) + list(clips_dir.glob("**/*.webm")))

    if limit is not None:
        all_clips = all_clips[:limit]

    log.info("Found %d clips to replay in %s", len(all_clips), clips_dir)
    if not all_clips:
        log.warning("No clips found in %s", clips_dir)
        return {"error": "no_clips"}

    total_clips = len(all_clips)
    total_base_tracks = 0
    total_cand_tracks = 0
    total_matched_tracks = 0
    total_lost_tracks = 0
    total_gained_tracks = 0
    total_tiles_base = 0
    total_tiles_cand = 0
    total_time_base = 0.0
    total_time_cand = 0.0

    all_lost_crops = []
    clip_summaries = []

    t_start = time.time()
    for idx, clip in enumerate(all_clips, 1):
        res = replay_clip(clip)
        if "error" in res:
            continue

        total_base_tracks += res["tracks_base_count"]
        total_cand_tracks += res["tracks_cand_count"]
        total_matched_tracks += res["matched_count"]
        total_lost_tracks += res["lost_tracks_count"]
        total_gained_tracks += res["gained_tracks_count"]
        total_tiles_base += res["tiles_base"]
        total_tiles_cand += res["tiles_cand"]
        total_time_base += res["time_base"]
        total_time_cand += res["time_cand"]

        all_lost_crops.extend(res["lost_crops"])
        clip_summaries.append({k: v for k, v in res.items() if k != "lost_crops"})

        if idx % 10 == 0 or idx == total_clips:
            savings = (1.0 - total_tiles_cand / max(1, total_tiles_base)) * 100.0
            log.info(
                "[%d/%d clips] Base tracks: %d, Cand tracks: %d, Lost: %d, Gained: %d | "
                "Tiles: %d base vs %d cand (-%.1f%%) | Elapsed: %.1fs",
                idx,
                total_clips,
                total_base_tracks,
                total_cand_tracks,
                total_lost_tracks,
                total_gained_tracks,
                total_tiles_base,
                total_tiles_cand,
                savings,
                time.time() - t_start,
            )

    tile_reduction_pct = (1.0 - total_tiles_cand / max(1, total_tiles_base)) * 100.0 if total_tiles_base else 0.0
    speedup = total_tiles_base / max(1, total_tiles_cand) if total_tiles_cand else 1.0

    # Save lost crops contact sheet
    if all_lost_crops:
        sheet = make_contact_sheet([c for c, _ in all_lost_crops], [t for _, t in all_lost_crops])
        cv2.imwrite(str(out_dir / "contact_sheet_stage2_losses.jpg"), sheet)
        log.info("Saved %d lost track crops to contact_sheet_stage2_losses.jpg", len(all_lost_crops))

    # Determine Gate result
    gate_passed = total_lost_tracks <= MAX_MISSED_GATE

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clips_evaluated": total_clips,
        "tracks": {
            "baseline": total_base_tracks,
            "candidate": total_cand_tracks,
            "matched": total_matched_tracks,
            "lost (n10)": total_lost_tracks,
            "gained (n01)": total_gained_tracks,
            "track_recall_rate": total_matched_tracks / max(1, total_base_tracks),
        },
        "compute": {
            "total_tiles_baseline": total_tiles_base,
            "total_tiles_candidate": total_tiles_cand,
            "tile_reduction_percentage": tile_reduction_pct,
            "compute_speedup_factor": speedup,
            "total_wall_time_base_seconds": total_time_base,
            "total_wall_time_cand_seconds": total_time_cand,
        },
        "gate_2": {
            "max_allowed_misses": MAX_MISSED_GATE,
            "actual_misses": total_lost_tracks,
            "passed": gate_passed,
        },
        "clips": clip_summaries,
    }

    report_path = out_dir / "stage2_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Wrote Stage 2 report to %s", report_path)

    print("\n" + "=" * 60)
    print("STAGE 2 VIDEO REPLAY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Clips Evaluated:      {total_clips}")
    print(f"Baseline Tracks Found:     {total_base_tracks}")
    print(f"Candidate Tracks Found:    {total_cand_tracks}")
    print(f"Matched Tracks:            {total_matched_tracks}")
    print(f"Lost Tracks (Candidate):   {total_lost_tracks}")
    print(f"Gained Tracks (Candidate): {total_gained_tracks}")
    print(f"Track Net Recall Rate:     {report['tracks']['track_recall_rate']:.2%}")
    print("-" * 60)
    print(f"Baseline Total Tiles:      {total_tiles_base}")
    print(f"Candidate Total Tiles:     {total_tiles_cand} (-{tile_reduction_pct:.1f}%)")
    print(f"Inference Speedup Factor:  {speedup:.2f}x")
    print("-" * 60)
    print(f"Gate 2 Criterion:          Misses <= {MAX_MISSED_GATE}")
    print(f"Gate 2 Decision:           {'PASSED' if gate_passed else 'FAILED'}")
    print("=" * 60 + "\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=Path("backend/data/clips"))
    parser.add_argument("--out-dir", type=Path, default=Path("backend/data/eval_stage2"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = run_stage2_replay(args.clips_dir, args.out_dir, limit=args.limit)
    if report.get("gate_2", {}).get("passed"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
