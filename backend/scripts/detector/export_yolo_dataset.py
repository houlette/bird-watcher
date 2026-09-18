"""Export YOLO11s fine-tuning dataset from the archive (DETECTOR_PLAN.md Step 4).

Features:
  - Strictly refuses Step 2 held-out days (heldout_frames.json).
  - Evaluates tiles using production's 1024 px grid (_tile_offsets).
  - Translates and normalizes bird bboxes into tile coordinates.
  - Samples 1:1 matching clean background tiles with no objects.
  - Partitions into 80% train / 20% val split by capture day (seed=0).
  - Emits Ultralytics data.yaml and manifest.json.
  - Packages final dataset into tar.gz archive.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import (
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    SENTINEL_LABELS,
    Correction,
    Detection,
    Species,
    Visit,
)
from pipeline.detect import TILE_OVERLAP_PX, TILE_PX, _tile_offsets
from scripts.train import heldout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SPLIT_SEED = 0
VAL_DAY_SHARE = 0.20  # 20% of training days to validation


def box_intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


def main():
    parser = argparse.ArgumentParser(description="Export YOLO Fine-tuning Dataset")
    parser.add_argument("--db", type=str, default="/app/data/birdwatcher.db")
    parser.add_argument("--frames-dir", type=str, default="/app/data/frames")
    parser.add_argument("--heldout-meta", type=str, default="/app/scripts/detector/heldout_frames.json")
    parser.add_argument("--out-dir", type=str, default="/app/data/dataset_yolo11s")
    parser.add_argument("--archive-path", type=str, default="/app/data/bird_yolo11s_dataset.tar.gz")
    parser.add_argument("--limit-frames", type=int, default=None, help="Limit frames for testing")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    img_train_dir = out_dir / "images" / "train"
    img_val_dir = out_dir / "images" / "val"
    lbl_train_dir = out_dir / "labels" / "train"
    lbl_val_dir = out_dir / "labels" / "val"

    for p in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        p.mkdir(parents=True, exist_ok=True)

    # 1. Load held-out days to strictly refuse
    with open(args.heldout_meta) as f:
        heldout_meta = json.load(f)
    heldout_days = set(heldout_meta["pre_june_drawn_days"] + heldout_meta["binary_heldout_days"])
    log.info("Refusing %d held-out capture days", len(heldout_days))

    engine = create_engine(f"sqlite:///{args.db}")
    with Session(engine) as db:
        user_labels = heldout.latest_labels(db, sources=heldout.USER_SOURCES)
        tz = ZoneInfo("America/New_York")
        sentinel_ids = {s.id for s in db.query(Species).filter(Species.common_name.in_(SENTINEL_LABELS)).all()}

        rows = (
            db.query(
                Detection.id,
                Detection.visit_id,
                Detection.track_id,
                Detection.bbox,
                Visit.started_at,
                Detection.species_id,
                Detection.confidence,
            )
            .join(Visit, Visit.id == Detection.visit_id)
            .all()
        )

        frames_dir = Path(args.frames_dir)
        birds_by_frame: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        junk_by_frame: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        frame_day_map: dict[str, str] = {}

        for det_id, visit_id, track_id, bbox, started_at, initial_species_id, conf in rows:
            if not bbox or len(bbox) != 4:
                continue
            day = heldout.local_day(started_at, tz)
            if day in heldout_days:
                continue

            fname = f"v{visit_id:08d}_t{track_id:04d}.jpg"
            fpath = frames_dir / fname
            if not fpath.exists():
                continue

            frame_day_map[fname] = day
            gt_box = tuple(bbox)

            if det_id in user_labels:
                corr_id, species_id, label = user_labels[det_id]
                if label == POOR_QUALITY_LABEL:
                    continue
                elif label == NOT_A_BIRD_LABEL:
                    junk_by_frame[fname].append(gt_box)
                else:
                    birds_by_frame[fname].append(gt_box)
            else:
                # High-confidence implicit birds from 2026-09-01
                if day >= "2026-09-01":
                    if initial_species_id is not None and initial_species_id not in sentinel_ids:
                        if conf is not None and conf >= 0.60:
                            birds_by_frame[fname].append(gt_box)

    all_frame_names = sorted(birds_by_frame.keys())
    if args.limit_frames:
        all_frame_names = all_frame_names[: args.limit_frames]

    log.info("Total usable bird frames with labels: %d", len(all_frame_names))

    # 2. Partition training capture days into train / val
    all_days = sorted({frame_day_map[fn] for fn in all_frame_names})
    rng = random.Random(SPLIT_SEED)
    val_day_count = max(1, round(len(all_days) * VAL_DAY_SHARE))
    val_days = set(rng.sample(all_days, val_day_count))
    train_days = set(all_days) - val_days

    log.info("Day split: %d train days, %d val days (Seed=%d)", len(train_days), len(val_days), SPLIT_SEED)

    tile_rects = _tile_offsets(3840, 2160)
    tile_counter = 0
    stats_counts = {"train_birds": 0, "train_bg": 0, "val_birds": 0, "val_bg": 0}

    for idx, fname in enumerate(all_frame_names):
        if (idx + 1) % 250 == 0:
            log.info("Processing frame %d / %d...", idx + 1, len(all_frame_names))

        day = frame_day_map[fname]
        split = "val" if day in val_days else "train"
        target_img_dir = img_val_dir if split == "val" else img_train_dir
        target_lbl_dir = lbl_val_dir if split == "val" else lbl_train_dir

        fpath = frames_dir / fname
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        bird_boxes = birds_by_frame[fname]
        junk_boxes = junk_by_frame.get(fname, [])

        pos_tiles = []
        neg_tiles = []

        for tile in tile_rects:
            tx, ty, tw, th = tile
            tile_labels = []

            # Check bird overlaps
            for gx, gy, gw, gh in bird_boxes:
                inter = box_intersection_area(tile, (gx, gy, gw, gh))
                bird_area = gw * gh
                # If at least 35% of the bird is in this tile, include it
                if bird_area > 0 and (inter / float(bird_area)) >= 0.35:
                    bx = max(0, gx - tx)
                    by = max(0, gy - ty)
                    bw = min(gx + gw, tx + tw) - (tx + bx)
                    bh = min(gy + gh, ty + th) - (ty + by)
                    if bw > 10 and bh > 10:
                        xc = (bx + bw / 2.0) / float(tw)
                        yc = (by + bh / 2.0) / float(th)
                        wn = float(bw) / float(tw)
                        hn = float(bh) / float(th)
                        tile_labels.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            if tile_labels:
                pos_tiles.append((tile, tile_labels))
            else:
                # Ensure no junk box overlaps this background tile
                has_junk = any(box_intersection_area(tile, j) > 0 for j in junk_boxes)
                if not has_junk:
                    neg_tiles.append(tile)

        # Write positive tiles
        for (tx, ty, tw, th), labels in pos_tiles:
            tile_counter += 1
            tid_str = f"{tile_counter:07d}"
            tile_img = img[ty : ty + th, tx : tx + tw]

            cv2.imwrite(str(target_img_dir / f"tile_{tid_str}.jpg"), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(target_lbl_dir / f"tile_{tid_str}.txt", "w") as f:
                f.write("\n".join(labels) + "\n")

            if split == "train":
                stats_counts["train_birds"] += 1
            else:
                stats_counts["val_birds"] += 1

        # Sample 1:1 matching negative tiles from the same frame
        num_pos = len(pos_tiles)
        if num_pos > 0 and neg_tiles:
            chosen_neg = rng.sample(neg_tiles, min(num_pos, len(neg_tiles)))
            for (tx, ty, tw, th) in chosen_neg:
                tile_counter += 1
                tid_str = f"{tile_counter:07d}"
                tile_img = img[ty : ty + th, tx : tx + tw]

                cv2.imwrite(str(target_img_dir / f"tile_{tid_str}.jpg"), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # Empty label file for pure background
                with open(target_lbl_dir / f"tile_{tid_str}.txt", "w") as f:
                    pass

                if split == "train":
                    stats_counts["train_bg"] += 1
                else:
                    stats_counts["val_bg"] += 1

    log.info("Dataset stats: %s", stats_counts)

    # 3. Write data.yaml
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "bird"},
    }
    import yaml
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    # 4. Write manifest.json
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_days": sorted(train_days),
        "val_days": sorted(val_days),
        "heldout_days_refused": sorted(heldout_days),
        "counts": stats_counts,
        "total_tiles": tile_counter,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Creating tarball archive at %s...", args.archive_path)
    with tarfile.open(args.archive_path, "w:gz") as tar:
        tar.add(str(out_dir), arcname=out_dir.name)

    archive_size_mb = os.path.getsize(args.archive_path) / (1024 * 1024)
    log.info("Dataset packaged successfully: %.1f MB archive at %s", archive_size_mb, args.archive_path)


if __name__ == "__main__":
    main()
