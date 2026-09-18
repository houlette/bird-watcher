"""Gate 4 Audit Sample Generator (DETECTOR_PLAN.md Step 4).

Generates three 100-sample contact sheets for Ryan's visual audit:
  1. audit_explicit_birds_100.png: 100 random user-confirmed bird boxes on training days.
  2. audit_implicit_birds_100.png: 100 random uncorrected feed birds from 2026-09-01 (conf >= 0.60).
  3. audit_junk_100.png: 100 random user-rejected junk boxes on training days.

Pass criteria:
  - <= 3 wrong in explicit bird sheet (<= 3%)
  - <= 3 wrong in implicit bird sheet (<= 3%)
  - <= 3 wrong in junk sheet (<= 3%)
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
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
from scripts.train import heldout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED = 0
SAMPLE_SIZE = 100
THUMB_SIZE = 240
COLS = 10


def draw_crop(img: np.ndarray, bbox: tuple[int, int, int, int], label_text: str, pad: float = 0.5) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    x, y, w, h = bbox
    pad_w = int(max(w * pad, 30))
    pad_h = int(max(h * pad, 30))
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    crop = img[y1:y2, x1:x2].copy()
    cx = x - x1
    cy = y - y1
    # Green rectangle around target box
    cv2.rectangle(crop, (cx, cy), (cx + w, cy + h), (0, 255, 0), 2)
    return crop


def make_sheet(crops: list[np.ndarray], titles: list[str], indices: list[int]) -> np.ndarray:
    rows = math.ceil(len(crops) / COLS)
    sheet = np.full((rows * THUMB_SIZE, COLS * THUMB_SIZE, 3), 30, dtype=np.uint8)

    for idx, (crop, title, num) in enumerate(zip(crops, titles, indices, strict=True)):
        r = idx // COLS
        c = idx % COLS
        resized = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        # Overlay number and title
        cv2.putText(resized, f"#{num} {title[:16]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(resized, f"#{num} {title[:16]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sheet[r * THUMB_SIZE : (r + 1) * THUMB_SIZE, c * THUMB_SIZE : (c + 1) * THUMB_SIZE] = resized

    return sheet


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="/app/data/birdwatcher.db")
    parser.add_argument("--frames-dir", type=str, default="/app/data/frames")
    parser.add_argument("--heldout-meta", type=str, default="/app/scripts/detector/heldout_frames.json")
    parser.add_argument("--out-dir", type=str, default="/app/data/gate4_audit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.heldout_meta) as f:
        heldout_meta = json.load(f)
    heldout_days = set(heldout_meta["pre_june_drawn_days"] + heldout_meta["binary_heldout_days"])

    engine = create_engine(f"sqlite:///{args.db}")
    with Session(engine) as db:
        user_labels = heldout.latest_labels(db, sources=heldout.USER_SOURCES)
        tz = ZoneInfo("America/New_York")
        sentinel_ids = {s.id for s in db.query(Species).filter(Species.common_name.in_(SENTINEL_LABELS)).all()}
        species_names = {s.id: s.common_name for s in db.query(Species).all()}

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

        explicit_birds = []
        implicit_birds = []
        explicit_junk = []

        for det_id, visit_id, track_id, bbox, started_at, initial_species_id, conf in rows:
            if not bbox or len(bbox) != 4:
                continue
            day = heldout.local_day(started_at, tz)
            if day in heldout_days:
                continue

            fname = f"v{visit_id:08d}_t{track_id:04d}.jpg"
            fpath = os.path.join(args.frames_dir, fname)
            if not os.path.exists(fpath):
                continue

            if det_id in user_labels:
                corr_id, species_id, label = user_labels[det_id]
                if label == POOR_QUALITY_LABEL:
                    continue
                elif label == NOT_A_BIRD_LABEL:
                    explicit_junk.append((det_id, visit_id, track_id, bbox, day, "Junk", fpath))
                else:
                    explicit_birds.append((det_id, visit_id, track_id, bbox, day, label, fpath))
            else:
                if day >= "2026-09-01":
                    if initial_species_id is not None and initial_species_id not in sentinel_ids:
                        if conf is not None and conf >= 0.60:
                            sp_name = species_names.get(initial_species_id, "Unknown bird")
                            implicit_birds.append((det_id, visit_id, track_id, bbox, day, sp_name, fpath))

    log.info("Candidate pools: %d explicit birds, %d implicit birds, %d explicit junk",
             len(explicit_birds), len(implicit_birds), len(explicit_junk))

    rng = random.Random(SEED)
    sample_exp = rng.sample(explicit_birds, min(SAMPLE_SIZE, len(explicit_birds)))
    sample_imp = rng.sample(implicit_birds, min(SAMPLE_SIZE, len(implicit_birds)))
    sample_jnk = rng.sample(explicit_junk, min(SAMPLE_SIZE, len(explicit_junk)))

    manifest = {"explicit_birds": [], "implicit_birds": [], "junk": []}

    # Generate Explicit Birds Sheet
    log.info("Generating explicit birds contact sheet...")
    crops_exp, titles_exp, nums_exp = [], [], []
    for idx, (det_id, vid, tid, bbox, day, sp, fpath) in enumerate(sample_exp, 1):
        img = cv2.imread(fpath)
        if img is not None:
            crops_exp.append(draw_crop(img, tuple(bbox), sp))
            titles_exp.append(sp)
            nums_exp.append(idx)
            manifest["explicit_birds"].append({"num": idx, "det_id": det_id, "visit_id": vid, "track_id": tid, "species": sp, "day": day, "bbox": bbox})

    sheet_exp = make_sheet(crops_exp, titles_exp, nums_exp)
    cv2.imwrite(str(out_dir / "audit_explicit_birds_100.png"), sheet_exp)

    # Generate Implicit Birds Sheet
    log.info("Generating implicit birds contact sheet...")
    crops_imp, titles_imp, nums_imp = [], [], []
    for idx, (det_id, vid, tid, bbox, day, sp, fpath) in enumerate(sample_imp, 1):
        img = cv2.imread(fpath)
        if img is not None:
            crops_imp.append(draw_crop(img, tuple(bbox), sp))
            titles_imp.append(sp)
            nums_imp.append(idx)
            manifest["implicit_birds"].append({"num": idx, "det_id": det_id, "visit_id": vid, "track_id": tid, "species": sp, "day": day, "bbox": bbox})

    sheet_imp = make_sheet(crops_imp, titles_imp, nums_imp)
    cv2.imwrite(str(out_dir / "audit_implicit_birds_100.png"), sheet_imp)

    # Generate Junk Sheet
    log.info("Generating junk contact sheet...")
    crops_jnk, titles_jnk, nums_jnk = [], [], []
    for idx, (det_id, vid, tid, bbox, day, sp, fpath) in enumerate(sample_jnk, 1):
        img = cv2.imread(fpath)
        if img is not None:
            crops_jnk.append(draw_crop(img, tuple(bbox), sp))
            titles_jnk.append("Junk")
            nums_jnk.append(idx)
            manifest["junk"].append({"num": idx, "det_id": det_id, "visit_id": vid, "track_id": tid, "species": sp, "day": day, "bbox": bbox})

    sheet_jnk = make_sheet(crops_jnk, titles_jnk, nums_jnk)
    cv2.imwrite(str(out_dir / "audit_junk_100.png"), sheet_jnk)

    with open(out_dir / "audit_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Gate 4 audit contact sheets generated successfully in %s", out_dir)


if __name__ == "__main__":
    main()
