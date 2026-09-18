"""Generate stratified N=1,000 bird held-out yardstick dataset (Milestone 1).

Selects 1,000 bird frames and 400 junk frames with verified frame files on disk,
stratified across all capture months and enforcing the <=8.0% single-day cap:
  - May: 210 frames (21.0%)
  - June: 260 frames (26.0%)
  - July: 201 frames (20.1%)
  - August: 200 frames (20.0%)
  - September: 129 frames (12.9%)
  Total: 1,000 bird frames across >= 90 days.

Usage:
  python backend/scripts/detector/sample_heldout_1000.py --out backend/scripts/detector/heldout_frames_1000.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import (
    NOT_A_BIRD_LABEL,
    POOR_QUALITY_LABEL,
    UNKNOWN_BIRD_LABEL,
    Correction,
    Detection,
    Species,
    Visit,
)
from db.session import SessionLocal
from scripts.train import heldout
from settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUT = Path(__file__).resolve().parent / "heldout_frames_1000.json"
DEFAULT_FRAMES_DIR = Path("/app/data/frames")
LOCAL_FRAMES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "frames"


def frame_name(visit_id: int, track_id: int) -> str:
    return f"v{visit_id:08d}_t{track_id:04d}.jpg"


def build_stratified_yardstick(
    db,
    frames_dir: Path,
    tz: ZoneInfo,
    seed: int = 42,
    target_birds: int = 1000,
    target_junk: int = 400,
) -> dict:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory {frames_dir} does not exist.")

    existing_files = set(os.listdir(frames_dir))
    log.info("Found %d image files in %s", len(existing_files), frames_dir)

    labels = heldout.latest_labels(db, sources=heldout.USER_SOURCES)
    rows = (
        db.query(Detection.id, Detection.visit_id, Detection.track_id, Detection.bbox, Visit.started_at)
        .join(Visit, Visit.id == Detection.visit_id)
        .filter(Detection.id.in_(db.query(Correction.detection_id)))
        .all()
    )

    bird_pool = []
    junk_pool = []

    for det_id, visit_id, track_id, bbox, started_at in rows:
        if det_id not in labels:
            continue
        common_name = labels[det_id][2]
        if common_name is None or common_name == POOR_QUALITY_LABEL:
            continue

        fname = frame_name(visit_id, track_id)
        if fname not in existing_files:
            continue

        local_date_str = heldout.local_day(started_at, tz)
        month_str = local_date_str[:7]

        entry = {
            "detection_id": det_id,
            "visit_id": visit_id,
            "track_id": track_id,
            "frame_name": fname,
            "bbox": bbox,
            "month": month_str,
            "day": local_date_str,
            "label": common_name,
        }

        if common_name == NOT_A_BIRD_LABEL:
            junk_pool.append(entry)
        else:
            bird_pool.append(entry)

    log.info("Available on disk: %d bird frames, %d junk frames", len(bird_pool), len(junk_pool))

    rng = random.Random(seed)

    # 1. Stratified sampling for birds
    by_month_day = defaultdict(lambda: defaultdict(list))
    for b in bird_pool:
        by_month_day[b["month"]][b["day"]].append(b)

    # Quotas matching empirical pool distribution while respecting month proportions
    bird_quotas = {
        "2026-05": 210,
        "2026-06": 260,
        "2026-07": min(201, len([b for b in bird_pool if b["month"] == "2026-07"])),
        "2026-08": 200,
        "2026-09": len([b for b in bird_pool if b["month"] == "2026-09"]),
    }
    rem = target_birds - sum(bird_quotas.values())
    if rem > 0:
        bird_quotas["2026-05"] += rem

    selected_birds = []
    max_per_day = int(0.08 * target_birds)  # <= 8% from any single day (80 max)

    for month, quota in bird_quotas.items():
        days = list(by_month_day[month].keys())
        rng.shuffle(days)
        month_selected = []
        while len(month_selected) < quota:
            added_any = False
            for day in days:
                pool = by_month_day[month][day]
                day_count = sum(1 for x in month_selected if x["day"] == day)
                if pool and day_count < max_per_day and len(month_selected) < quota:
                    idx = rng.randrange(len(pool))
                    month_selected.append(pool.pop(idx))
                    added_any = True
            if not added_any:
                break
        selected_birds.extend(month_selected)

    # 2. Stratified sampling for junk
    by_month_day_junk = defaultdict(lambda: defaultdict(list))
    for j in junk_pool:
        by_month_day_junk[j["month"]][j["day"]].append(j)

    selected_junk = []
    junk_days = [d for m in by_month_day_junk for d in by_month_day_junk[m]]
    rng.shuffle(junk_days)
    max_junk_per_day = max(5, int(0.10 * target_junk))

    while len(selected_junk) < target_junk and junk_days:
        added_any = False
        for day in list(junk_days):
            m = day[:7]
            pool = by_month_day_junk[m][day]
            day_count = sum(1 for x in selected_junk if x["day"] == day)
            if pool and day_count < max_junk_per_day and len(selected_junk) < target_junk:
                idx = rng.randrange(len(pool))
                selected_junk.append(pool.pop(idx))
                added_any = True
            elif not pool:
                junk_days.remove(day)
        if not added_any:
            break

    bird_day_counts = Counter(b["day"] for b in selected_birds)
    bird_month_counts = Counter(b["month"] for b in selected_birds)

    log.info("Selected %d bird frames across %d distinct days", len(selected_birds), len(bird_day_counts))
    log.info("Bird frames by month: %s", sorted(bird_month_counts.items()))
    log.info("Max bird frames in single day: %d (cap: %d)", max(bird_day_counts.values()), max_per_day)
    log.info("Selected %d junk frames", len(selected_junk))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera_timezone": str(tz),
        "seed": seed,
        "targets": {"birds": target_birds, "junk": target_junk},
        "summary": {
            "bird_frames": len(selected_birds),
            "bird_visits": len({b["visit_id"] for b in selected_birds}),
            "bird_days": len(bird_day_counts),
            "max_birds_single_day": max(bird_day_counts.values()),
            "birds_by_month": dict(sorted(bird_month_counts.items())),
            "junk_frames": len(selected_junk),
            "junk_visits": len({j["visit_id"] for j in selected_junk}),
            "junk_days": len({j["day"] for j in selected_junk}),
        },
        "bird_frames": selected_birds,
        "junk_frames": selected_junk,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Path to output JSON")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Path to source frames directory")
    args = parser.parse_args()

    # Auto-detect frames directory: container path or local path
    frames_dir = args.frames_dir
    if frames_dir is None:
        if DEFAULT_FRAMES_DIR.exists():
            frames_dir = DEFAULT_FRAMES_DIR
        elif LOCAL_FRAMES_DIR.exists() and any(LOCAL_FRAMES_DIR.iterdir()):
            frames_dir = LOCAL_FRAMES_DIR
        else:
            frames_dir = DEFAULT_FRAMES_DIR

    db = SessionLocal()
    try:
        tz = ZoneInfo(settings.camera_timezone)
        payload = build_stratified_yardstick(db, frames_dir, tz)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        log.info("Wrote yardstick dataset to %s (%d bytes)", args.out, args.out.stat().st_size)
    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
