"""The held-out yardstick for the bird detector (DETECTOR_PLAN.md, step 2).

Every detector candidate from step 3 on is judged on frames from capture days
that no detector training set may include. This module picks those days,
freezes the labelled frames on them, and gives the step 4 export the check
that refuses them.

A frame is a saved source frame, v{visit}_t{track}.jpg, the best frame of one
track. It counts if the file exists and its detection's latest user label
(Correction.source NULL or user-confirmed) is a species or Unknown bird (a
bird frame) or Not a bird (a junk frame). Poor quality and unlabelled frames
are left out.

Held-out days are the binary filter yardstick's heldout_days, so both models
stay held out together for end-to-end checks, plus a seeded random quarter of
the capture days before 2026-06-07 that hold a bird or junk frame. The binary
yardstick's own days all fall on or after that date.

The split is by the camera's local capture day, never by frame: frames of one
day share the light, and frames of one visit usually share the bird.

Count first, then freeze. Both read the live database, so run them in the api
container:
    python scripts/detector/yardstick.py count [--pre-share 0.25]
    python scripts/detector/yardstick.py freeze --out /tmp/heldout_frames.json

Freezing refuses to overwrite. The plan forbids refreezing once step 3 has
started, because every candidate must be scored on the same frames.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import NOT_A_BIRD_LABEL, POOR_QUALITY_LABEL, Correction, Detection, Visit  # noqa: E402
from scripts.train import heldout  # noqa: E402

FROZEN_PATH = Path(__file__).with_name("heldout_frames.json")
FRAMES_DIR = heldout.DATA_DIR / "frames"

PRE_JUNE_SHARE = 0.25
SPLIT_SEED = 0

BIRD = "bird"
JUNK = "junk"
KINDS = (BIRD, JUNK)

# Gate 2's minimums, from DETECTOR_PLAN.md.
GATE = {
    "heldout_bird_frames": 300,
    "heldout_junk_frames": 150,
    "heldout_days_with_frames": 15,
    "training_bird_boxes": 1500,
}


@dataclass(frozen=True)
class Frame:
    detection_id: int
    visit_id: int
    name: str
    day: str
    kind: str


def frame_name(visit_id: int, track_id: int) -> str:
    """The file name pipeline.process._save_source_frames writes."""
    return f"v{visit_id:08d}_t{track_id:04d}.jpg"


def frame_kind(label: str | None) -> str | None:
    """bird, junk, or None for a label that is neither."""
    if label is None or label == POOR_QUALITY_LABEL:
        return None
    if label == NOT_A_BIRD_LABEL:
        return JUNK
    return BIRD


def labelled_frames(db, frame_names: set[str], tz: ZoneInfo) -> tuple[list[Frame], Counter]:
    """Every frame that counts, and how many labelled detections were left
    out and why."""
    labels = heldout.latest_labels(db, sources=heldout.USER_SOURCES)
    rows = (
        db.query(Detection.id, Detection.visit_id, Detection.track_id, Visit.started_at)
        .join(Visit, Visit.id == Detection.visit_id)
        .filter(Detection.id.in_(select(Correction.detection_id)))
        .all()
    )
    frames: list[Frame] = []
    skipped: Counter[str] = Counter()
    for det_id, visit_id, track_id, started_at in rows:
        if det_id not in labels:
            continue  # corrected only by a model, never by the user
        kind = frame_kind(labels[det_id][2])
        if kind is None:
            skipped["poor_quality"] += 1
            continue
        name = frame_name(visit_id, track_id)
        if name not in frame_names:
            skipped[f"{kind}:frame_missing"] += 1
            continue
        frames.append(Frame(det_id, visit_id, name, heldout.local_day(started_at, tz), kind))
    frames.sort(key=lambda f: f.detection_id)
    return frames, skipped


def pick_heldout_days(
    frames: list[Frame], binary_days: frozenset[str], share: float = PRE_JUNE_SHARE, seed: int = SPLIT_SEED
) -> tuple[set[str], list[str], list[str]]:
    """(held-out days, pre-June pool, the days drawn from it)."""
    pool = sorted({f.day for f in frames if f.day < heldout.DAY_SPLIT_FROM})
    drawn = sorted(random.Random(seed).sample(pool, round(share * len(pool))))
    return set(binary_days) | set(drawn), pool, drawn


def summarize(frames: list[Frame], heldout_days: set[str]) -> dict[str, int]:
    held = [f for f in frames if f.day in heldout_days]
    train = [f for f in frames if f.day not in heldout_days]
    out = {}
    for kind in KINDS:
        out[f"heldout_{kind}_frames"] = sum(f.kind == kind for f in held)
        out[f"heldout_{kind}_visits"] = len({f.visit_id for f in held if f.kind == kind})
    out["heldout_days_with_frames"] = len({f.day for f in held})
    # One box per bird frame: the frame's own track. Step 4 also counts other
    # tracks' boxes at the same frame index, so this is a lower bound.
    out["training_bird_boxes"] = sum(f.kind == BIRD for f in train)
    out["training_junk_frames"] = sum(f.kind == JUNK for f in train)
    out["training_days_with_frames"] = len({f.day for f in train})
    return out


def gate_result(summary: dict[str, int]) -> dict[str, dict]:
    return {k: {"value": summary[k], "minimum": m, "pass": summary[k] >= m} for k, m in GATE.items()}


@dataclass(frozen=True)
class DetectorYardstick:
    frozen_at: str
    camera_timezone: str
    heldout_days: frozenset[str]
    frames: dict[str, tuple[int, ...]]

    def refuses(self, started_at: datetime) -> bool:
        """True if a visit that started at this naive UTC time was captured on
        a held-out day. Every detector training export must drop such rows."""
        return heldout.local_day(started_at, ZoneInfo(self.camera_timezone)) in self.heldout_days


def load_yardstick(path: Path = FROZEN_PATH) -> DetectorYardstick:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. A detector training export must not run without "
            "it, because it could then train on held-out days."
        )
    raw = json.loads(path.read_text())
    return DetectorYardstick(
        frozen_at=raw["frozen_at"],
        camera_timezone=raw["camera_timezone"],
        heldout_days=frozenset(raw["heldout_days"]),
        frames={k: tuple(int(i) for i in raw["frames"][k]) for k in KINDS},
    )


def build(db, tz: ZoneInfo, share: float, frames_dir: Path = FRAMES_DIR) -> dict:
    binary = heldout.load_yardstick()
    names = {p.name for p in frames_dir.glob("v*_t*.jpg")}
    frames, skipped = labelled_frames(db, names, tz)
    days, pool, drawn = pick_heldout_days(frames, binary.heldout_days, share)
    summary = summarize(frames, days)
    held = [f for f in frames if f.day in days]
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "camera_timezone": str(tz),
        "split_seed": SPLIT_SEED,
        "pre_june_share": share,
        "day_split_from": heldout.DAY_SPLIT_FROM,
        "binary_yardstick_frozen_at": binary.frozen_at,
        "pre_june_pool_days": len(pool),
        "pre_june_drawn_days": drawn,
        "binary_heldout_days": sorted(binary.heldout_days),
        "heldout_days": sorted(days),
        "frame_files_on_disk": len(names),
        "summary": summary,
        "gate": gate_result(summary),
        "excluded": dict(sorted(skipped.items())),
        "frames": {k: {str(f.detection_id): f.name for f in held if f.kind == k} for k in KINDS},
    }


def report(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        f"pre-June pool {payload['pre_june_pool_days']} days, drew {len(payload['pre_june_drawn_days'])} "
        f"at share {payload['pre_june_share']}; binary yardstick days {len(payload['binary_heldout_days'])}; "
        f"held-out days {len(payload['heldout_days'])}",
        f"held out: {s['heldout_bird_frames']} bird frames ({s['heldout_bird_visits']} visits), "
        f"{s['heldout_junk_frames']} junk frames ({s['heldout_junk_visits']} visits), "
        f"on {s['heldout_days_with_frames']} days",
        f"training: {s['training_bird_boxes']} bird boxes (lower bound), {s['training_junk_frames']} junk frames, "
        f"on {s['training_days_with_frames']} days",
        f"left out: {payload['excluded'] or 'none'}; frame files on disk {payload['frame_files_on_disk']}",
    ]
    for name, g in payload["gate"].items():
        lines.append(f"  gate {name}: {g['value']} >= {g['minimum']}  {'PASS' if g['pass'] else 'FAIL'}")
    lines.append("GATE 2 " + ("PASS" if all(g["pass"] for g in payload["gate"].values()) else "FAIL"))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("count", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--pre-share", type=float, default=PRE_JUNE_SHARE)
    sub.choices["freeze"].add_argument("--out", type=Path, default=FROZEN_PATH)
    sub.choices["freeze"].add_argument("--force", action="store_true", help="Overwrite an existing yardstick.")
    args = ap.parse_args()

    if args.cmd == "freeze" and args.out.exists() and not args.force:
        print(f"{args.out} exists. Refreezing changes the frames every candidate is "
              "scored on; pass --force only before step 3 has started.", file=sys.stderr)
        return 1

    from db.session import SessionLocal
    from settings import settings

    db = SessionLocal()
    try:
        payload = build(db, ZoneInfo(settings.camera_timezone), args.pre_share)
    finally:
        db.close()

    print(report(payload))
    if args.cmd == "freeze":
        args.out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
