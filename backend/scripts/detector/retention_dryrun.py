"""What would the frame retention pass delete if it ran now? Deletes nothing.

Frame retention keeps two groups indefinitely (pipeline.worker
._preserved_detection_keys): detections with a Correction, and detections the
feed shows as a bird, which are Ryan's implicit bird labels. Everything else
goes at FRAME_RETENTION_DAYS. Those frames are the detector fine-tune's
training data, so before trusting the pass, rehearse it: this prints what it
would keep and delete, with each would-be deletion grouped by the label its
detection carries.

Anything reported under "user_labelled" or "corrected" is a defect: those
frames must never expire.

    docker exec birdwatcher-api python scripts/detector/retention_dryrun.py
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db.models import Correction, Detection, Species  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline import worker  # noqa: E402
from scripts.train import heldout  # noqa: E402

_FRAME_RE = re.compile(r"^v(\d+)_t(\d+)\.jpg$")


def main() -> int:
    preserved = worker._preserved_detection_keys()
    if preserved is None:
        print("The database could not be read, so the pass would delete nothing.")
        return 1
    db = SessionLocal()
    try:
        user = heldout.latest_labels(db, sources=heldout.USER_SOURCES)
        corrected = {d for (d,) in db.query(Correction.detection_id).distinct()}
        label = {
            (vid, tid): (det_id, name)
            for det_id, vid, tid, name in db.query(
                Detection.id, Detection.visit_id, Detection.track_id, Species.common_name
            ).outerjoin(Species, Species.id == Detection.species_id)
        }
    finally:
        db.close()

    cutoff = time.time() - worker.FRAME_RETENTION_DAYS * 86400
    counts: Counter[str] = Counter()
    for path in worker.FRAMES_DIR.glob("v*_t*.jpg"):
        m = _FRAME_RE.match(path.name)
        key = (int(m.group(1)), int(m.group(2))) if m else None
        if key in preserved:
            counts["kept: on the preserve list"] += 1
        elif path.stat().st_mtime >= cutoff:
            counts[f"kept: newer than {worker.FRAME_RETENTION_DAYS} days"] += 1
        else:
            det_id, name = label.get(key, (None, "no detection row"))
            if det_id in user:
                kind = "user_labelled"
            elif det_id in corrected:
                kind = "corrected"
            else:
                kind = name or "unidentified"
            counts[f"would delete: {kind}"] += 1

    print(f"{len(preserved)} detections on the preserve list")
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
