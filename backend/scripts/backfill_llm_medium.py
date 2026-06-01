"""Backfill Corrections for MEDIUM-confidence Claude calls captured in
the JSONL audit trail.

The first wave of llm_classify_unidentified.py runs only auto-committed
HIGH-confidence calls. The MEDIUMs landed in the JSONL but never made
it to the DB, so the user couldn't review them in the feed. This script
reads every JSONL under data/llm_classify_results/ and writes a
Correction for each MEDIUM entry whose Detection has no Correction yet,
tagged source="llm-claude-medium" so the LLM-review filter can find
them.

Idempotent: the existence check (no Correction yet for the detection)
skips anything already committed by a prior backfill run OR by a HIGH
auto-commit from the live script.

Usage:
    cd backend
    python scripts/backfill_llm_medium.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Correction, Detection, Species  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_llm_medium")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_classify_results"
SOURCE_TAG = "llm-claude-medium"


def _load_medium_results() -> dict[int, dict]:
    """Return {detection_id: latest_medium_result} across all JSONLs.

    Last-write-wins: if a detection has multiple MEDIUM JSONL entries
    (e.g., the user re-ran with --limit on a small window), the most
    recent one wins. Sorting filenames by name preserves chronology
    because they're timestamp-prefixed.
    """
    out: dict[int, dict] = {}
    for jsonl in sorted(RESULTS_DIR.glob("run_*.jsonl")):
        n = 0
        for line in jsonl.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            res = r.get("result") or {}
            if res.get("confidence") != "MEDIUM":
                continue
            label = res.get("label")
            if not label:
                continue
            out[int(r["detection_id"])] = {
                "label": label,
                "rationale": res.get("rationale"),
            }
            n += 1
        log.info("Read %s: %d MEDIUM rows", jsonl.name, n)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to the DB; just log what would happen.")
    args = parser.parse_args()

    init_db()
    if not RESULTS_DIR.exists():
        log.error("Results dir missing: %s", RESULTS_DIR)
        return 1

    candidates = _load_medium_results()
    log.info("Loaded %d MEDIUM candidates from JSONL", len(candidates))
    if not candidates:
        return 0

    db = SessionLocal()
    written = 0
    skipped_no_det = 0
    skipped_already = 0
    skipped_unknown_species = 0
    try:
        # Pre-fetch the set of detection_ids that already have any
        # Correction — saves N queries.
        det_ids_with_correction = {
            row[0] for row in db.query(Correction.detection_id).all()
        }
        # Cache Species lookups so we don't hit the DB per row.
        species_cache: dict[str, Species] = {
            sp.common_name: sp for sp in db.query(Species).all()
        }

        for det_id, payload in candidates.items():
            if det_id in det_ids_with_correction:
                skipped_already += 1
                continue
            det = db.get(Detection, det_id)
            if det is None:
                skipped_no_det += 1
                continue
            label = payload["label"]
            sp = species_cache.get(label)
            if sp is None:
                # Only create on the fly for known yard species — if the
                # label is genuinely off-list it's likely a hallucination
                # and we don't want to pollute the Species table.
                skipped_unknown_species += 1
                continue
            if args.dry_run:
                written += 1
                continue
            det.species_id = sp.id
            db.add(Correction(
                detection_id=det_id,
                correct_species_id=sp.id,
                source=SOURCE_TAG,
                rationale=payload.get("rationale"),
            ))
            written += 1
            # Commit in chunks so a crash doesn't lose all progress.
            if written % 200 == 0:
                db.commit()
                log.info("Committed %d so far", written)
        db.commit()
        log.info(
            "%s %d MEDIUM correction(s); skipped %d (already corrected), "
            "%d (detection gone), %d (unknown species)",
            "Would write" if args.dry_run else "Wrote",
            written, skipped_already, skipped_no_det, skipped_unknown_species,
        )
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
