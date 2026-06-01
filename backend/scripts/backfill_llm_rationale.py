"""Backfill Correction.rationale for LLM-sourced corrections.

The first ~280 llm-claude corrections were written before
Correction.rationale existed. Their explanations live in the JSONL
audit files under data/llm_classify_results/. This script reads every
JSONL in that directory and populates Correction.rationale for any
matching detection_id whose Correction.source = "llm-claude" and
.rationale IS NULL.

Idempotent — re-runs are a no-op once rationales are populated.

Usage:
    cd backend
    python scripts/backfill_llm_rationale.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.models import Correction  # noqa: E402
from db.session import SessionLocal, init_db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_llm_rationale")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_classify_results"


def main() -> int:
    init_db()
    if not RESULTS_DIR.exists():
        log.error("Results dir missing: %s", RESULTS_DIR)
        return 1

    # det_id → rationale, last-write-wins (later JSONL entries override).
    rationales: dict[int, str] = {}
    for jsonl in sorted(RESULTS_DIR.glob("run_*.jsonl")):
        n = 0
        for line in jsonl.open():
            r = json.loads(line)
            res = r.get("result") or {}
            if res.get("confidence") == "HIGH" and res.get("rationale"):
                rationales[int(r["detection_id"])] = res["rationale"]
                n += 1
        log.info("Read %s: %d HIGH-confidence rationales", jsonl.name, n)
    log.info("Total HIGH rationales loaded: %d", len(rationales))

    db = SessionLocal()
    updated = 0
    skipped = 0
    try:
        for c in (
            db.query(Correction)
            .filter(Correction.source == "llm-claude")
            .filter(Correction.rationale.is_(None))
            .all()
        ):
            rat = rationales.get(c.detection_id)
            if rat:
                c.rationale = rat
                updated += 1
            else:
                skipped += 1
        if updated:
            db.commit()
        log.info("Updated %d Corrections; skipped %d (no JSONL match)", updated, skipped)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
