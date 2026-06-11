"""Manual trigger for the Haikubox audio backfill + re-correlate pass.

The same job runs automatically every day at 03:00 UTC via the worker
scheduler (see pipeline.worker.start_worker). This script is for
ad-hoc runs — backfilling history after a poller outage, sanity-
checking new wiring, or measuring how much the pass would gain right
now without waiting for cron.

Usage (on the VM):

    docker compose exec api python scripts/run_audio_backfill.py

Exit codes:
    0 — pass ran (may have inserted 0 / confirmed 0 rows; that's not
        a failure, just nothing to do).
    2 — Haikubox API key / serial not configured.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.haikubox import (  # noqa: E402
    BACKFILL_HOURS,
    RECORRELATE_LOOKBACK_HOURS,
    backfill_and_recorrelate,
)
from settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_audio_backfill")


def main() -> int:
    if not (settings.haikubox_api_key and settings.haikubox_serial):
        log.error(
            "Haikubox API not configured (HAIKUBOX_API_KEY / HAIKUBOX_SERIAL unset). "
            "Nothing to do."
        )
        return 2
    log.info(
        "Running backfill: fetching last %d h of audio, then re-correlating "
        "detections captured in the last %d h.",
        BACKFILL_HOURS, RECORRELATE_LOOKBACK_HOURS,
    )
    result = backfill_and_recorrelate()
    log.info(
        "Done. Audio rows inserted: %d.  Detections newly audio-confirmed: %d.",
        result["inserted"], result["confirmed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
