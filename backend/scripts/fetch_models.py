"""One-shot: download ML model weights into the persistent models/ volume.

Run this after a fresh deploy (or after bumping model versions) so the first
request doesn't pay the multi-megabyte download cost:

    docker compose run --rm api python scripts/fetch_models.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/fetch_models.py` from the backend/ working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> int:
    hf_home = os.environ.get("HF_HOME")
    log.info("HF_HOME = %s", hf_home)
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)

    # 1) YOLO detection weights. ultralytics downloads on first construction.
    from pipeline.detect import DEFAULT_WEIGHTS_PATH, YOLO_WEIGHTS_FILE  # noqa: WPS433
    from ultralytics import YOLO  # noqa: WPS433

    log.info("Fetching YOLO detector: %s", YOLO_WEIGHTS_FILE)
    DEFAULT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    YOLO(str(DEFAULT_WEIGHTS_PATH) if DEFAULT_WEIGHTS_PATH.exists() else YOLO_WEIGHTS_FILE)
    log.info("YOLO weights ready: %s", DEFAULT_WEIGHTS_PATH)

    # 2) Bird species classifier (HuggingFace).
    from pipeline.classify import warmup  # noqa: WPS433

    log.info("Fetching bird species classifier…")
    warmup()
    log.info("Classifier ready.")

    log.info("All models fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
