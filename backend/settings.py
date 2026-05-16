"""Runtime configuration via environment variables.

Centralizes env reads so test code and routers don't sprinkle os.getenv calls.
"""
from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Haikubox audio integration. Both must be set for the poller to do real work;
    # if either is empty, the poller logs a warning and no-ops on each tick.
    haikubox_api_key: str = os.getenv("HAIKUBOX_API_KEY", "")
    haikubox_serial: str = os.getenv("HAIKUBOX_SERIAL", "")

    # The classifier is allowed up to this many seconds of look-back into audio
    # detections to count a bird as "audio-confirmed."
    audio_correlation_window_seconds: int = 90

    # Bird species classifier (HuggingFace transformers image-classification).
    # Default: dennisjooo/Birds-Classifier-EfficientNetB2 (525-class gpiosenka
    # dataset), post-filtered to an eastern-NA backyard species allow-list
    # inside pipeline/classify.py. Override to swap models entirely; if your
    # replacement is already North-America-only, you can disable the allow-list
    # by editing classify.NA_BACKYARD_ALLOWLIST.
    bird_classifier_model: str = os.getenv(
        "BIRD_CLASSIFIER_MODEL",
        "dennisjooo/Birds-Classifier-EfficientNetB2",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
