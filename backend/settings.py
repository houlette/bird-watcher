"""Runtime configuration via environment variables.

Centralizes env reads so test code and routers don't sprinkle os.getenv calls.
"""
from __future__ import annotations

import os
from pathlib import Path

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

    # Web Push (VAPID). The public key is sent to the browser at subscription
    # time; the private key signs the JWT in each push request. Generate both
    # via scripts/generate_vapid_keys.py.
    vapid_public_key: str = os.getenv("VAPID_PUBLIC_KEY", "")
    vapid_private_pem_path: str = os.getenv(
        "VAPID_PRIVATE_PEM_PATH",
        str(Path(__file__).resolve().parent / "secrets" / "vapid_private.pem"),
    )
    # 'sub' claim for the VAPID JWT — push services require either a mailto:
    # or an https URL so they can contact the app owner if abuse happens.
    vapid_subject: str = os.getenv("VAPID_SUBJECT", "mailto:you@example.com")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
