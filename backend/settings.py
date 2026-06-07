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

    # Optional binary bird-vs-NAB post-filter. When BIRD_BINARY_FILTER_MODEL
    # is set to a path or HF repo, every detection is also scored by a
    # 2-class head trained on this yard's labeled crops; if it says NAB
    # with probability above the threshold below, the species classifier's
    # output is overridden to "Not a bird". Suppresses YOLO false positives
    # the species classifier lets through (ivy, leaves, debris). Empty
    # string disables the post-filter entirely.
    bird_binary_filter_model: str = os.getenv("BIRD_BINARY_FILTER_MODEL", "")
    # Tunable threshold. Higher = stricter (fewer overrides, more false
    # bird-keeps); lower = more aggressive (more overrides, more real
    # birds suppressed). 0.75 matches the best-epoch validation point
    # where leak-rate and miss-rate were roughly balanced.
    bird_binary_nab_threshold: float = float(
        os.getenv("BIRD_BINARY_NAB_THRESHOLD", "0.75")
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
    # Set VAPID_SUBJECT in .env to your real contact; the default is a no-op
    # placeholder so the repo doesn't ship a personal email address.
    vapid_subject: str = os.getenv("VAPID_SUBJECT", "mailto:you@example.com")

    # Camera location, used by pipeline.daylight to skip visits captured
    # outside daylight hours. At night the Reolink switches to IR/grayscale
    # mode (which the species classifier wasn't trained on) and birds aren't
    # active anyway — processing those clips wastes CPU and fills the feed
    # with unidentifiable shadows. Defaults to NYC; override in backend/.env
    # with the actual yard's coordinates for accurate sunrise/sunset times.
    camera_latitude: float = float(os.getenv("CAMERA_LATITUDE", "40.7128"))
    camera_longitude: float = float(os.getenv("CAMERA_LONGITUDE", "-74.0060"))
    # IANA timezone name for the camera's physical location. Used by
    # pipeline.daylight to compute LOCAL sunrise / sunset — astral's date
    # arg gets interpreted in this zone, which avoids a UTC/local date
    # mismatch (a 20:00 EDT sunset is 00:00 next-day UTC, so "sunset on
    # May 27 UTC" actually means May 26 EDT's sunset).
    camera_timezone: str = os.getenv("CAMERA_TIMEZONE", "America/New_York")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
