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
    # detections before visit.started_at to count a bird as "audio-confirmed."
    audio_correlation_window_seconds: int = 90

    # Look-ahead seconds after visit.started_at for audio correlation.
    # Video clips are ~20s; looking ahead 30s covers vocalizations during
    # the visit and immediate departure, as well as minor camera NTP clock skew.
    audio_correlation_lookahead_seconds: int = 30

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

    # The per-hour backdrop filter, off by default. Measured 2026-09-13
    # against June 2026 (630 user-confirmed birds, 930 user-confirmed junk,
    # scored on a June-era model with capture-time binning), it removes
    # 37.5% of the birds to remove 34.0% of the junk, and 30.4% of the 23
    # independently audio-confirmed birds. It is slightly worse than a coin
    # flip at every threshold tried, so it stays off until the mechanism
    # itself changes. Set BACKDROP_FILTER_ENABLED=1 to put it back.
    backdrop_filter_enabled: bool = os.getenv("BACKDROP_FILTER_ENABLED", "") not in ("", "0", "false", "False")

    # Phase 0 motion-gated spatial tiling engine. When enabled, non-keyframe frames
    # only evaluate tiles exhibiting motion, active tracks, or recent perch memory.
    # Keyframes (every 1.0s / 3rd frame) evaluate all 15 tiles. Default is enabled after
    # passing Stage 1 and Stage 2 validation gates. Set MOTION_GATED_TILES_ENABLED=0
    # to disable if needed.
    motion_gated_tiles_enabled: bool = os.getenv("MOTION_GATED_TILES_ENABLED", "1") in ("1", "true", "True", "yes")

    # Lucky imaging: hunt adjacent source video frames around soft/blurry crops
    # to find motion-still, peak-sharpness micro-moments.
    lucky_imaging_enabled: bool = os.getenv("LUCKY_IMAGING_ENABLED", "1") in ("1", "true", "True", "yes")
    lucky_imaging_threshold: float = float(os.getenv("LUCKY_IMAGING_THRESHOLD", "200.0"))

    # Chroma-guided filter: reconstruct 4:2:0 subsampled chroma channels
    # using full-resolution luma as guide to eliminate color bleed and chroma noise.
    chroma_filter_enabled: bool = os.getenv("CHROMA_FILTER_ENABLED", "1") in ("1", "true", "True", "yes")

    # Mertens exposure fusion: multiscale exposure fusion (Tommert & Mertens)
    # blending synthetic exposure brackets to recover shadow plumage and highlights without halos.
    mertens_fusion_enabled: bool = os.getenv("MERTENS_FUSION_ENABLED", "1") in ("1", "true", "True", "yes")

    # Multi-frame shift-and-add super-resolution: sub-pixel frame registration
    # across burst frames from source video to gain optical resolution and reduce noise.
    super_res_enabled: bool = os.getenv("SUPER_RES_ENABLED", "1") in ("1", "true", "True", "yes")
    super_res_scale: int = int(os.getenv("SUPER_RES_SCALE", "2"))
    super_res_max_crop_size: int = int(os.getenv("SUPER_RES_MAX_CROP_SIZE", "240"))

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
