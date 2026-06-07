"""Binary bird-vs-NAB post-filter.

Sits after the species classifier. Looks at the same crop and emits a
single number: P(NAB). Callers (process.py) override the species
classifier's output to "Not a bird" when this probability crosses
`settings.bird_binary_nab_threshold` — currently 0.75.

Architecture notes:
  - Same EfficientNet-B2 backbone the species classifier uses, with a
    fresh 2-way head trained on every labeled crop in the DB (GOLD +
    Claude HIGH + Claude MEDIUM). See scripts/train/finetune_binary.py.
  - Disabled cleanly when the env var is unset: callers can ask for a
    probability and get None back without raising. Pre-deploy
    environments (fresh checkouts, tests) just keep working.
  - Lazy load + thread-safe singleton — model weights aren't touched
    until the first detection comes in. Workers and request handlers
    share one instance.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from settings import settings

log = logging.getLogger(__name__)

_model = None
_processor = None
_device = None
_nab_idx: int | None = None
_lock = threading.Lock()
# Sticky disable. Once we've decided this process can't load the model
# (env unset, missing path, load error), don't keep retrying — every
# detection would re-pay the cost. A restart re-arms the load.
_disabled = False


def _resolve_device():
    """Pick the best available device. Container CPUs only have CPU
    today; this hook lets the same module light up on GPU in dev/MPS."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except ImportError:
        pass
    return None  # signals CPU; torch device created below


def is_enabled() -> bool:
    """True iff a binary filter model is configured AND hasn't failed to load."""
    if _disabled:
        return False
    return bool(settings.bird_binary_filter_model)


def _load() -> bool:
    """Lazy-load model on first call. Returns True if usable, False if disabled."""
    global _model, _processor, _device, _nab_idx, _disabled
    if _disabled:
        return False
    if _model is not None:
        return True

    with _lock:
        if _disabled:
            return False
        if _model is not None:
            return True

        path = settings.bird_binary_filter_model
        if not path:
            _disabled = True
            log.info("BIRD_BINARY_FILTER_MODEL not set; binary post-filter disabled.")
            return False

        # Allow either a local path (preferred — model is yard-specific)
        # or an HF repo name (for shared/community filters down the road).
        local = Path(path)
        if "/" in path and not local.exists() and len(local.parts) <= 2:
            # Looks like an HF repo name (e.g. "user/model"). Let HF handle it.
            pass
        elif not local.exists():
            log.warning("Binary filter model dir not found: %s — disabling.", path)
            _disabled = True
            return False

        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoModelForImageClassification,
            )

            log.info("Loading binary post-filter: %s", path)
            _processor = AutoImageProcessor.from_pretrained(path)
            _model = AutoModelForImageClassification.from_pretrained(path)
            # Find the NAB class index from the model's own id2label so we
            # don't hardcode the ordering — `ImageFolder` sorts class
            # names alphabetically, which means ["bird", "not_a_bird"]
            # → indices (0, 1). But the model file is the source of truth.
            id2label = _model.config.id2label
            _nab_idx = next(
                (int(i) for i, lbl in id2label.items() if "not" in str(lbl).lower()),
                1,
            )
            log.info("    NAB class index: %d (id2label=%s)", _nab_idx, id2label)
            dev = _resolve_device()
            _device = dev if dev is not None else torch.device("cpu")
            _model.to(_device)
            _model.eval()
        except Exception:
            log.exception("Failed to load binary filter from %s — disabling.", path)
            _disabled = True
            return False

        return True


def nab_probability(crop_bgr: np.ndarray) -> float | None:
    """Return P(NAB) ∈ [0, 1] for the given crop, or None if disabled."""
    if not _load():
        return None
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    import torch

    # The HF AutoImageProcessor expects RGB; OpenCV gives BGR.
    crop_rgb = crop_bgr[..., ::-1]
    inputs = _processor(images=crop_rgb, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
    return float(probs[_nab_idx])


def warmup() -> None:
    """Pre-load the model so the first real detection doesn't pay the
    one-time load cost. Safe to call from process startup."""
    if is_enabled():
        _load()
