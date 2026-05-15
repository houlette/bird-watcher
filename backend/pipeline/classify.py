"""Bird species classifier.

Wraps a HuggingFace image-classification model fine-tuned on a bird dataset.
The default is `dennisjooo/Birds-Classifier-EfficientNetB2` (525 species), but
the model can be swapped via the BIRD_CLASSIFIER_MODEL env var without changing
this file — useful for evaluation or for a domain-fine-tuned model produced by
Phase 6 active-learning.

Phase 4 will consume the top-5 returned here and fuse with Haikubox audio +
seasonal priors. For Phase 3 we just take the top-1 and store top-5 raw.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid heavy imports at module load
    from transformers import PreTrainedModel  # type: ignore
    from transformers.image_processing_utils import BaseImageProcessor  # type: ignore

log = logging.getLogger(__name__)

DEFAULT_MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2"
MODEL_NAME = os.getenv("BIRD_CLASSIFIER_MODEL", DEFAULT_MODEL)

# Number of predictions to keep per crop. Five is plenty for fusion with the
# Haikubox prior and for showing "did you mean..." options in the PWA later.
TOP_K = 5


@dataclass
class SpeciesPrediction:
    species: str       # human-readable label as the model returns it
    probability: float


_lock = Lock()
_model: "PreTrainedModel | None" = None
_processor: "BaseImageProcessor | None" = None


def _load() -> tuple["PreTrainedModel", "BaseImageProcessor"]:
    """Lazily load the model + image processor as a process-wide singleton."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    with _lock:
        if _model is None or _processor is None:
            # Heavy imports deferred so unit tests that monkey-patch classify_bird
            # don't need torch installed.
            from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: WPS433
            import torch  # noqa: WPS433

            log.info("Loading bird classifier: %s", MODEL_NAME)
            _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            _model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
            _model.eval()
            # CPU-only deployment by default. If the VM gains a GPU later,
            # set CUDA_VISIBLE_DEVICES and uncomment:
            #   _model.to("cuda")
            _ = torch  # imported to ensure torch is available at this point
    return _model, _processor


def classify_bird(crop_bgr: np.ndarray) -> list[SpeciesPrediction]:
    """Run the classifier on a single BGR crop. Returns top-K predictions desc by p."""
    import torch  # noqa: WPS433

    if crop_bgr.size == 0:
        return []

    model, processor = _load()

    # The processor expects RGB; OpenCV gives us BGR.
    rgb = crop_bgr[:, :, ::-1].copy()
    inputs = processor(images=rgb, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(TOP_K, probs.shape[0]))

    id2label = model.config.id2label
    return [
        SpeciesPrediction(species=id2label[int(idx)].strip(), probability=float(p))
        for p, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]


def warmup() -> None:
    """Optionally pre-load the model at startup so the first detection isn't slow."""
    _load()
