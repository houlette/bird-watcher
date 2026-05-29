"""Bird species classifier — eastern North American backyard feeder filter.

The base model is `dennisjooo/Birds-Classifier-EfficientNetB2`, a HuggingFace
image-classification model trained on the 525-species gpiosenka dataset.
Without filtering it confidently emits exotic species (Fiordland Penguin,
Baikal Teal, etc.) for any unfamiliar-looking crop — see notes in the smoke
test transcript.

To keep the labels honest for a New England backyard, we apply a curated
allow-list of ~60 common eastern-NA species at the post-softmax stage:

  1. Zero out probability mass on any class not in the allow-list.
  2. Renormalize over the allow-list.
  3. If the *original* (pre-zero) allow-list mass is below
     IN_RANGE_THRESHOLD, return an empty list. That's our cheap stand-in for
     a `not_a_bird` reject — when the model's confidence is concentrated on
     species that don't live here, the crop is most likely a leaf, shadow,
     or non-bird animal and shouldn't produce a Detection row.

When you swap to a North-American-trained model (e.g. a real NABirds-tuned
EfficientNet), drop the allow-list filter — it should be a no-op anyway.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedModel  # type: ignore
    from transformers.image_processing_utils import BaseImageProcessor  # type: ignore

from pipeline import calibration
from settings import settings

log = logging.getLogger(__name__)

DEFAULT_MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2"

TOP_K = 5

# If the total post-softmax mass on allow-listed species is below this fraction,
# the crop is treated as a false positive and yields no detection.
IN_RANGE_THRESHOLD = 0.10

# Eastern-NA backyard / feeder species. All match the uppercase, mixed-hyphen
# convention used by the gpiosenka dataset that the default model is trained on
# (e.g. "DARK EYED JUNCO" not "DARK-EYED JUNCO"). When in doubt, the smoke-test
# transcript or the model's id2label is ground truth.
NA_BACKYARD_ALLOWLIST: frozenset[str] = frozenset({
    # Cardinals, finches, sparrows, buntings — feeder regulars
    "NORTHERN CARDINAL",
    "PYRRHULOXIA",  # cardinal lookalike, for SW visitors
    "HOUSE FINCH",
    "PURPLE FINCH",
    "CASSINS FINCH",
    "AMERICAN GOLDFINCH",
    "LESSER GOLDFINCH",
    "EVENING GROSBEAK",
    "PINE GROSBEAK",
    "ROSE BREASTED GROSBEAK",
    "INDIGO BUNTING",
    "PAINTED BUNTING",
    "LARK BUNTING",
    "SONG SPARROW",
    "WHITE-THROATED SPARROW",
    "WHITE CROWNED SPARROW",
    "CHIPPING SPARROW",
    "FOX SPARROW",
    "HOUSE SPARROW",
    "DARK EYED JUNCO",
    "EASTERN TOWHEE",
    # Chickadees, titmice, nuthatches, wrens
    "BLACK CAPPED CHICKADEE",
    "CAROLINA CHICKADEE",
    "TUFTED TITMOUSE",
    "WHITE BREASTED NUTHATCH",
    "RED BREASTED NUTHATCH",
    "CAROLINA WREN",
    "HOUSE WREN",
    # Woodpeckers
    "DOWNY WOODPECKER",
    "HAIRY WOODPECKER",
    "RED BELLIED WOODPECKER",
    "RED HEADED WOODPECKER",
    "PILEATED WOODPECKER",
    "NORTHERN FLICKER",
    # Jays, crows, mockingbirds, catbirds, robins, doves
    "BLUE JAY",
    "AMERICAN CROW",
    "FISH CROW",
    "COMMON RAVEN",
    "GRAY CATBIRD",
    "NORTHERN MOCKINGBIRD",
    "BROWN THRASHER",
    "AMERICAN ROBIN",
    "WOOD THRUSH",
    "HERMIT THRUSH",
    "MOURNING DOVE",
    "ROCK DOVE",
    "EUROPEAN STARLING",
    # Blackbirds, orioles, grackles
    "COMMON GRACKLE",
    "BOAT TAILED GRACKLE",
    "RED WINGED BLACKBIRD",
    "RUSTY BLACKBIRD",
    "BROWN HEADED COWBIRD",
    "BALTIMORE ORIOLE",
    "ORCHARD ORIOLE",
    # Hummingbirds, warblers, vireos
    "RUBY THROATED HUMMINGBIRD",
    "YELLOW WARBLER",
    "YELLOW RUMPED WARBLER",
    "AMERICAN REDSTART",
    "COMMON YELLOWTHROAT",
    "RED EYED VIREO",
    # Hawks (frequent feeder visitors hunting other birds)
    "COOPERS HAWK",
    "SHARP SHINNED HAWK",
    "RED TAILED HAWK",
})


@dataclass
class SpeciesPrediction:
    species: str            # normalized species (title case, no plumage qualifier)
    probability: float
    raw_label: str = ""     # the model's original uppercase label


_lock = Lock()
_model: "PreTrainedModel | None" = None
_processor: "BaseImageProcessor | None" = None
_allowed_idxs: np.ndarray | None = None
_allowed_mask: np.ndarray | None = None


def _hyphen_insensitive(label: str) -> str:
    """Lower-cased, hyphens collapsed to spaces, apostrophes stripped, runs
    of whitespace normalized.

    Used for matching the model's labels against the allow-list since the
    gpiosenka dataset:
      - Is inconsistent about whether to write 'WHITE-THROATED SPARROW'
        or 'WHITE THROATED SPARROW'.
      - Drops possessive apostrophes — 'ANNAS HUMMINGBIRD' for
        'Anna's Hummingbird', 'BREWERS BLACKBIRD' for "Brewer's Blackbird".
    Strip both differences before comparing.
    """
    return re.sub(r"[-\s]+", " ", label.replace("'", "")).strip().lower()


# A small curated table of confirmed gpiosenka-dataset typos. The class
# label on the LEFT is what the model emits; the RIGHT is the correct
# common name. We add the model's misspelled label to the allow-list at
# load time AND map it back to the correct spelling in
# _normalize_for_display so the user never sees the typo.
#
# Tried automated fuzzy-matching (ratio ≥ 0.9) but it gave 3 false
# matches for every 4 real typos — Eastern/Western Bluebird and
# Bank/Barn Swallow would silently conflate distinct species. Curated
# is small and unambiguous; if we ever spot a new typo, add a line here.
MODEL_TYPO_FIXES: dict[str, str] = {
    "BLACKBURNIAM WARBLER": "Blackburnian Warbler",
    "VERMILION FLYCATHER":  "Vermilion Flycatcher",
    "EASTERN TOWEE":        "Eastern Towhee",
    "BROWN CREPPER":        "Brown Creeper",
}


def _normalize_for_display(uppercase_label: str) -> str:
    """Convert 'DARK EYED JUNCO' → 'Dark-eyed Junco' for storage / UI / fusion.

    The dennisjooo model's labels are uppercase and inconsistent about hyphens;
    we normalize to the standard hyphenated forms used by eBird / Haikubox so
    audio correlation can match by exact species name.
    """
    # Short-circuit for the curated set of model-side typos: bypass the
    # general fixups and return the correctly-spelled common name so the
    # user never sees the typo in the feed or in fusion logs.
    if uppercase_label in MODEL_TYPO_FIXES:
        return MODEL_TYPO_FIXES[uppercase_label]
    label = uppercase_label.title()
    # Compound-adjective hyphenations in NA bird names. The leading \s+
    # consumes the space so we get "Dark-eyed" rather than "Dark -eyed".
    fixups = (
        (r"\s+Eyed\b", "-eyed"),
        (r"\s+Throated\b", "-throated"),
        (r"\s+Crowned\b", "-crowned"),
        (r"\s+Breasted\b", "-breasted"),
        (r"\s+Bellied\b", "-bellied"),
        (r"\s+Headed\b", "-headed"),
        (r"\s+Winged\b", "-winged"),
        (r"\s+Shinned\b", "-shinned"),
        (r"\s+Capped\b", "-capped"),
        (r"\s+Backed\b", "-backed"),
        (r"\s+Tailed\b", "-tailed"),
        (r"\s+Rumped\b", "-rumped"),
        (r"\s+Sided\b", "-sided"),
        (r"\s+Ringed\b", "-ringed"),
        (r"\s+Bibbed\b", "-bibbed"),
    )
    for pat, rep in fixups:
        label = re.sub(pat, rep, label, flags=re.IGNORECASE)
    # Restore possessive apostrophes the model strips.
    label = re.sub(r"\bCoopers\b", "Cooper's", label)
    label = re.sub(r"\bCassins\b", "Cassin's", label)
    label = re.sub(r"\bSwainsons\b", "Swainson's", label)
    label = re.sub(r"\s{2,}", " ", label).strip()
    return label


def _load() -> tuple["PreTrainedModel", "BaseImageProcessor"]:
    global _model, _processor, _allowed_idxs, _allowed_mask
    if _model is not None and _processor is not None:
        return _model, _processor
    with _lock:
        if _model is None or _processor is None:
            from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: WPS433
            import torch  # noqa: WPS433

            repo = settings.bird_classifier_model or DEFAULT_MODEL
            log.info("Loading bird classifier: %s", repo)
            _processor = AutoImageProcessor.from_pretrained(repo)
            _model = AutoModelForImageClassification.from_pretrained(repo)
            _model.eval()

            id2label = _model.config.id2label
            # Build the allow-list as the UNION of:
            #   (a) yard-specific list (yard_priors.json from
            #       scripts/calibrate_from_haikubox.py — species the Haikubox
            #       has actually heard at this site), and
            #   (b) the broader curated NA bird list (na_birds.NA_BIRD_SPECIES).
            # Fall back to the hand-coded NA_BACKYARD_ALLOWLIST if neither is
            # available.
            #
            # Why the union: the sweep showed yard-only rejected 88% of
            # user-labeled real birds (classifier had zero in-range mass
            # on any of the 31 yard species for most crops); broadening to
            # ~67 model-matched species recovered 27 more real-bird
            # acceptances per 128-crop batch with NAB rejection essentially
            # unchanged (99% → 97%). We keep the yard list in the union so
            # site-specific species the curated NA list happens to miss
            # still pass through.
            from na_birds import NA_BIRD_SPECIES
            calibrated = calibration.get_allowlist()
            if calibrated:
                source_list = set(calibrated) | set(NA_BIRD_SPECIES)
                log.info("Allow-list source: yard calibration (%d) ∪ NA broad (%d) = %d species",
                         len(calibrated), len(NA_BIRD_SPECIES), len(source_list))
            else:
                source_list = set(NA_BIRD_SPECIES) | set(NA_BACKYARD_ALLOWLIST)
                log.info("Allow-list source: NA broad (%d) ∪ eastern-NA fallback (%d) = %d species",
                         len(NA_BIRD_SPECIES), len(NA_BACKYARD_ALLOWLIST), len(source_list))
            allowlist_normalized = {_hyphen_insensitive(name) for name in source_list}
            allowed = [
                int(idx)
                for idx, label in id2label.items()
                if _hyphen_insensitive(label) in allowlist_normalized
                # Whitelist the model's typo'd labels for species we DO want
                # — the corresponding correct spelling sits in allowlist_normalized
                # already; the model just emits a misspelled key for it.
                or label in MODEL_TYPO_FIXES
            ]
            n_classes = len(id2label)
            mask = np.zeros(n_classes, dtype=bool)
            mask[allowed] = True
            _allowed_idxs = np.array(allowed, dtype=np.int64)
            _allowed_mask = mask
            log.info(
                "Classifier ready: %d total classes, %d allow-listed",
                n_classes, len(allowed),
            )
            _ = torch
    return _model, _processor


# CLAHE (Contrast-Limited Adaptive Histogram Equalization) pre-processing.
# Applied to the L channel of LAB so the chroma stays put — a per-channel
# CLAHE in BGR shifts colors. The A/B run on 942 labeled crops showed it:
#   - holds NAB rejection at 97%
#   - accepts 15% more real birds past the in-range filter
#   - lifts real-bird top-5 accuracy from 62% → 72%
#   - keeps absolute top-1 count unchanged (so no real-bird losses)
# clipLimit=2.0 and 8×8 tiles are the OpenCV defaults; a good fit for
# 200-400 px crops.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _preprocess_for_classifier(bgr: np.ndarray) -> np.ndarray:
    """Apply lighting normalization (CLAHE on L channel) before classification."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _CLAHE.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def classify_bird(crop_bgr: np.ndarray) -> list[SpeciesPrediction]:
    import torch  # noqa: WPS433

    if crop_bgr.size == 0:
        return []

    model, processor = _load()
    assert _allowed_mask is not None  # populated by _load()

    # Lighting-normalize the crop before handing it to the classifier.
    # Doesn't affect the SAVED crop on disk — that path uses crop_bgr
    # unmodified for the user-visible feed image.
    crop_for_model = _preprocess_for_classifier(crop_bgr)
    rgb = crop_for_model[:, :, ::-1].copy()
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Reject crops whose mass is mostly on non-NA species — likely a false
    # positive from YOLO (leaf, shadow, partial bird).
    in_range_mass = float(probs[_allowed_mask].sum())
    if in_range_mass < IN_RANGE_THRESHOLD:
        log.debug("Rejecting crop: in-range mass %.3f < %.2f", in_range_mass, IN_RANGE_THRESHOLD)
        return []

    # Zero out non-allow-listed species and renormalize over what remains.
    filtered = np.where(_allowed_mask, probs, 0.0)
    filtered = filtered / filtered.sum()

    top_idxs = np.argsort(filtered)[::-1][:TOP_K]
    id2label = model.config.id2label
    return [
        SpeciesPrediction(
            species=_normalize_for_display(id2label[int(i)]),
            probability=float(filtered[i]),
            raw_label=id2label[int(i)],
        )
        for i in top_idxs
        if filtered[i] > 0
    ]


def warmup() -> None:
    _load()
