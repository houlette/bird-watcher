"""A/B benchmark: current classifier (dennisjooo EfficientNetB2) vs
candidate (prithivMLmods SigLIP-2) on the user-labeled corrections.

Runs both models on every Detection that has a Correction (i.e. the user
verified its true label). Reports per-species top-1 / top-5 accuracy and
'not a bird' rejection rates so we can see whether the SigLIP-2 swap is
worth doing for the live pipeline.

Both models go through the same allow-list filter and IN_RANGE_THRESHOLD
as production — the only thing varying is the model weights.

Usage:
    docker compose exec api python scripts/benchmark_classifiers.py
    # or override the candidate:
    docker compose exec api python scripts/benchmark_classifiers.py \\
        --candidate chriamue/bird-species-classifier
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from db.models import NOT_A_BIRD_LABEL, Correction, Detection, Species
from db.session import SessionLocal, init_db
from pipeline import calibration
from pipeline.classify import (
    IN_RANGE_THRESHOLD,
    NA_BACKYARD_ALLOWLIST,
    _hyphen_insensitive,
    _normalize_for_display,
)

DATA_DIR = Path("/app/data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")


def get_allowlist() -> set[str]:
    """Mirror of classify._load()'s allow-list source selection."""
    cal = calibration.get_allowlist()
    return cal if cal else set(NA_BACKYARD_ALLOWLIST)


def build_mask(id2label: dict, allowlist: set[str]) -> np.ndarray:
    """Boolean mask over the model's class indices for our allow-list."""
    allowlist_norm = {_hyphen_insensitive(s) for s in allowlist}
    mask = np.zeros(len(id2label), dtype=bool)
    for idx, label in id2label.items():
        if _hyphen_insensitive(label) in allowlist_norm:
            mask[int(idx)] = True
    return mask


def predict(model, processor, mask: np.ndarray, crop_bgr: np.ndarray) -> list[tuple[str, float]]:
    """Mirror of classify.classify_bird()'s filter logic.

    Returns top-5 (base_species, probability) pairs, OR an empty list if
    the model rejected the crop (in-range mass < IN_RANGE_THRESHOLD).
    """
    import torch

    rgb = crop_bgr[:, :, ::-1].copy()
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    in_range_mass = float(probs[mask].sum())
    if in_range_mass < IN_RANGE_THRESHOLD:
        return []

    filtered = np.where(mask, probs, 0.0)
    filtered = filtered / filtered.sum()

    id2label = model.config.id2label

    # Aggregate plumage variants per base species (same as classify.py).
    by_base: dict[str, float] = {}
    for idx in np.argsort(filtered)[::-1]:
        if not mask[idx] or filtered[idx] == 0:
            continue
        base = _normalize_for_display(id2label[int(idx)])
        by_base[base] = by_base.get(base, 0.0) + float(filtered[idx])
        if len(by_base) >= 10:  # keep loop bounded
            break

    return sorted(by_base.items(), key=lambda kv: -kv[1])[:5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default="dennisjooo/Birds-Classifier-EfficientNetB2")
    ap.add_argument("--candidate", default="prithivMLmods/Bird-Species-Classifier-526")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    init_db()
    db = SessionLocal()

    log.info("Loading labeled corrections from DB…")
    rows = (
        db.query(Correction, Detection, Species)
        .join(Detection, Correction.detection_id == Detection.id)
        .join(Species, Correction.correct_species_id == Species.id)
        .all()
    )
    log.info("Found %d corrections", len(rows))

    by_species: dict[str, int] = defaultdict(int)
    for _, _, sp in rows:
        by_species[sp.common_name] += 1
    log.info("Corrections per species:")
    for name, n in sorted(by_species.items(), key=lambda kv: -kv[1]):
        log.info("  %-30s %d", name, n)

    allowlist = get_allowlist()
    log.info("Allow-list: %d species", len(allowlist))

    log.info("=" * 70)
    log.info("Loading models…")
    log.info("  current:   %s", args.current)
    cur_model = AutoModelForImageClassification.from_pretrained(args.current)
    cur_processor = AutoImageProcessor.from_pretrained(args.current)
    cur_model.eval()
    cur_mask = build_mask(cur_model.config.id2label, allowlist)
    log.info("    %d total classes, %d allow-listed", len(cur_model.config.id2label), int(cur_mask.sum()))

    log.info("  candidate: %s", args.candidate)
    cand_model = AutoModelForImageClassification.from_pretrained(args.candidate)
    cand_processor = AutoImageProcessor.from_pretrained(args.candidate)
    cand_model.eval()
    cand_mask = build_mask(cand_model.config.id2label, allowlist)
    log.info("    %d total classes, %d allow-listed", len(cand_model.config.id2label), int(cand_mask.sum()))

    # Tally:
    #   stats[model][true_label] = {top1_correct, top5_correct, rejected, total}
    stats: dict = {name: defaultdict(lambda: {"top1": 0, "top5": 0, "rej": 0, "total": 0}) for name in ("current", "candidate")}

    log.info("=" * 70)
    log.info("Running inference on %d labeled crops…", len(rows))
    skipped = 0
    for i, (_, det, sp) in enumerate(rows, 1):
        crop_path = DATA_DIR / det.crop_path
        if not crop_path.exists():
            skipped += 1
            continue
        crop = cv2.imread(str(crop_path))
        if crop is None:
            skipped += 1
            continue

        true_label = sp.common_name
        is_nab = true_label == NOT_A_BIRD_LABEL

        for tag, model, proc, mask in [
            ("current", cur_model, cur_processor, cur_mask),
            ("candidate", cand_model, cand_processor, cand_mask),
        ]:
            preds = predict(model, proc, mask, crop)
            top_names = [p[0] for p in preds]
            top1 = top_names[0] if top_names else None

            s = stats[tag][true_label]
            s["total"] += 1
            if is_nab:
                if not preds:
                    s["rej"] += 1
            else:
                if top1 == true_label:
                    s["top1"] += 1
                if true_label in top_names:
                    s["top5"] += 1

        if i % 25 == 0:
            log.info("  …%d/%d", i, len(rows))

    log.info("=" * 70)
    log.info("Results (skipped %d / %d due to missing crops)", skipped, len(rows))
    log.info("=" * 70)

    for tag in ("current", "candidate"):
        repo = args.current if tag == "current" else args.candidate
        log.info("")
        log.info("%s [%s]", tag.upper(), repo)
        # Per-species rows
        real_total = real_top1 = real_top5 = 0
        for label, s in sorted(stats[tag].items(), key=lambda kv: -kv[1]["total"]):
            if label == NOT_A_BIRD_LABEL:
                pct = 100 * s["rej"] / max(s["total"], 1)
                log.info("  %-30s rejected %d/%d (%.0f%%)", label, s["rej"], s["total"], pct)
            else:
                real_total += s["total"]
                real_top1 += s["top1"]
                real_top5 += s["top5"]
                log.info(
                    "  %-30s top-1 %d/%d   top-5 %d/%d",
                    label, s["top1"], s["total"], s["top5"], s["total"],
                )
        if real_total:
            log.info(
                "  %-30s top-1 %d/%d (%.0f%%)   top-5 %d/%d (%.0f%%)",
                "TOTAL real species",
                real_top1, real_total, 100 * real_top1 / real_total,
                real_top5, real_total, 100 * real_top5 / real_total,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
