"""Compare three classifier allow-list policies on the labeled crop set.

The IN_RANGE_THRESHOLD sweep showed the classifier rejects 88-94 % of
real-species crops at any threshold — the in-range mass is ~zero for
the rejected ones, meaning the model has 0 probability on any of the 31
allow-listed species. This script tests whether broadening the allow-
list (or removing it) recovers those rejected real-bird crops.

Three modes:
  - yard:      current production setup (31 species, from yard calibration
               or the eastern-NA-backyard hand-coded fallback)
  - na_broad:  all ~200 species in na_birds.NA_BIRD_SPECIES
  - all:       no filter; any of the 525 model classes can pass

Runs the model ONCE per crop and applies all three filters to the same
output — so the runtime is the same as a single classifier sweep
(~30 s of inference plus model load).
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_SWEEP_DIR = Path(__file__).resolve().parent
_SWEEP_DATA = _SWEEP_DIR / "data"
_RESULTS = _SWEEP_DIR / "results"

NAB = "Not a bird"
UNKNOWN = "Unknown bird"
THRESHOLD = 0.10           # production current; not tuned in this experiment
TOP_K = 5


def _build_mask(id2label: dict, species_set: set[str]) -> np.ndarray:
    """Boolean mask over the model's 525 classes for the given species set."""
    from pipeline.classify import _hyphen_insensitive
    target_norm = {_hyphen_insensitive(s) for s in species_set}
    mask = np.zeros(len(id2label), dtype=bool)
    for idx, label in id2label.items():
        if _hyphen_insensitive(label) in target_norm:
            mask[int(idx)] = True
    return mask


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("allowlist")

    from pipeline import classify
    from pipeline.classify import _normalize_for_display, NA_BACKYARD_ALLOWLIST
    from na_birds import NA_BIRD_SPECIES

    manifest = json.loads((_SWEEP_DATA / "manifest.json").read_text())
    gt_list = manifest["crops_only_ground_truth"]
    log.info("Crops to evaluate: %d", len(gt_list))

    # Pre-load crops.
    crops = []
    for gt in gt_list:
        crop_path = _SWEEP_DATA / gt["crop_path"]
        if not crop_path.exists():
            continue
        img = cv2.imread(str(crop_path))
        if img is None or img.size == 0:
            continue
        crops.append((img, gt["ground_truth_species"]))
    log.info("Loaded %d valid crops", len(crops))

    # Force a model load by calling classify_bird once on a tiny dummy crop
    # (we'll discard the output). This populates the singletons.
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    _ = classify.classify_bird(dummy)
    model = classify._model
    processor = classify._processor
    id2label = model.config.id2label
    log.info("Model has %d classes", len(id2label))

    # Build the three masks.
    yard_mask = classify._allowed_mask           # production-current mask
    na_broad_mask = _build_mask(id2label, set(NA_BIRD_SPECIES))
    all_mask = np.ones(len(id2label), dtype=bool)

    log.info("Yard mask:     %d / %d species allowed", int(yard_mask.sum()), len(id2label))
    log.info("NA-broad mask: %d / %d species allowed", int(na_broad_mask.sum()), len(id2label))
    log.info("All mask:      %d / %d species allowed", int(all_mask.sum()), len(id2label))

    # Per-mode tally.
    modes = {"yard": yard_mask, "na_broad": na_broad_mask, "all": all_mask}
    stats = {name: {
        "n_nab": 0, "n_real": 0, "n_unknown": 0,
        "nab_rejected": 0, "real_rejected": 0,
        "real_top1_correct": 0, "real_top5_correct": 0,
        "per_species_top1": Counter(),
        "per_species_count": Counter(),
    } for name in modes}

    # Run inference once per crop; apply all three filters.
    import torch
    for i, (img, gt_species) in enumerate(crops, 1):
        rgb = img[:, :, ::-1].copy()
        inputs = processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for name, mask in modes.items():
            s = stats[name]
            if gt_species == NAB:
                s["n_nab"] += 1
            elif gt_species == UNKNOWN:
                s["n_unknown"] += 1
            else:
                s["n_real"] += 1
                s["per_species_count"][gt_species] += 1

            in_range_mass = float(probs[mask].sum())
            if in_range_mass < THRESHOLD:
                if gt_species == NAB:
                    s["nab_rejected"] += 1
                elif gt_species != UNKNOWN:
                    s["real_rejected"] += 1
                continue

            filtered = np.where(mask, probs, 0.0)
            filtered = filtered / filtered.sum()
            top_idxs = np.argsort(filtered)[::-1][:TOP_K]
            top_names = [_normalize_for_display(id2label[int(idx)]) for idx in top_idxs]

            if gt_species not in (NAB, UNKNOWN):
                if top_names[0] == gt_species:
                    s["real_top1_correct"] += 1
                    s["per_species_top1"][gt_species] += 1
                if gt_species in top_names:
                    s["real_top5_correct"] += 1

        if i % 200 == 0:
            log.info("  %d/%d", i, len(crops))

    # Pretty print + write JSON.
    print()
    print(f"{'mode':<10}{'NAB rej':<22}{'real top-1':<20}{'real top-5':<20}{'real rej':<22}")
    out = {}
    for name, s in stats.items():
        accepted = s["n_real"] - s["real_rejected"]
        top1_acc = s["real_top1_correct"] / accepted if accepted else 0.0
        top5_acc = s["real_top5_correct"] / accepted if accepted else 0.0
        nab_rej_rate = s["nab_rejected"] / s["n_nab"] if s["n_nab"] else 0.0
        real_rej_rate = s["real_rejected"] / s["n_real"] if s["n_real"] else 0.0
        print(f"{name:<10}"
              f"{s['nab_rejected']}/{s['n_nab']:<6} ({100*nab_rej_rate:>3.0f}%)   "
              f"{s['real_top1_correct']}/{accepted:<3} ({100*top1_acc:>4.1f}%)     "
              f"{s['real_top5_correct']}/{accepted:<3} ({100*top5_acc:>4.1f}%)     "
              f"{s['real_rejected']}/{s['n_real']} ({100*real_rej_rate:>3.0f}%)")
        out[name] = {
            "mask_size": int(modes[name].sum()),
            "n_nab": s["n_nab"], "n_real": s["n_real"], "n_unknown": s["n_unknown"],
            "nab_rejected": s["nab_rejected"], "nab_rejection_rate": nab_rej_rate,
            "real_rejected": s["real_rejected"], "real_rejection_rate": real_rej_rate,
            "real_top1_correct": s["real_top1_correct"], "real_top1_acc": top1_acc,
            "real_top5_correct": s["real_top5_correct"], "real_top5_acc": top5_acc,
            "per_species_count": dict(s["per_species_count"]),
            "per_species_top1": dict(s["per_species_top1"]),
        }

    _RESULTS.mkdir(exist_ok=True)
    out_path = _RESULTS / "allowlist_experiment.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
