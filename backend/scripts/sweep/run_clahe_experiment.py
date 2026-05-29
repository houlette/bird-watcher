"""A/B test CLAHE pre-processing on the classifier input.

CLAHE (Contrast-Limited Adaptive Histogram Equalization) brings out
detail in shadowed regions without blowing out highlights, and is cheap
(~1 ms per crop on CPU). The hypothesis is that for our crops — which
often capture a small bird against a brighter background or under harsh
shadow — CLAHE'd input is closer to the gpiosenka training distribution
than our raw crops are.

Risk: the classifier was trained on un-pre-processed images. If our raw
crops happen to be close enough to training-distribution that CLAHE
shifts them OUT, accuracy could drop. This script measures the actual
delta on the labeled crop set before we commit to changing production.

For each of the 942 user-labeled crops:
  1. Run the model on the raw BGR (the current production input).
  2. Run the model on the CLAHE'd BGR (applied to L channel of LAB so
     colors don't shift, then converted back to BGR).
  3. Score both against the user's ground-truth label using the
     existing yard ∪ NA_BROAD allow-list + IN_RANGE_THRESHOLD policy.

Reports NAB rejection rate, real top-1, real top-5, and real rejection
rate for both. We ship CLAHE iff it holds or improves on every metric.
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
THRESHOLD = 0.10
TOP_K = 5

# CLAHE parameters. clipLimit=2.0 is a moderate contrast boost; 8×8 tiles
# is the OpenCV default and a good fit for our ~200-400 px crops.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB, preserving color.

    Converting to LAB and only equalizing L means the chroma stays put — a
    direct CLAHE on each BGR channel independently shifts colors.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("clahe")

    from pipeline import classify
    from pipeline.classify import _normalize_for_display

    manifest = json.loads((_SWEEP_DATA / "manifest.json").read_text())
    gt_list = manifest["crops_only_ground_truth"]

    crops = []
    for gt in gt_list:
        p = _SWEEP_DATA / gt["crop_path"]
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            continue
        crops.append((img, gt["ground_truth_species"]))
    log.info("Loaded %d crops", len(crops))

    # Warm up the classifier so its singletons are populated.
    _ = classify.classify_bird(np.zeros((10, 10, 3), dtype=np.uint8))
    model, processor = classify._model, classify._processor
    mask = classify._allowed_mask
    id2label = model.config.id2label

    import torch

    def evaluate(label: str, transform) -> dict:
        s = {
            "n_nab": 0, "n_real": 0, "n_unknown": 0,
            "nab_rejected": 0, "real_rejected": 0,
            "real_top1_correct": 0, "real_top5_correct": 0,
            "per_species_count": Counter(),
            "per_species_top1": Counter(),
        }
        for i, (img, gt_species) in enumerate(crops, 1):
            crop = transform(img)
            rgb = crop[:, :, ::-1].copy()
            inputs = processor(images=rgb, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            if gt_species == NAB:
                s["n_nab"] += 1
            elif gt_species == UNKNOWN:
                s["n_unknown"] += 1
            else:
                s["n_real"] += 1
                s["per_species_count"][gt_species] += 1

            in_range = float(probs[mask].sum())
            if in_range < THRESHOLD:
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
                log.info("  %s: %d/%d", label, i, len(crops))
        accepted = s["n_real"] - s["real_rejected"]
        return {
            **{k: v for k, v in s.items() if not isinstance(v, Counter)},
            "per_species_count": dict(s["per_species_count"]),
            "per_species_top1": dict(s["per_species_top1"]),
            "nab_rejection_rate": s["nab_rejected"] / s["n_nab"] if s["n_nab"] else 0,
            "real_rejection_rate": s["real_rejected"] / s["n_real"] if s["n_real"] else 0,
            "real_top1_acc": s["real_top1_correct"] / accepted if accepted else 0,
            "real_top5_acc": s["real_top5_correct"] / accepted if accepted else 0,
        }

    log.info("=== Arm A: raw (current production) ===")
    raw = evaluate("raw", lambda x: x)
    log.info("=== Arm B: CLAHE on L channel ===")
    clahe = evaluate("clahe", apply_clahe)

    # Pretty table.
    print()
    print(f"{'arm':<10}{'NAB rej (↑)':<18}{'real top-1 (↑)':<22}{'real top-5 (↑)':<22}{'real rej (↓)':<18}")
    for name, r in [("raw", raw), ("clahe", clahe)]:
        accepted = r["n_real"] - r["real_rejected"]
        print(f"{name:<10}"
              f"{r['nab_rejected']}/{r['n_nab']:<6}({100*r['nab_rejection_rate']:>3.0f}%)   "
              f"{r['real_top1_correct']}/{accepted:<3}({100*r['real_top1_acc']:>5.1f}%)       "
              f"{r['real_top5_correct']}/{accepted:<3}({100*r['real_top5_acc']:>5.1f}%)       "
              f"{r['real_rejected']}/{r['n_real']} ({100*r['real_rejection_rate']:>3.0f}%)")
    # Per-metric delta
    print()
    print("Delta (clahe − raw):")
    print(f"  NAB rejection rate:   {100*(clahe['nab_rejection_rate'] - raw['nab_rejection_rate']):+.1f}pp")
    print(f"  Real top-1 accuracy:  {100*(clahe['real_top1_acc'] - raw['real_top1_acc']):+.1f}pp")
    print(f"  Real top-5 accuracy:  {100*(clahe['real_top5_acc'] - raw['real_top5_acc']):+.1f}pp")
    print(f"  Real rejection rate:  {100*(clahe['real_rejection_rate'] - raw['real_rejection_rate']):+.1f}pp")

    _RESULTS.mkdir(exist_ok=True)
    (_RESULTS / "clahe_experiment.json").write_text(json.dumps({"raw": raw, "clahe": clahe}, indent=2))
    print(f"\nWrote {_RESULTS / 'clahe_experiment.json'}")


if __name__ == "__main__":
    sys.exit(main())
