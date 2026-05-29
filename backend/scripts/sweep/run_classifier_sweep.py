"""Classifier-only OFAT sweep on the labeled crop set.

The classifier stage runs on already-saved crop files; it doesn't depend
on which YOLO config produced them. So we can sweep `IN_RANGE_THRESHOLD`
(and any future classifier-stage knob) cheaply, on the FULL 942-crop
labeled dataset, without re-decoding any video.

Metrics:

  * nab_rejection_rate: fraction of NAB-labeled crops the classifier
    correctly rejects (returns []). Higher = better.
  * real_top1_accuracy: of real-species crops, fraction where the
    classifier's top-1 matches the GT species. Higher = better.
  * real_top5_accuracy: same, top-5.
  * real_rejection_rate: fraction of real-species crops the classifier
    wrongly rejects (returns []). Lower = better — this is the cost of
    over-suppression.

Run from backend/:
    python -m scripts.sweep.run_classifier_sweep
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_SWEEP_DIR = Path(__file__).resolve().parent
_SWEEP_DATA = _SWEEP_DIR / "data"
_RESULTS = _SWEEP_DIR / "results"

NAB = "Not a bird"
UNKNOWN = "Unknown bird"


@dataclass
class CrossTab:
    threshold: float
    n_nab: int = 0
    n_real: int = 0
    n_unknown: int = 0
    n_skipped: int = 0           # crop file missing
    nab_rejected: int = 0
    real_rejected: int = 0
    real_top1_correct: int = 0
    real_top5_correct: int = 0
    per_species: dict = field(default_factory=lambda: Counter())
    per_species_top1: dict = field(default_factory=lambda: Counter())

    @property
    def nab_rejection_rate(self) -> float:
        return self.nab_rejected / self.n_nab if self.n_nab else 0.0

    @property
    def real_rejection_rate(self) -> float:
        return self.real_rejected / self.n_real if self.n_real else 0.0

    @property
    def real_top1_acc(self) -> float:
        # Among real-species crops the classifier accepted (didn't reject).
        accepted = self.n_real - self.real_rejected
        return self.real_top1_correct / accepted if accepted else 0.0

    @property
    def real_top5_acc(self) -> float:
        accepted = self.n_real - self.real_rejected
        return self.real_top5_correct / accepted if accepted else 0.0


def evaluate(thresholds: list[float]) -> list[CrossTab]:
    from pipeline import classify

    manifest = json.loads((_SWEEP_DATA / "manifest.json").read_text())
    gt_list = manifest["crops_only_ground_truth"]
    logging.info("Crops to evaluate: %d", len(gt_list))

    # Pre-load all crops into memory ONCE — they're tiny (~100 KB each).
    crops = []
    for gt in gt_list:
        crop_path = _SWEEP_DATA / gt["crop_path"]
        if not crop_path.exists():
            continue
        img = cv2.imread(str(crop_path))
        if img is None or img.size == 0:
            continue
        crops.append((img, gt["ground_truth_species"]))
    logging.info("Loaded %d crops with valid image data", len(crops))

    results = []
    saved_threshold = classify.IN_RANGE_THRESHOLD
    try:
        for thresh in thresholds:
            classify.IN_RANGE_THRESHOLD = thresh
            x = CrossTab(threshold=thresh)
            for img, gt_species in crops:
                if gt_species == NAB:
                    x.n_nab += 1
                elif gt_species == UNKNOWN:
                    x.n_unknown += 1
                else:
                    x.n_real += 1
                    x.per_species[gt_species] += 1

                preds = classify.classify_bird(img)
                if not preds:  # classifier rejected
                    if gt_species == NAB:
                        x.nab_rejected += 1
                    elif gt_species != UNKNOWN:
                        x.real_rejected += 1
                    continue

                # Got predictions. For real-species crops, score top-1/top-5.
                if gt_species not in (NAB, UNKNOWN):
                    top_names = [p.species for p in preds]
                    if top_names[0] == gt_species:
                        x.real_top1_correct += 1
                        x.per_species_top1[gt_species] += 1
                    if gt_species in top_names:
                        x.real_top5_correct += 1
            results.append(x)
            logging.info("threshold=%.3f → NAB-reject %d/%d (%.0f%%); real top-1 %d/%d (%.0f%%)",
                         thresh, x.nab_rejected, x.n_nab, 100*x.nab_rejection_rate,
                         x.real_top1_correct, x.n_real, 100*x.real_top1_acc)
    finally:
        classify.IN_RANGE_THRESHOLD = saved_threshold

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _RESULTS.mkdir(exist_ok=True)
    thresholds = [0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30]
    results = evaluate(thresholds)

    out = []
    for r in results:
        out.append({
            "threshold": r.threshold,
            "n_nab": r.n_nab,
            "n_real": r.n_real,
            "n_unknown": r.n_unknown,
            "nab_rejected": r.nab_rejected,
            "real_rejected": r.real_rejected,
            "real_top1_correct": r.real_top1_correct,
            "real_top5_correct": r.real_top5_correct,
            "nab_rejection_rate": r.nab_rejection_rate,
            "real_rejection_rate": r.real_rejection_rate,
            "real_top1_acc": r.real_top1_acc,
            "real_top5_acc": r.real_top5_acc,
            "per_species_top1": dict(r.per_species_top1),
            "per_species_count": dict(r.per_species),
        })
    out_path = _RESULTS / "classifier_in_range_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    logging.info("Wrote %s", out_path)

    # Pretty table.
    print()
    print(f"{'thresh':<8}{'NAB rej':<14}{'real rej':<14}{'top-1 acc':<12}{'top-5 acc':<12}")
    for r in results:
        print(f"{r.threshold:<8.3f}{r.nab_rejected:>4}/{r.n_nab:<8} ({100*r.nab_rejection_rate:>3.0f}%)  "
              f"{r.real_rejected:>3}/{r.n_real:<3} ({100*r.real_rejection_rate:>3.0f}%)  "
              f"{r.real_top1_correct:>3}/{r.n_real - r.real_rejected:<3} ({100*r.real_top1_acc:>4.1f}%)  "
              f"{r.real_top5_correct:>3}/{r.n_real - r.real_rejected:<3} ({100*r.real_top5_acc:>4.1f}%)")


if __name__ == "__main__":
    main()
