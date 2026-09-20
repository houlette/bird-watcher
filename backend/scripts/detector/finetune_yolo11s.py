"""Step 5: Fine-tune single-class YOLO11s on yard training data (DETECTOR_PLAN.md).

Trains on the curated archive dataset (18,200 training tiles, 848 val tiles).
Uses Apple Silicon Metal (MPS) acceleration.
Exports the resulting best model to OpenVINO with dynamic shapes for Gate 5 evaluation.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11s on Yard Dataset")
    parser.add_argument("--data", type=Path, default=Path("backend/data/dataset_yolo11s/data.yaml"))
    parser.add_argument("--weights", type=str, default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("backend/data/finetune_runs"))
    parser.add_argument("--name", type=str, default="yolo11s_yard")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Dataset YAML %s not found. Make sure dataset is extracted first.", args.data)
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Starting fine-tune with device=%s, weights=%s, epochs=%d, batch=%d, workers=%d, mosaic=%.1f, cls=%.2f",
             device, args.weights, args.epochs, args.batch, args.workers, args.mosaic, args.cls)

    # Initialize YOLO11s
    model = YOLO(args.weights)

    train_kwargs = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(args.project.resolve()),
        name=args.name,
        single_cls=True,  # Class 0: bird
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
        workers=args.workers,
        mosaic=args.mosaic,
        cls=args.cls,
        exist_ok=True,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs

    # Train
    results = model.train(**train_kwargs)

    best_weights = args.project / args.name / "weights" / "best.pt"
    log.info("Training complete! Best weights saved at: %s", best_weights)

    if best_weights.exists():
        log.info("Exporting fine-tuned model to OpenVINO with dynamic shapes...")
        finetuned_model = YOLO(str(best_weights))
        ov_path = finetuned_model.export(format="openvino", dynamic=True, imgsz=args.imgsz)
        log.info("Fine-tuned OpenVINO model exported to: %s", ov_path)

        # Calibrate confidence threshold on validation split
        log.info("Calibrating confidence threshold on validation split...")
        import json
        import numpy as np

        val_metrics = finetuned_model.val(
            data=str(args.data.resolve()),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            plots=True,
        )

        r_curve = val_metrics.box.r_curve
        p_curve = val_metrics.box.p_curve
        f1_curve = val_metrics.box.f1_curve

        calib_data = {
            "mAP50": float(val_metrics.box.map50),
            "mAP50_95": float(val_metrics.box.map),
            "precision": float(val_metrics.box.mp),
            "recall": float(val_metrics.box.mr),
            "thresholds": {},
        }

        # Check precision & recall across candidate thresholds
        best_f1 = 0.0
        best_thresh = 0.25
        for conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            idx = min(int(conf * len(r_curve[0])), len(r_curve[0]) - 1)
            rec = float(r_curve[0][idx])
            prec = float(p_curve[0][idx])
            f1 = 2 * prec * rec / (prec + rec + 1e-6)
            calib_data["thresholds"][f"{conf:.2f}"] = {
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
            }
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = conf

        calib_data["optimal_conf"] = best_thresh
        calib_data["optimal_f1"] = round(best_f1, 4)

        calib_file = args.project / args.name / "calibration.json"
        calib_file.write_text(json.dumps(calib_data, indent=2))
        log.info("Validation calibration saved to %s: optimal conf=%.2f (F1=%.4f, mAP50=%.4f)",
                 calib_file, best_thresh, best_f1, val_metrics.box.map50)


if __name__ == "__main__":
    main()
