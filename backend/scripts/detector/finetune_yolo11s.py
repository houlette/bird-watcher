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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--project", type=Path, default=Path("backend/data/finetune_runs"))
    parser.add_argument("--name", type=str, default="yolo11s_yard")
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Dataset YAML %s not found. Make sure dataset is extracted first.", args.data)
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Starting fine-tune with device=%s, weights=%s, epochs=%d, batch=%d",
             device, args.weights, args.epochs, args.batch)

    # Initialize YOLO11s
    model = YOLO(args.weights)

    # Train
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(args.project.resolve()),
        name=args.name,
        single_cls=True,  # Class 0: bird
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        workers=4,
    )

    best_weights = args.project / args.name / "weights" / "best.pt"
    log.info("Training complete! Best weights saved at: %s", best_weights)

    if best_weights.exists():
        log.info("Exporting fine-tuned model to OpenVINO with dynamic shapes...")
        finetuned_model = YOLO(str(best_weights))
        ov_path = finetuned_model.export(format="openvino", dynamic=True, imgsz=args.imgsz)
        log.info("Fine-tuned OpenVINO model exported to: %s", ov_path)


if __name__ == "__main__":
    main()
