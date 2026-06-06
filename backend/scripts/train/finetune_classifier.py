"""Fine-tune `dennisjooo/Birds-Classifier-EfficientNetB2` on yard data.

The base model recognizes 525 species but underperforms on this yard's
mix (Mourning Dove / Rock Pigeon confusion, false positives on ivy/
leaves). We replace its classification head with a 7-way head matching
the export_classifier_dataset.py output (NAB + the top species we have
enough samples for) and fine-tune the backbone at a low LR.

Designed to run on a single Apple-Silicon GPU (MPS) in <10 minutes for
the current dataset. Class-weighted cross-entropy compensates for the
NAB-dominated imbalance.

Usage:
    cd backend
    python scripts/train/finetune_classifier.py \\
        --data /tmp/birdwatcher/finetune_dataset \\
        --out /tmp/birdwatcher/finetuned

Outputs an HF-compatible directory at --out that the production
classifier loader (pipeline/classify.py) can swap in via the
BIRD_CLASSIFIER_MODEL env var.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("finetune_classifier")

BASE_MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path,
                    help="Dataset root with train/ and val/ subdirs (ImageFolder layout).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to save the fine-tuned model (HF format).")
    ap.add_argument("--epochs", type=int, default=8,
                    help="Number of training epochs.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Backbone LR. Head LR is 10× this (newly-initialized).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = _device()
    log.info("Using device: %s", device)

    # --- Data + transforms (match the base processor's expectations) ---
    # The HF AutoImageProcessor for EfficientNetB2 expects 260×260 RGB
    # normalized to ImageNet stats. We hand-roll equivalent torchvision
    # transforms so we can apply augmentation on top.
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    img_size = processor.size["height"] if isinstance(processor.size, dict) else 260
    mean = processor.image_mean
    std = processor.image_std

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = ImageFolder(args.data / "train", transform=train_tf)
    val_ds = ImageFolder(args.data / "val", transform=val_tf)
    assert train_ds.classes == val_ds.classes, "train/val class mismatch"
    num_classes = len(train_ds.classes)
    log.info("Classes (%d): %s", num_classes, train_ds.classes)
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Class-weighted loss to handle NAB-dominated imbalance (NAB is ~50%
    # of the set). Without this the model learns "predict NAB" and
    # tanks recall on the rare classes.
    counts = Counter(train_ds.targets)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32, device=device,
    )
    log.info("Class weights: %s", {train_ds.classes[i]: round(w.item(), 2) for i, w in enumerate(weights)})

    # --- Model: replace the head, keep the backbone ---
    # `ignore_mismatched_sizes=True` lets HF swap the 525-way head for a
    # fresh 7-way head without crashing. The backbone weights are
    # preserved. New head is randomly initialized.
    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_classes,
        id2label={i: c for i, c in enumerate(train_ds.classes)},
        label2id={c: i for i, c in enumerate(train_ds.classes)},
        ignore_mismatched_sizes=True,
    ).to(device)

    # Differential LR: backbone slow, head fast. The head is fresh and
    # needs much more learning; the backbone just needs to adapt.
    head_params = list(model.classifier.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 10},
        ],
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # --- Train loop ---
    best_val_acc = 0.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        t0 = time.time()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(pixel_values=imgs).logits
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        scheduler.step()

        # val
        model.eval()
        val_correct = 0
        val_total = 0
        per_class_correct = Counter()
        per_class_total = Counter()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(pixel_values=imgs).logits
                preds = out.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                for cls in range(num_classes):
                    mask = labels == cls
                    per_class_total[cls] += int(mask.sum().item())
                    per_class_correct[cls] += int(((preds == labels) & mask).sum().item())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        elapsed = time.time() - t0
        per_class = {
            train_ds.classes[c]: (per_class_correct[c], per_class_total[c],
                                  per_class_correct[c] / max(per_class_total[c], 1))
            for c in range(num_classes)
        }
        log.info("epoch %d/%d  train_loss=%.3f  train_acc=%.3f  val_acc=%.3f  %.1fs",
                 epoch, args.epochs, train_loss / max(train_total, 1),
                 train_acc, val_acc, elapsed)
        for name, (c, t, a) in per_class.items():
            log.info("    %-20s val %d/%d (%.0f%%)", name, c, t, a * 100)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "per_class": {k: list(v) for k, v in per_class.items()},
        })

        # Save best by val_acc.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            (args.out / "training_meta.json").write_text(json.dumps({
                "base_model": BASE_MODEL,
                "best_val_acc": best_val_acc,
                "best_epoch": epoch,
                "classes": train_ds.classes,
                "history": history,
            }, indent=2))
            log.info("    ↑ saved (best val_acc=%.3f) to %s", best_val_acc, args.out)

    log.info("Done. Best val_acc=%.3f  → %s", best_val_acc, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
