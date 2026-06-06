"""Fine-tune a binary bird/not-a-bird post-filter.

Architecture: same EfficientNet-B2 backbone the species classifier uses
(so the deployed runtime already has the weights cached locally) with a
fresh 2-way head. Trained at low LR on the binary export from
`export_binary_dataset.py`.

Deployment is *additive* — this classifier runs AFTER the species
classifier and overrides species → "Not a bird" when it's confident NAB.
The species classifier keeps the species ID job; this one just cleans
up the false-positive tail YOLO leaks through.

Usage:
    cd backend
    python scripts/train/finetune_binary.py \\
        --data /tmp/birdwatcher/binary_dataset \\
        --out /tmp/birdwatcher/binary_filter

Outputs an HF-format directory. Wire into production by setting
BIRD_BINARY_FILTER_MODEL=<path> in backend/.env (separate from
BIRD_CLASSIFIER_MODEL — they're different roles).
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
log = logging.getLogger("finetune_binary")

BASE_MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2"


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = _device()
    log.info("Using device: %s", device)

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    img_size = processor.size["height"] if isinstance(processor.size, dict) else 260
    mean = processor.image_mean
    std = processor.image_std

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 24, img_size + 24)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        # Heavier aug than the species head: bird/not-bird is much more
        # invariant to color/light shifts.
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20),
        transforms.RandomRotation(8),
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
    assert train_ds.classes == val_ds.classes == ["bird", "not_a_bird"], \
        f"unexpected classes: {train_ds.classes}"
    log.info("Train: %d  Val: %d", len(train_ds), len(val_ds))
    log.info("Class counts (train): %s",
             {train_ds.classes[c]: n for c, n in Counter(train_ds.targets).items()})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Class-weighted loss in case NAB still dominates (typically yes,
    # ~1.7× the bird count).
    counts = Counter(train_ds.targets)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (2 * counts[i]) for i in range(2)],
        dtype=torch.float32, device=device,
    )

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "bird", 1: "not_a_bird"},
        label2id={"bird": 0, "not_a_bird": 1},
        ignore_mismatched_sizes=True,
    ).to(device)

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

    best_val_acc = 0.0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
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

        model.eval()
        # Confusion matrix: TP/FP/TN/FN for the "is bird" call. We care
        # most about FP (NAB classified as bird) — that's the YOLO-leak
        # rate. And we want FN (real birds suppressed) to stay near 0.
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(pixel_values=imgs).logits.argmax(1)
                for p, l in zip(preds.tolist(), labels.tolist()):
                    if l == 0 and p == 0: tp += 1   # bird correctly = bird
                    elif l == 0 and p == 1: fn += 1  # bird wrongly = NAB
                    elif l == 1 and p == 1: tn += 1  # NAB correctly = NAB
                    elif l == 1 and p == 0: fp += 1  # NAB wrongly = bird (LEAK)

        val_total = tp + fp + tn + fn
        val_acc = (tp + tn) / val_total if val_total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        leak_rate = fp / (fp + tn) if (fp + tn) else 0.0  # of true NAB, what frac passes as bird
        miss_rate = fn / (fn + tp) if (fn + tp) else 0.0  # of true bird, what frac wrongly NAB
        elapsed = time.time() - t0
        log.info("epoch %d/%d  train_loss=%.3f  train_acc=%.3f  val_acc=%.3f  "
                 "leak=%.1f%%  miss=%.1f%%  %.1fs",
                 epoch, args.epochs, train_loss / max(train_total, 1),
                 train_correct / max(train_total, 1),
                 val_acc, leak_rate * 100, miss_rate * 100, elapsed)
        log.info("    confusion: TP=%d FN=%d FP=%d TN=%d", tp, fn, fp, tn)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "leak_rate": leak_rate,
            "miss_rate": miss_rate,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            (args.out / "training_meta.json").write_text(json.dumps({
                "base_model": BASE_MODEL,
                "best_val_acc": best_val_acc,
                "best_epoch": epoch,
                "classes": ["bird", "not_a_bird"],
                "history": history,
            }, indent=2))
            log.info("    ↑ saved (best val_acc=%.3f) to %s", best_val_acc, args.out)

    log.info("Done. Best val_acc=%.3f  → %s", best_val_acc, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
