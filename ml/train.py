"""
Phase 4: Train a tampering-detection CNN via transfer learning.

Approach: MobileNetV2 pretrained on ImageNet, backbone frozen, only a new
small classifier head trained on our synthetic dataset. This is the
standard approach when your own labeled dataset is small (800 images here)
-- training a CNN from scratch on that little data would badly overfit.

Run: python3 train.py
Outputs: models/tamper_classifier.pt, plus a printed classification report
on a held-out validation split (never seen during training).
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "models", "tamper_classifier.pt")
IMG_SIZE = 128
BATCH_SIZE = 16
import argparse
import json

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "models", "_checkpoint.pt")
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "_best_checkpoint.pt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "models", "_train_log.json")
EPOCHS = 15
LR = 1e-3
SEED = 42

CLASS_NAMES = ["authentic", "tampered"]  # label 0, 1


class DocumentDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_samples():
    samples = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            samples.append((os.path.join(cls_dir, fname), cls_idx))
    return samples


class TamperCNN(nn.Module):
    """
    A compact CNN trained from scratch. No pretrained weights needed —
    this project's environment couldn't reach the ImageNet weight host
    (download.pytorch.org), and keeping the whole pipeline self-contained
    with no required external downloads is a reasonable trade for a
    small, well-defined binary task like this one anyway.

    4 conv blocks (each: conv -> batchnorm -> relu -> maxpool) followed by
    global average pooling and a small classifier head. Batchnorm + dropout
    keep this from overfitting the relatively small (800-image) dataset.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 128->64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 32->16
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 16->8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model():
    return TamperCNN()


def evaluate(model, loader, device):
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            for p, y in zip(preds.tolist(), labels.tolist()):
                if p == 1 and y == 1: tp += 1
                elif p == 0 and y == 0: tn += 1
                elif p == 1 and y == 0: fp += 1
                elif p == 0 and y == 1: fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total epochs to reach")
    parser.add_argument("--max-time", type=float, default=250, help="Stop after this many seconds (for resumable runs)")
    args = parser.parse_args()

    import time
    start_time = time.time()

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATA_MEAN, DATA_STD = 0.9554, 0.0616
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[DATA_MEAN] * 3, std=[DATA_STD] * 3),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[DATA_MEAN] * 3, std=[DATA_STD] * 3),
    ])

    samples = build_samples()
    random.shuffle(samples)
    n_val = int(0.2 * len(samples))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_ds = DocumentDataset(train_samples, train_transform)
    val_ds = DocumentDataset(val_samples, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    history = []
    best_f1 = -1.0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", -1.0)
        if os.path.exists(LOG_PATH):
            history = json.load(open(LOG_PATH))
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']}  (best_f1 so far: {best_f1:.3f})")
        print(f"Train: {len(train_samples)}  Val (held out): {len(val_samples)}  [note: re-shuffled split this run]")
    else:
        print(f"Train: {len(train_samples)}  Val (held out): {len(val_samples)}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = running_loss / len(train_ds)
        train_acc = correct / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)
        is_best = val_metrics["f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_metrics['accuracy']:.3f}  val_f1={val_metrics['f1']:.3f}"
              f"{'  <- best so far, saved' if is_best else ''}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, **val_metrics})

        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_f1": best_f1}, CHECKPOINT_PATH)
        json.dump(history, open(LOG_PATH, "w"))

        if time.time() - start_time > args.max_time:
            print(f"\nTime budget ({args.max_time}s) reached after epoch {epoch}. "
                  f"Run again to continue from this checkpoint.")
            return

    print(f"\n=== Best held-out validation metrics (epoch with highest val_f1, best_f1={best_f1:.3f}) ===")
    best_epoch_record = max(history, key=lambda r: r["f1"])
    for k, v in best_epoch_record.items():
        print(f"  {k}: {v}")

    import shutil
    shutil.copy(BEST_MODEL_PATH, MODEL_OUT)
    print(f"\nSaved BEST model (by val_f1, not necessarily the final epoch) to {MODEL_OUT}")
    print(f"Reached target epoch {args.epochs} — training complete.")


if __name__ == "__main__":
    main()
