#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SWLLSEDataset
from src.data.features import FeatureSpec
from src.models.tcn import GRUBaseline, TCNClassifier
from src.utils.io import ensure_dir, save_json
from src.utils.metrics import macro_f1, top1_accuracy
from src.utils.train_utils import seed_everything


def build_model(name: str, input_dim: int, num_classes: int) -> nn.Module:
    if name == "gru":
        return GRUBaseline(input_dim=input_dim, num_classes=num_classes)
    return TCNClassifier(input_dim=input_dim, num_classes=num_classes)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    loss_sum = 0.0
    ys, ps = [], []

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    ys = np.concatenate(ys) if ys else np.array([])
    ps = np.concatenate(ps) if ps else np.array([])
    mean_loss = loss_sum / max(len(loader.dataset), 1)
    return mean_loss, top1_accuracy(ys, ps), macro_f1(ys, ps), ys, ps


def save_confusion_matrix(y_true, y_pred, out_path: Path, num_classes: int = 300):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("."))
    ap.add_argument("--mediapipe_dir", type=Path, required=True)
    ap.add_argument("--splits_dir", type=Path, required=True)
    ap.add_argument("--annotations_csv", type=Path, required=True)
    ap.add_argument("--feature_spec", type=Path, default=Path("artifacts/feature_spec.json"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--t_fixed", type=int, default=None)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["tcn", "gru"], default="tcn")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.feature_spec.exists():
        spec = FeatureSpec.from_json(args.feature_spec)
    else:
        spec = FeatureSpec.default(t_fixed=args.t_fixed or 128)
    if args.t_fixed is not None:
        spec.t_fixed = args.t_fixed

    artifacts_dir = ensure_dir("artifacts")
    checkpoints_dir = ensure_dir("checkpoints")
    logs_dir = ensure_dir("logs")

    # Label map
    ann = pd.read_csv(args.annotations_csv)
    if not {"CLASS_ID", "LABEL"}.issubset(ann.columns):
        raise ValueError("annotations CSV debe tener columnas CLASS_ID y LABEL")
    label_map = {int(r.CLASS_ID): str(r.LABEL) for _, r in ann.iterrows()}
    save_json({str(k): v for k, v in label_map.items()}, artifacts_dir / "label_map.json")
    spec.to_json(artifacts_dir / "feature_spec.json")

    train_ds = SWLLSEDataset(args.splits_dir / "train.csv", args.mediapipe_dir, spec, augment=True, strict_missing=False)
    val_ds = SWLLSEDataset(args.splits_dir / "val.csv", args.mediapipe_dir, spec, augment=False, strict_missing=False)
    test_ds = SWLLSEDataset(args.splits_dir / "test.csv", args.mediapipe_dir, spec, augment=False, strict_missing=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, input_dim=spec.d_frame, num_classes=300).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    bad_epochs = 0
    history_path = logs_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr"])

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
            va_loss, va_acc, va_f1, _, _ = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow([epoch, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1, lr])
            f.flush()
            print(f"Epoch {epoch:03d} | tr_acc={tr_acc:.4f} va_acc={va_acc:.4f} va_f1={va_f1:.4f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                bad_epochs = 0
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "feature_spec": spec.__dict__,
                    "best_val_acc": best_val_acc,
                }
                torch.save(ckpt, checkpoints_dir / "best.pt")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print("Early stopping triggered")
                    break

    # Test with best checkpoint
    ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc, te_f1, y_true, y_pred = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    print(f"TEST | loss={te_loss:.4f} acc={te_acc:.4f} macro_f1={te_f1:.4f}")

    save_confusion_matrix(y_true, y_pred, artifacts_dir / "confusion_matrix.png", num_classes=300)


if __name__ == "__main__":
    main()
