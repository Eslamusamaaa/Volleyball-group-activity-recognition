"""Phase A: fine-tune or frozen-feature-extractor training of the ResNet50 backbone.

Covers:
  B1-tuned  : --task scene  --mode finetune
  B2        : --task scene  --mode frozen   (comparison only, backbone not fine-tuned)
  B3-A      : --task person --mode finetune

Mixed precision (torch.cuda.amp) throughout to fit the 4GB RTX 3050.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.annotations import (
    load_or_cache, split_keys, SCENE_CLASSES, ACTION_CLASSES,
    EXPECTED_SCENE_COUNTS, EXPECTED_ACTION_COUNTS,
)
from src.data.datasets import SceneImageDataset, PersonCropDataset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from src.models.backbone import ResNetBackbone
from src.evaluation.metrics import format_report, accuracy


def class_weights(counts_dict, classes):
    counts = np.array([counts_dict[c] for c in classes], dtype=np.float64)
    inv = 1.0 / counts
    w = inv / inv.sum() * len(classes)
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, device, optimizer=None, scaler=None, criterion=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, all_true, all_pred = 0.0, [], []
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / len(loader.dataset), all_true, all_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["scene", "person"], required=True)
    ap.add_argument("--mode", choices=["finetune", "frozen"], default="finetune")
    ap.add_argument("--root", default=r"E:\volleyball datasets\volleyball_\videos")
    ap.add_argument("--cache", default=r"E:\volleyball datasets\outputs\annotations.pkl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=r"E:\volleyball datasets\outputs")
    ap.add_argument("--name", default=None, help="checkpoint/results basename, defaults to <task>_<mode>")
    args = ap.parse_args()

    name = args.name or f"{args.task}_{args.mode}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = load_or_cache(args.root, args.cache)
    train_keys = split_keys(data, "train")
    val_keys = split_keys(data, "val")
    test_keys = split_keys(data, "test")

    if args.task == "scene":
        classes = SCENE_CLASSES
        weights = class_weights(EXPECTED_SCENE_COUNTS, classes)
        train_ds = SceneImageDataset(args.root, data, train_keys, TRAIN_TRANSFORM)
        val_ds = SceneImageDataset(args.root, data, val_keys, EVAL_TRANSFORM)
        test_ds = SceneImageDataset(args.root, data, test_keys, EVAL_TRANSFORM)
    else:
        classes = ACTION_CLASSES
        weights = class_weights(EXPECTED_ACTION_COUNTS, classes)
        train_ds = PersonCropDataset(args.root, data, train_keys, TRAIN_TRANSFORM)
        val_ds = PersonCropDataset(args.root, data, val_keys, EVAL_TRANSFORM)
        test_ds = PersonCropDataset(args.root, data, test_keys, EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ResNetBackbone(num_classes=len(classes), mode=args.mode).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoints", f"{name}.pt")
    log = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, tr_true, tr_pred = run_epoch(model, train_loader, device, optimizer, scaler, criterion,
                                                  desc=f"epoch {epoch}/{args.epochs} train")
        val_loss, va_true, va_pred = run_epoch(model, val_loader, device, None, None, criterion,
                                                desc=f"epoch {epoch}/{args.epochs} val")
        scheduler.step()
        train_acc, val_acc = accuracy(tr_true, tr_pred), accuracy(va_true, va_pred)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f}  ({dt:.1f}s)")
        log.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc, "seconds": dt})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "mode": args.mode,
                        "task": args.task, "val_acc": val_acc}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, te_true, te_pred = run_epoch(model, test_loader, device, None, None, criterion)
    report = format_report(te_true, te_pred, len(classes), classes)
    print("\n=== TEST (best val checkpoint) ===")
    print(report)

    with open(os.path.join(args.out_dir, "results", f"{name}.json"), "w") as f:
        json.dump({"name": name, "task": args.task, "mode": args.mode, "epochs": args.epochs,
                    "best_val_acc": best_val_acc, "test_acc": accuracy(te_true, te_pred),
                    "log": log}, f, indent=2)
    with open(os.path.join(args.out_dir, "results", f"{name}_test_report.txt"), "w") as f:
        f.write(report)
    print(f"\nSaved checkpoint to {ckpt_path}, results to outputs/results/{name}.json")


if __name__ == "__main__":
    main()
