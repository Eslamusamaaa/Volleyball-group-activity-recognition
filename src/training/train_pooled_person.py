"""B2 and B3 (steps B+C): pool per-player single-frame crop features, then
classify the pooled representation into the 8-way scene class.

B2: --backbone-mode frozen, no --backbone-checkpoint
    (plain ImageNet ResNet50, never fine-tuned on this dataset at all)
B3: --backbone-mode finetune --backbone-checkpoint outputs/checkpoints/person_finetune_b3a.pt
    (ResNet50 already fine-tuned on person action classification, B3-A)

In both cases the backbone itself is NOT trained further here — it is only
used to extract a 2048-d feature per player crop (frozen at this stage,
matching the paper's "max pool features -> train NN classifier" step C).
Only the small pooled classifier head is trained.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.annotations import load_or_cache, split_keys, SCENE_CLASSES, EXPECTED_SCENE_COUNTS
from src.data.datasets import ClipPersonCropsDataset, MAX_PLAYERS, TRAIN_TRANSFORM, EVAL_TRANSFORM
from src.models.backbone import ResNetBackbone, FEATURE_DIM
from src.models.temporal import masked_max_pool
from src.models.heads import SceneHead
from src.evaluation.metrics import format_report, accuracy
from src.training.train_backbone import class_weights


class PooledPersonClassifier(nn.Module):
    def __init__(self, backbone_checkpoint, backbone_mode, device):
        super().__init__()
        if backbone_checkpoint:
            ckpt = torch.load(backbone_checkpoint, map_location=device, weights_only=False)
            num_classes = len(ckpt["classes"])
            self.backbone = ResNetBackbone(num_classes=num_classes, mode="finetune")
            self.backbone.load_state_dict(ckpt["model"])
        else:
            self.backbone = ResNetBackbone(num_classes=1, mode="frozen", pretrained=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = SceneHead(FEATURE_DIM, num_classes=len(SCENE_CLASSES))
        self.backbone_mode = backbone_mode

    def forward(self, crops, mask):
        B, P, C, H, W = crops.shape
        flat = crops.reshape(B * P, C, H, W)
        with torch.no_grad():
            feats = self.backbone.extract_features(flat)
        feats = feats.reshape(B, P, -1)
        pooled = masked_max_pool(feats, mask)
        return self.head(pooled)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.head.train() if train else model.head.eval()
    total_loss, all_true, all_pred = 0.0, [], []
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for crops, mask, label in pbar:
        crops, mask, label = crops.to(device, non_blocking=True), mask.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(crops, mask)
                loss = criterion(logits, label)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * label.size(0)
        all_true.extend(label.cpu().tolist())
        all_pred.extend(logits.argmax(1).cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / len(loader.dataset), all_true, all_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-checkpoint", default=None,
                     help="fine-tuned person-action backbone (B3-A) for B3; omit for B2 (raw ImageNet ResNet50)")
    ap.add_argument("--backbone-mode", choices=["frozen", "finetune"], default="frozen",
                     help="label only, used in the saved result name/metadata")
    ap.add_argument("--root", default=r"E:\volleyball datasets\volleyball_\videos")
    ap.add_argument("--cache", default=r"E:\volleyball datasets\outputs\annotations.pkl")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=r"E:\volleyball datasets\outputs")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = load_or_cache(args.root, args.cache)
    train_ds = ClipPersonCropsDataset(args.root, data, split_keys(data, "train"), TRAIN_TRANSFORM)
    val_ds = ClipPersonCropsDataset(args.root, data, split_keys(data, "val"), EVAL_TRANSFORM)
    test_ds = ClipPersonCropsDataset(args.root, data, split_keys(data, "test"), EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = PooledPersonClassifier(args.backbone_checkpoint, args.backbone_mode, device).to(device)
    weights = class_weights(EXPECTED_SCENE_COUNTS, SCENE_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoints", f"{args.name}.pt")
    log, best_val_acc = [], -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_t, tr_p = run_epoch(model, train_loader, device, criterion, optimizer, scaler,
                                         desc=f"epoch {epoch}/{args.epochs} train")
        va_loss, va_t, va_p = run_epoch(model, val_loader, device, criterion, None, None,
                                         desc=f"epoch {epoch}/{args.epochs} val")
        tr_acc, va_acc = accuracy(tr_t, tr_p), accuracy(va_t, va_p)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss {tr_loss:.4f} acc {tr_acc:.3f}  "
              f"val_loss {va_loss:.4f} acc {va_acc:.3f}  ({dt:.1f}s)")
        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"head": model.head.state_dict(), "val_acc": va_acc}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.head.load_state_dict(ckpt["head"])
    _, te_t, te_p = run_epoch(model, test_loader, device, criterion, None, desc="test")
    report = format_report(te_t, te_p, len(SCENE_CLASSES), SCENE_CLASSES)
    print("\n=== TEST (best val checkpoint) ===")
    print(report)

    with open(os.path.join(args.out_dir, "results", f"{args.name}.json"), "w") as f:
        json.dump({"name": args.name, "backbone_mode": args.backbone_mode, "epochs": args.epochs,
                    "best_val_acc": best_val_acc, "test_acc": accuracy(te_t, te_p), "log": log}, f, indent=2)
    with open(os.path.join(args.out_dir, "results", f"{args.name}_test_report.txt"), "w") as f:
        f.write(report)
    print(f"\nSaved checkpoint to {ckpt_path}, results to outputs/results/{args.name}.json")


if __name__ == "__main__":
    main()
