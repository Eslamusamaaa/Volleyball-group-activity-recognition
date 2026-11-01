"""Phase C: train temporal models (B4-B8) on cached backbone features, plus the
mandatory multi-task extension (Section 6) on top of B8.

Baselines, all driven by --variant:
  b4  : image-level LSTM over B1 features              -> scene (8-way)
  b5  : per-player LSTM (LSTM1) over B3-A features, final-step max-pooled (no LSTM2) -> scene
  b6  : image-level LSTM, but over B3-A-derived per-frame pooled features   -> scene
  b7  : LSTM1 (person) -> max-pool all players -> LSTM2 (scene)            -> scene
  b8  : like b7 but team-separated pooling (concat(pool(team0), pool(team1)))
  b8mt: b8 + per-player action head, multi-task loss (scene + lambda * person_action)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.annotations import (
    load_or_cache, split_keys, SCENE_CLASSES, ACTION_CLASSES,
    EXPECTED_SCENE_COUNTS, EXPECTED_ACTION_COUNTS, SCENE_TO_IDX,
)
from src.data.datasets import MAX_PLAYERS
from src.models.backbone import FEATURE_DIM
from src.models.temporal import FrameLSTM, PersonLSTM, masked_max_pool, team_pool
from src.models.heads import SceneHead, PersonActionHead
from src.evaluation.metrics import format_report, accuracy
from src.training.train_backbone import class_weights


class ImageFeatureSeqDataset(Dataset):
    """B4/B6: (T, 2048) cached per-frame features -> scene label."""

    def __init__(self, cache, data, keys, n_frames=9):
        self.cache, self.data, self.keys, self.n_frames = cache, data, keys, n_frames

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video, clip = self.keys[i]
        feats = torch.stack([self.cache[(video, clip, f)] for f in range(self.n_frames)])
        label = SCENE_TO_IDX[self.data[(video, clip)]["category"]]
        return feats, label


class PersonFeatureSeqDataset(Dataset):
    """B5/B7/B8: (MAX_PLAYERS, T, 2048) cached per-player-per-frame features."""

    def __init__(self, cache, data, keys, n_frames=9):
        self.cache, self.data, self.keys, self.n_frames = cache, data, keys, n_frames

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video, clip = self.keys[i]
        players = self.data[(video, clip)]["players"][:MAX_PLAYERS]
        feats = torch.zeros(MAX_PLAYERS, self.n_frames, FEATURE_DIM)
        mask = torch.zeros(MAX_PLAYERS, dtype=torch.bool)
        action_labels = torch.full((MAX_PLAYERS,), -1, dtype=torch.long)
        team_mask = torch.zeros(MAX_PLAYERS, dtype=torch.long)
        xs = [p[1] for p in players]
        mid_x = (min(xs) + max(xs)) / 2 if xs else 0
        for p_idx, (action, x, y, w, h) in enumerate(players):
            for f_idx in range(self.n_frames):
                feats[p_idx, f_idx] = self.cache[(video, clip, p_idx, f_idx)]
            mask[p_idx] = True
            action_labels[p_idx] = ACTION_CLASSES.index(action)
            team_mask[p_idx] = 0 if (x + w / 2) < mid_x else 1
        label = SCENE_TO_IDX[self.data[(video, clip)]["category"]]
        return feats, mask, action_labels, team_mask, label


# ---- models per variant ----

class B4Model(nn.Module):
    """Image-level LSTM over whole-frame (B1) features -> scene class."""

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lstm = FrameLSTM(FEATURE_DIM, hidden_dim)
        self.head = SceneHead(self.lstm.out_dim)

    def forward(self, feats, **kw):
        _, final = self.lstm(feats)
        return self.head(final), None


class B6Model(nn.Module):
    """Same B3 steps A+B per frame (max-pool B3-A player features within each
    of the 9 frames), then a single LSTM over the resulting 9-step sequence
    -> scene class. No per-player LSTM (LSTM1) here -- "LSTM applied on the
    image level only" per the project brief.
    """

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lstm = FrameLSTM(FEATURE_DIM, hidden_dim)
        self.head = SceneHead(self.lstm.out_dim)

    def forward(self, feats, mask, **kw):
        # feats: (B, P, T, D) cached per-player per-frame B3-A features
        B, P, T, D = feats.shape
        per_frame = [masked_max_pool(feats[:, :, t, :], mask) for t in range(T)]
        seq = torch.stack(per_frame, dim=1)  # (B, T, D)
        _, final = self.lstm(seq)
        return self.head(final), None


class B5Model(nn.Module):
    """Per-player LSTM, final-step max-pooled across players (no scene-level LSTM)."""

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.lstm1 = PersonLSTM(FEATURE_DIM, hidden_dim)
        self.head = SceneHead(self.lstm1.out_dim)

    def forward(self, feats, mask, **kw):
        _, final = self.lstm1(feats)  # (B, P, H)
        pooled = masked_max_pool(final, mask)
        return self.head(pooled), None


class B7B8Model(nn.Module):
    def __init__(self, hidden_dim=512, team_pooling=False, multitask=False):
        super().__init__()
        self.lstm1 = PersonLSTM(FEATURE_DIM, hidden_dim)
        self.team_pooling = team_pooling
        scene_in = self.lstm1.out_dim * (2 if team_pooling else 1)
        self.lstm2 = FrameLSTM(scene_in, hidden_dim)
        self.head = SceneHead(self.lstm2.out_dim)
        self.multitask = multitask
        if multitask:
            self.person_head = PersonActionHead(self.lstm1.out_dim)

    def forward(self, feats, mask, team_mask=None, **kw):
        B, P, T, D = feats.shape
        seq, final = self.lstm1(feats)  # seq: (B,P,T,H), final: (B,P,H)
        per_frame = []
        for t in range(T):
            step = seq[:, :, t, :]  # (B, P, H)
            if self.team_pooling:
                pooled = team_pool(step, mask, team_mask)
            else:
                pooled = masked_max_pool(step, mask)
            per_frame.append(pooled)
        scene_seq = torch.stack(per_frame, dim=1)  # (B, T, scene_in)
        _, scene_final = self.lstm2(scene_seq)
        scene_logits = self.head(scene_final)
        person_logits = self.person_head(final) if self.multitask else None  # (B, P, 9)
        return scene_logits, person_logits


VARIANTS = {
    "b4": (ImageFeatureSeqDataset, B4Model, False),
    "b6": (PersonFeatureSeqDataset, B6Model, True),
    "b5": (PersonFeatureSeqDataset, B5Model, True),
    "b7": (PersonFeatureSeqDataset, lambda: B7B8Model(team_pooling=False), True),
    "b8": (PersonFeatureSeqDataset, lambda: B7B8Model(team_pooling=True), True),
    "b8mt": (PersonFeatureSeqDataset, lambda: B7B8Model(team_pooling=True, multitask=True), True),
}


def load_feature_cache(cache_dir, name, split):
    return torch.load(os.path.join(cache_dir, f"{name}_{split}.pt"), weights_only=False)


def run_epoch(model, loader, device, is_person_ds, scene_criterion, person_criterion=None,
              lam=1.0, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    scene_true, scene_pred = [], []
    person_true, person_pred = [], []
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in pbar:
        if is_person_ds:
            feats, mask, action_labels, team_mask, label = batch
            feats, mask, action_labels, team_mask, label = (
                feats.to(device), mask.to(device), action_labels.to(device),
                team_mask.to(device), label.to(device))
        else:
            feats, label = batch
            feats, label = feats.to(device), label.to(device)
            mask = team_mask = action_labels = None

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                scene_logits, person_logits = model(feats, mask=mask, team_mask=team_mask)
                loss = scene_criterion(scene_logits, label)
                if person_logits is not None and person_criterion is not None:
                    valid = action_labels >= 0
                    if valid.any():
                        p_loss = person_criterion(person_logits[valid], action_labels[valid])
                        loss = loss + lam * p_loss
                        person_true.extend(action_labels[valid].cpu().tolist())
                        person_pred.extend(person_logits[valid].argmax(-1).cpu().tolist())
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        bs = label.size(0)
        total_loss += loss.item() * bs
        n += bs
        scene_true.extend(label.cpu().tolist())
        scene_pred.extend(scene_logits.argmax(-1).cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / n, scene_true, scene_pred, person_true, person_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), required=True)
    ap.add_argument("--feature-name", required=True, help="cache name prefix, e.g. b1_image or b3a_person")
    ap.add_argument("--cache-dir", default=r"E:\volleyball datasets\outputs\features_cache")
    ap.add_argument("--root", default=r"E:\volleyball datasets\volleyball_\videos")
    ap.add_argument("--ann-cache", default=r"E:\volleyball datasets\outputs\annotations.pkl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0, help="person-action loss weight (b8mt only)")
    ap.add_argument("--out-dir", default=r"E:\volleyball datasets\outputs")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None,
                     help="path to an existing checkpoint to continue training from, for --epochs more epochs")
    args = ap.parse_args()

    name = args.name or args.variant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  variant: {args.variant}")

    data = load_or_cache(args.root, args.ann_cache)
    ds_cls, model_fn, is_person_ds = VARIANTS[args.variant]

    caches = {s: load_feature_cache(args.cache_dir, args.feature_name, s) for s in ("train", "val", "test")}
    datasets = {s: ds_cls(caches[s], data, split_keys(data, s)) for s in ("train", "val", "test")}
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size, shuffle=(s == "train"),
                              num_workers=0) for s in ("train", "val", "test")}

    model = model_fn().to(device) if callable(model_fn) and not isinstance(model_fn, type) else model_fn().to(device)
    if args.resume:
        prev = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(prev["model"])
        print(f"resumed from {args.resume} (val_acc {prev.get('val_acc', '?')}), training {args.epochs} more epochs")
    scene_weights = class_weights(EXPECTED_SCENE_COUNTS, SCENE_CLASSES).to(device)
    scene_criterion = nn.CrossEntropyLoss(weight=scene_weights)
    person_criterion = None
    if args.variant == "b8mt":
        person_weights = class_weights(EXPECTED_ACTION_COUNTS, ACTION_CLASSES).to(device)
        person_criterion = nn.CrossEntropyLoss(weight=person_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "results"), exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "checkpoints", f"{name}.pt")
    last_ckpt_path = os.path.join(args.out_dir, "checkpoints", f"{name}_last.pt")
    log, best_val_acc = [], -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_t, tr_p, _, _ = run_epoch(model, loaders["train"], device, is_person_ds,
                                               scene_criterion, person_criterion, args.lam, optimizer, scaler,
                                               desc=f"epoch {epoch}/{args.epochs} train")
        va_loss, va_t, va_p, va_pt, va_pp = run_epoch(model, loaders["val"], device, is_person_ds,
                                                       scene_criterion, person_criterion, args.lam,
                                                       desc=f"epoch {epoch}/{args.epochs} val")
        scheduler.step()
        tr_acc, va_acc = accuracy(tr_t, tr_p), accuracy(va_t, va_p)
        dt = time.time() - t0
        msg = f"epoch {epoch:3d}/{args.epochs}  train_loss {tr_loss:.4f} acc {tr_acc:.3f}  val_loss {va_loss:.4f} acc {va_acc:.3f}"
        if va_pt:
            msg += f"  person_acc {accuracy(va_pt, va_pp):.3f}"
        print(msg + f"  ({dt:.1f}s)")
        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(), "variant": args.variant, "val_acc": va_acc}, ckpt_path)
        torch.save({"model": model.state_dict(), "variant": args.variant, "val_acc": va_acc,
                     "train_acc": tr_acc, "epoch": epoch}, last_ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    _, te_t, te_p, te_pt, te_pp = run_epoch(model, loaders["test"], device, is_person_ds,
                                             scene_criterion, person_criterion, args.lam, desc="test")
    report = format_report(te_t, te_p, len(SCENE_CLASSES), SCENE_CLASSES)
    print("\n=== TEST scene (best val checkpoint) ===")
    print(report)
    person_report = None
    if te_pt:
        person_report = format_report(te_pt, te_pp, len(ACTION_CLASSES), ACTION_CLASSES)
        print("\n=== TEST person-action ===")
        print(person_report)

    result = {"name": name, "variant": args.variant, "epochs": args.epochs, "lam": args.lam,
              "best_val_acc": best_val_acc, "test_scene_acc": accuracy(te_t, te_p), "log": log}
    if person_report:
        result["test_person_acc"] = accuracy(te_pt, te_pp)
    with open(os.path.join(args.out_dir, "results", f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out_dir, "results", f"{name}_test_report.txt"), "w") as f:
        f.write(report + ("\n\n" + person_report if person_report else ""))
    print(f"\nSaved checkpoint to {ckpt_path}, results to outputs/results/{name}.json")


if __name__ == "__main__":
    main()
