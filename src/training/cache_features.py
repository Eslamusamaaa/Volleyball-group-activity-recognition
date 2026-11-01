"""Phase B: run a fine-tuned backbone once (no grad) over all clips/players/frames
and cache the resulting 2048-d feature vectors to disk.

Cache layout (Section 7.5): one .pt file per split containing a dict keyed by
  image-level : (video, clip, frame_idx) -> float32[2048]
  person-level: (video, clip, player_idx, frame_idx) -> float32[2048]
so every downstream baseline (B4-B8) looks features up the same way.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.annotations import load_or_cache, split_keys
from src.data.datasets import frame_window, EVAL_TRANSFORM, load_image, MAX_PLAYERS
from src.data.tracking import load_tracks, match_players_to_tracks, player_box_per_frame
from src.models.backbone import ResNetBackbone


@torch.no_grad()
def cache_image_level(model, device, root, data, keys, n_frames=9):
    cache = {}
    for video, clip in tqdm(keys, desc="caching image-level features"):
        paths = frame_window(root, video, clip, n_frames)
        imgs = torch.stack([load_image(p, EVAL_TRANSFORM) for p in paths]).to(device)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            feats = model.extract_features(imgs)  # (n_frames, 2048)
        feats = feats.float().cpu()
        for f_idx in range(n_frames):
            cache[(video, clip, f_idx)] = feats[f_idx]
    return cache


@torch.no_grad()
def cache_person_level(model, device, root, data, keys, n_frames=9, batch_size=64,
                        box_mode="tracked", tracking_root=None):
    from PIL import Image
    cache = {}
    missed_tracks, total_players = 0, 0
    for video, clip in tqdm(keys, desc="caching person-level features"):
        paths = frame_window(root, video, clip, n_frames)
        frame_nums = [int(os.path.splitext(os.path.basename(p))[0]) for p in paths]
        frames = [Image.open(p).convert("RGB") for p in paths]
        players = data[(video, clip)]["players"][:MAX_PLAYERS]
        center_frame = int(clip)

        tracks, matches = {}, [None] * len(players)
        if box_mode == "tracked":
            try:
                tracks = load_tracks(tracking_root, video, clip)
                matches = match_players_to_tracks(players, tracks, center_frame)
            except FileNotFoundError:
                matches = [None] * len(players)

        crops = []
        index = []
        for p_idx, (action, x, y, w, h) in enumerate(players):
            fallback = (x, y, w, h)
            total_players += 1
            if box_mode == "tracked" and matches[p_idx] is None:
                missed_tracks += 1
            for f_idx, (frame, fnum) in enumerate(zip(frames, frame_nums)):
                if box_mode == "tracked":
                    bx, by, bw, bh = player_box_per_frame(tracks, matches[p_idx], fnum, fallback)
                else:
                    bx, by, bw, bh = fallback
                box = (bx, by, bx + bw, by + bh)
                crops.append(EVAL_TRANSFORM(frame.crop(box)))
                index.append((p_idx, f_idx))
        for frame in frames:
            frame.close()
        if not crops:
            continue

        batch_feats = []
        for i in range(0, len(crops), batch_size):
            chunk = torch.stack(crops[i:i + batch_size]).to(device)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                f = model.extract_features(chunk)
            batch_feats.append(f.float().cpu())
        feats = torch.cat(batch_feats, dim=0)

        for (p_idx, f_idx), feat in zip(index, feats):
            cache[(video, clip, p_idx, f_idx)] = feat
    if box_mode == "tracked" and total_players:
        print(f"  track match rate: {total_players - missed_tracks}/{total_players} players matched "
              f"({missed_tracks} fell back to the static center-frame box)")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["image", "person"], required=True)
    ap.add_argument("--checkpoint", required=True, help="trained backbone .pt from train_backbone.py")
    ap.add_argument("--root", default=r"E:\volleyball datasets\volleyball_\videos")
    ap.add_argument("--cache", default=r"E:\volleyball datasets\outputs\annotations.pkl")
    ap.add_argument("--n-frames", type=int, default=9)
    ap.add_argument("--out-dir", default=r"E:\volleyball datasets\outputs\features_cache")
    ap.add_argument("--name", required=True, help="e.g. b1_image or b3a_person")
    ap.add_argument("--box-mode", choices=["tracked", "static"], default="tracked",
                     help="tracked = real per-frame player box from volleyball_tracking_annotation "
                          "(matches the paper's Dlib tracker); static = reuse the center-frame box "
                          "for all 9 frames (Section 7.2 choice (a) simplification, person-level only)")
    ap.add_argument("--tracking-root",
                     default=r"E:\volleyball datasets\volleyball_tracking_annotation\volleyball_tracking_annotation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = len(ckpt["classes"])
    model = ResNetBackbone(num_classes=num_classes, mode="finetune").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = load_or_cache(args.root, args.cache)
    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        keys = split_keys(data, split)
        print(f"[{split}] {len(keys)} clips")
        if args.level == "image":
            cache = cache_image_level(model, device, args.root, data, keys, args.n_frames)
        else:
            cache = cache_person_level(model, device, args.root, data, keys, args.n_frames,
                                        box_mode=args.box_mode, tracking_root=args.tracking_root)
        out_path = os.path.join(args.out_dir, f"{args.name}_{split}.pt")
        torch.save(cache, out_path)
        print(f"  saved {len(cache)} vectors -> {out_path}")


if __name__ == "__main__":
    main()
