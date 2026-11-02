"""Load Volleyball dataset annotations, print statistics, and visualize a random sample.

Usage: python explore_data.py [--root D:/volleyball datasets/volleyball_/videos] [--seed 0]
"""
import argparse
import os
import pickle
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

TRAIN = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
SPLIT = {v: s for s, vids in (("train", TRAIN), ("val", VAL), ("test", TEST)) for v in vids}


def load_annotations(root):
    """Return {(video, clip): {'category': str, 'players': [(action, x, y, w, h), ...]}}."""
    data = {}
    for vid in sorted(int(d) for d in os.listdir(root) if d.isdigit()):
        with open(os.path.join(root, str(vid), "annotations.txt")) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                clip = parts[0].split(".")[0]
                players = []
                for i in range(2, len(parts), 5):
                    # actual file order is "X Y W H action" (Info.txt lists action first)
                    x, y, w, h = map(int, parts[i:i + 4])
                    action = parts[i + 4]
                    players.append((action, x, y, w, h))
                data[(vid, clip)] = {"category": parts[1], "players": players}
    return data


def print_stats(data):
    print(f"Videos: {len({v for v, _ in data})}   Clips (annotated frames): {len(data)}")
    for name in ("train", "val", "test"):
        keys = [k for k in data if SPLIT[k[0]] == name]
        print(f"  {name:5s}: {len({v for v, _ in keys}):2d} videos, {len(keys):4d} clips")

    cats = Counter(d["category"] for d in data.values())
    print("\nGroup activity (scene) classes:")
    for c, n in cats.most_common():
        print(f"  {c:12s} {n:5d}  {100 * n / len(data):5.1f}%")

    actions = Counter(p[0] for d in data.values() for p in d["players"])
    total = sum(actions.values())
    print(f"\nPlayer actions (total boxes: {total}):")
    for a, n in actions.most_common():
        print(f"  {a:10s} {n:6d}  {100 * n / total:5.1f}%")

    npl = Counter(len(d["players"]) for d in data.values())
    print("\nPlayers per clip:", dict(sorted(npl.items())))

    wh = np.array([(p[3], p[4]) for d in data.values() for p in d["players"]])
    print(f"Box width  mean/min/max: {wh[:, 0].mean():.0f} / {wh[:, 0].min()} / {wh[:, 0].max()}")
    print(f"Box height mean/min/max: {wh[:, 1].mean():.0f} / {wh[:, 1].min()} / {wh[:, 1].max()}")

    print("\nScene class distribution per split:")
    for name in ("train", "val", "test"):
        c = Counter(d["category"] for k, d in data.items() if SPLIT[k[0]] == name)
        print(f"  {name:5s}", dict(sorted(c.items())))
    return cats, actions


def plot_distributions(cats, actions, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].bar(*zip(*sorted(cats.items())))
    ax[0].set_title("Group activity classes")
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].bar(*zip(*sorted(actions.items(), key=lambda kv: -kv[1])))
    ax[1].set_yscale("log")
    ax[1].set_title("Player actions (log scale)")
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def visualize(root, data, keys, out):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, key in zip(axes.ravel(), keys):
        vid, clip = key
        img = Image.open(os.path.join(root, str(vid), clip, f"{clip}.jpg"))
        ax.imshow(img)
        for action, x, y, w, h in data[key]["players"]:
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", lw=1.5))
            ax.text(x, y - 3, action, color="yellow", fontsize=7, backgroundcolor="black")
        ax.set_title(f"video {vid} / clip {clip}  [{SPLIT[vid]}]  ->  {data[key]['category']}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\volleyball datasets\volleyball_\videos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=r"D:\volleyball datasets\outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_annotations(args.root)
    with open(os.path.join(args.out, "annotations.pkl"), "wb") as f:
        pickle.dump(data, f)

    cats, actions = print_stats(data)
    plot_distributions(cats, actions, os.path.join(args.out, "distributions.png"))

    random.seed(args.seed)
    visualize(args.root, data, random.sample(sorted(data), 4), os.path.join(args.out, "sample.png"))
    print(f"\nSaved annotations.pkl, distributions.png, sample.png to {args.out}")
