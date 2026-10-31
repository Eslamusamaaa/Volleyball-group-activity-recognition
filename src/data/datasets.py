"""PyTorch Dataset classes for the Volleyball baselines (B1-B8).

Each clip folder on disk holds ~41 frames centered on the annotated frame
(the clip_id). The official baselines use a 9-frame window; we reuse the
center frame's player boxes for every frame in that window (Section 7.2,
choice (a): minimal player movement assumption over a short window).
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .annotations import SCENE_TO_IDX, ACTION_TO_IDX

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.0),  # flip would swap left/right labels; disabled by default
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def clip_dir(root, video, clip):
    return os.path.join(root, str(video), str(clip))


def center_frame_path(root, video, clip):
    return os.path.join(clip_dir(root, video, clip), f"{clip}.jpg")


def frame_window(root, video, clip, n_frames=9):
    """Return n_frames paths centered on the clip's annotated frame, numerically sorted.

    Falls back to whatever frames exist if the folder has fewer than requested.
    """
    d = clip_dir(root, video, clip)
    all_frames = sorted(
        (int(os.path.splitext(f)[0]) for f in os.listdir(d) if f.endswith(".jpg")),
    )
    center = int(clip)
    if center not in all_frames:
        raise ValueError(f"center frame {center} missing from {d}")
    idx = all_frames.index(center)
    half = n_frames // 2
    lo = max(0, idx - half)
    hi = min(len(all_frames), idx + half + 1)
    window = all_frames[lo:hi]
    # pad by repeating the nearest edge frame if the clip is short on one side
    while len(window) < n_frames:
        if lo == 0:
            window.append(window[-1])
        else:
            window.insert(0, window[0])
    return [os.path.join(d, f"{f}.jpg") for f in window[:n_frames]]


def load_image(path, transform):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return transform(img)


class SceneImageDataset(Dataset):
    """B1: center-frame image -> 8-way scene class."""

    def __init__(self, root, data, keys, transform=None):
        self.root = root
        self.data = data
        self.keys = keys
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video, clip = self.keys[i]
        img = load_image(center_frame_path(self.root, video, clip), self.transform)
        label = SCENE_TO_IDX[self.data[(video, clip)]["category"]]
        return img, label


class PersonCropDataset(Dataset):
    """B3-A: one crop per player box on the center frame -> 9-way action class."""

    def __init__(self, root, data, keys, transform=None):
        self.root = root
        self.data = data
        self.transform = transform or EVAL_TRANSFORM
        self.index = []  # (video, clip, player_idx)
        for video, clip in keys:
            for p_idx in range(len(data[(video, clip)]["players"])):
                self.index.append((video, clip, p_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        video, clip, p_idx = self.index[i]
        action, x, y, w, h = self.data[(video, clip)]["players"][p_idx]
        with Image.open(center_frame_path(self.root, video, clip)) as img:
            img = img.convert("RGB")
            crop = img.crop((x, y, x + w, y + h))
            crop = self.transform(crop)
        label = ACTION_TO_IDX[action]
        return crop, label

    def get_labels(self):
        """All labels in dataset order, for building a class-balanced sampler."""
        return [ACTION_TO_IDX[self.data[(v, c)]["players"][p][0]] for v, c, p in self.index]


MAX_PLAYERS = 12


class ClipPersonCropsDataset(Dataset):
    """B2 / B3 step B+C: ALL player crops of a clip's center frame (no time
    dimension) -> scene label. Used to build a per-clip pooled representation
    from single-frame person crops, either with a frozen backbone (B2) or a
    backbone fine-tuned on person actions (B3).

    Returns:
      crops: (MAX_PLAYERS, 3, 224, 224), zero-padded if fewer players
      mask: (MAX_PLAYERS,) bool, True where a real player exists
      scene_label: int
    """

    def __init__(self, root, data, keys, transform=None):
        self.root = root
        self.data = data
        self.keys = keys
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video, clip = self.keys[i]
        players = self.data[(video, clip)]["players"][:MAX_PLAYERS]
        crops = torch.zeros(MAX_PLAYERS, 3, 224, 224)
        mask = torch.zeros(MAX_PLAYERS, dtype=torch.bool)
        with Image.open(center_frame_path(self.root, video, clip)) as img:
            img = img.convert("RGB")
            for p_idx, (action, x, y, w, h) in enumerate(players):
                crop = img.crop((x, y, x + w, y + h))
                crops[p_idx] = self.transform(crop)
                mask[p_idx] = True
        label = SCENE_TO_IDX[self.data[(video, clip)]["category"]]
        return crops, mask, label


class ClipFrameSequenceDataset(Dataset):
    """B4/B6: 9-frame window of whole-image frames per clip -> 8-way scene class.

    Returns a (9, C, H, W) tensor; used both for raw-image caching (B4) and
    for caching with the B3-A backbone applied to full frames (B6).
    """

    def __init__(self, root, data, keys, n_frames=9, transform=None):
        self.root = root
        self.data = data
        self.keys = keys
        self.n_frames = n_frames
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video, clip = self.keys[i]
        paths = frame_window(self.root, video, clip, self.n_frames)
        imgs = torch.stack([load_image(p, self.transform) for p in paths])
        label = SCENE_TO_IDX[self.data[(video, clip)]["category"]]
        return imgs, label, video, clip


class PersonSequenceDataset(Dataset):
    """B5/B7/B8: per-player crops across a 9-frame window -> scene label + per-player action labels.

    Center-frame boxes are reused for every frame in the window (Section 7.2 choice (a)).
    Returns:
      crops: (MAX_PLAYERS, n_frames, C, H, W) tensor, zero-padded if fewer players
      player_mask: (MAX_PLAYERS,) bool, True where a real player exists
      action_labels: (MAX_PLAYERS,) long, -1 where padded
      scene_label: int
      team_mask: (MAX_PLAYERS,) long in {0, 1} — left/right half by x-position, for B8 pooling
    """

    def __init__(self, root, data, keys, n_frames=9, transform=None):
        self.root = root
        self.data = data
        self.keys = keys
        self.n_frames = n_frames
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video, clip = self.keys[i]
        paths = frame_window(self.root, video, clip, self.n_frames)
        frames = [Image.open(p).convert("RGB") for p in paths]
        players = self.data[(video, clip)]["players"]

        crops = torch.zeros(MAX_PLAYERS, self.n_frames, 3, 224, 224)
        player_mask = torch.zeros(MAX_PLAYERS, dtype=torch.bool)
        action_labels = torch.full((MAX_PLAYERS,), -1, dtype=torch.long)
        team_mask = torch.zeros(MAX_PLAYERS, dtype=torch.long)

        xs = [p[1] for p in players]
        mid_x = (min(xs) + max(xs)) / 2 if xs else 0

        for p_idx, (action, x, y, w, h) in enumerate(players[:MAX_PLAYERS]):
            box = (x, y, x + w, y + h)
            for f_idx, frame in enumerate(frames):
                crop = frame.crop(box)
                crops[p_idx, f_idx] = self.transform(crop)
            player_mask[p_idx] = True
            action_labels[p_idx] = ACTION_TO_IDX[action]
            team_mask[p_idx] = 0 if (x + w / 2) < mid_x else 1

        for frame in frames:
            frame.close()

        scene_label = SCENE_TO_IDX[self.data[(video, clip)]["category"]]
        return crops, player_mask, action_labels, team_mask, scene_label
