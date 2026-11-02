"""Section 10, stage 3: slide a 9-frame window across a tracked video and
run B8-mt on each window -> one (scene class, per-player action) prediction
per window. Stage 4 (timeline.py) turns these into timestamped events.
"""
from collections import deque

import cv2
import torch
from torchvision import transforms

from ..data.annotations import SCENE_CLASSES, ACTION_CLASSES
from ..data.datasets import IMAGENET_MEAN, IMAGENET_STD
from .tracker import PlayerTracker
from .model_io import load_backbone, load_b8mt

MAX_PLAYERS = 12  # matches training (src/data/datasets.py)

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def crop_bgr(frame_bgr, bbox):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class WindowPredictor:
    """Owns the loaded models; predict_window() does one 9-frame inference."""

    def __init__(self, device=None, backbone_ckpt=None, b8mt_ckpt=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        kwargs_b = {"device": self.device}
        if backbone_ckpt:
            kwargs_b["checkpoint"] = backbone_ckpt
        kwargs_m = {"device": self.device}
        if b8mt_ckpt:
            kwargs_m["checkpoint"] = b8mt_ckpt
        self.backbone = load_backbone(**kwargs_b)
        self.model = load_b8mt(**kwargs_m)

    @torch.no_grad()
    def predict_window(self, frame_buffer):
        """frame_buffer: deque of (frame_idx, frame_bgr, detections) of length
        n_frames, detections = list of {"track_id", "bbox", "conf"}.

        Only players tracked in EVERY frame of the window are used (a track
        that drops out mid-window has no 9-step sequence to feed LSTM1) --
        returns None if fewer than 2 such players survive.
        """
        n_frames = len(frame_buffer)
        track_ids_per_frame = [{d["track_id"] for d in dets} for _, _, dets in frame_buffer]
        stable_ids = set.intersection(*track_ids_per_frame) if track_ids_per_frame else set()
        if len(stable_ids) < 2:
            return None

        center_idx = n_frames // 2
        _, center_frame, center_dets = frame_buffer[center_idx]
        center_by_id = {d["track_id"]: d for d in center_dets}
        ordered_ids = sorted(stable_ids, key=lambda tid: center_by_id[tid]["bbox"][0])[:MAX_PLAYERS]
        P = len(ordered_ids)

        xs = [center_by_id[tid]["bbox"][0] + (center_by_id[tid]["bbox"][2] - center_by_id[tid]["bbox"][0]) / 2
              for tid in ordered_ids]
        mid_x = (min(xs) + max(xs)) / 2

        crops = []  # flat list, P * n_frames, in (player, frame) order
        for tid in ordered_ids:
            for f_idx, frame_idx, frame_bgr, dets in [(i, fi, fb, d) for i, (fi, fb, d) in enumerate(frame_buffer)]:
                # Safely find detection — track may have temporarily vanished
                det_list = [d for d in dets if d["track_id"] == tid]
                if not det_list:
                    crops.append(torch.zeros(3, 224, 224))
                    continue
                det = det_list[0]
                rgb = crop_bgr(frame_bgr, det["bbox"])
                crops.append(EVAL_TRANSFORM(rgb) if rgb is not None else torch.zeros(3, 224, 224))

        batch = torch.stack(crops).to(self.device)
        feats = self.backbone.extract_features(batch).float().cpu()  # (P*n_frames, 2048)
        feats = feats.view(P, n_frames, -1)

        full_feats = torch.zeros(1, MAX_PLAYERS, n_frames, feats.shape[-1])
        mask = torch.zeros(1, MAX_PLAYERS, dtype=torch.bool)
        team_mask = torch.zeros(1, MAX_PLAYERS, dtype=torch.long)
        full_feats[0, :P] = feats
        mask[0, :P] = True
        for i, tid in enumerate(ordered_ids):
            cx = center_by_id[tid]["bbox"][0] + (center_by_id[tid]["bbox"][2] - center_by_id[tid]["bbox"][0]) / 2
            team_mask[0, i] = 0 if cx < mid_x else 1

        full_feats, mask, team_mask = (t.to(self.device) for t in (full_feats, mask, team_mask))
        scene_logits, person_logits = self.model(full_feats, mask=mask, team_mask=team_mask)
        scene_prob = torch.softmax(scene_logits[0], dim=-1)
        scene_idx = int(scene_prob.argmax())
        scene_name = SCENE_CLASSES[scene_idx]

        person_preds = []
        if person_logits is not None:
            person_prob = torch.softmax(person_logits[0], dim=-1)
            for i, tid in enumerate(ordered_ids):
                a_idx = int(person_prob[i].argmax())
                person_preds.append({
                    "track_id": tid, "action": ACTION_CLASSES[a_idx],
                    "confidence": float(person_prob[i, a_idx]),
                })

        return {
            "scene_class": scene_name,
            "scene_confidence": float(scene_prob[scene_idx]),
            "team": "left" if scene_name[0] == "l" else "right",
            "event": scene_name[2:],  # SCENE_CLASSES mixes '-' and '_' as the separator at index 1
            "n_players": P,
            "players": person_preds,
        }


def iter_windows(video_path, predictor, n_frames=9, stride=9):
    """Yields (start_frame, end_frame, prediction_dict_or_None) for every
    stride-th window of the video. stride=n_frames -> non-overlapping
    windows (simplest, cheapest); a smaller stride gives finer event
    boundaries at higher compute cost (brief Section 10, stage 3)."""
    from .video_io import iter_video_frames

    tracker = PlayerTracker()
    tracker.reset()
    buf = deque(maxlen=n_frames)
    for idx, frame in iter_video_frames(video_path):
        dets = tracker.update(frame)
        buf.append((idx, frame, dets))
        if len(buf) == n_frames and (idx - n_frames + 1) % stride == 0:
            pred = predictor.predict_window(buf)
            yield buf[0][0], buf[-1][0], pred
