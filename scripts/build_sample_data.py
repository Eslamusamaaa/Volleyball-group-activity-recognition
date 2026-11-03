"""Build verified sample frames + annotations for the frontend.

Every sample is a test-split clip (matches never used for training). For each
candidate the full pipeline runs on the clip's 9-frame centre window
(YOLOv8n -> ByteTrack -> ResNet50 B3-A -> B8-mt). Each tracked player is matched
to the dataset's ground-truth box (IoU >= 0.5) so predicted actions are checked
against real annotations. Only clips whose predicted group activity equals the
ground-truth label (and, for player samples, whose featured player's action is
correct) are kept.

Writes frontend/assets/samples/<kind>_<label>.webp and frontend/data/samples.json.

    python scripts/build_sample_data.py
"""
import json
import os
import pickle
import sys
from collections import deque

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.annotations import TEST  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402
from src.inference.tracker import PlayerTracker  # noqa: E402

OUT_IMG = os.path.join(ROOT, "frontend", "assets", "samples")
OUT_JSON = os.path.join(ROOT, "frontend", "data", "samples.json")
GROUP_TARGETS = ["r_spike", "l_set", "r-pass", "l_winpoint"]
PLAYER_TARGETS = ["spiking", "blocking", "setting", "digging"]
TRIES = 10
MIN_H = 170

COLORS = {  # BGR (frontend css uses the same hues, see css/sections.css .act-*)
    "spiking": (47, 122, 255), "blocking": (255, 169, 90), "setting": (250, 139, 167), "digging": (196, 208, 53),
    "jumping": (66, 197, 245), "falling": (108, 93, 255), "moving": (252, 211, 125), "standing": (225, 213, 203), "waiting": (184, 163, 148),
}


def frame_path(vid, clip, n):
    return os.path.join(ROOT, "volleyball_", "videos", str(vid), str(clip), f"{n}.jpg")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def sharpness(img, box):
    x, y, w, h = box
    roi = img[max(0, y):y + h, max(0, x):x + w]
    return cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() if roi.size else 0.0


def analyse(predictor, ann, vid, clip):
    """Run the pipeline on the centre window; return dict or None."""
    c = int(clip)
    tracker = PlayerTracker()
    tracker.reset()
    buf = deque(maxlen=9)
    for n in range(c - 4, c + 5):
        fr = cv2.imread(frame_path(vid, clip, n))
        if fr is None:
            return None
        buf.append((n, fr, tracker.update(fr)))
    pred = predictor.predict_window(buf)
    if pred is None:
        return None
    center = buf[4][1]
    cdets = {d["track_id"]: d for d in buf[4][2]}
    gt = [(p[0], (p[1], p[2], p[1] + p[3], p[2] + p[4])) for p in ann[(vid, clip)]["players"]]
    used, players = set(), []
    for p in pred["players"]:
        bb = [float(v) for v in cdets[p["track_id"]]["bbox"]]
        best, bi = 0.0, None
        for gi, (_, gbox) in enumerate(gt):
            if gi in used:
                continue
            o = iou(bb, gbox)
            if o > best:
                best, bi = o, gi
        gt_action = None
        if bi is not None and best >= 0.5:
            used.add(bi)
            gt_action = gt[bi][0]
        players.append({"id": int(p["track_id"]), "action": p["action"], "conf": round(p["confidence"], 3), "gt": gt_action,
                        "correct": None if gt_action is None else gt_action == p["action"], "bbox": [round(v, 1) for v in bb]})
    return {"frame": center, "scene": pred["scene_class"], "scene_conf": round(pred["scene_confidence"], 3), "players": players}


def draw(img, players, featured=None):
    H, W = img.shape[:2]
    s = H / 1080.0
    placed = []
    for p in sorted(players, key=lambda q: q["bbox"][1]):
        x1, y1, x2, y2 = (int(v) for v in p["bbox"])
        feat = featured is not None and p["id"] == featured
        col = COLORS.get(p["action"], (225, 230, 238))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, int((4 if feat else 2) * s)), cv2.LINE_AA)
        text = f"#{p['id']} {p['action'].upper()}"
        scale = (0.62 if feat else 0.5) * max(s, 0.8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        lx, ly = x1, y1 - 4
        if ly - th - 6 < 0:
            ly = y2 + th + 10
        for _ in range(6):  # nudge up if it would sit on an earlier label
            if any(abs(lx - ox) < tw and abs(ly - oy) < th + 8 for ox, oy, _ in placed):
                ly -= th + 8
            else:
                break
        placed.append((lx, ly, tw))
        cv2.rectangle(img, (lx, ly - th - 6), (lx + tw + 8, ly + 2), col, -1)
        cv2.putText(img, text, (lx + 4, ly - 2), cv2.FONT_HERSHEY_DUPLEX, scale, (12, 12, 12), 1, cv2.LINE_AA)
    return img


def save(res, name, featured=None):
    img = draw(res["frame"].copy(), res["players"], featured)
    path = os.path.join(OUT_IMG, name + ".webp")
    cv2.imwrite(path, img, [cv2.IMWRITE_WEBP_QUALITY, 93])
    return f"assets/samples/{name}.webp", img.shape[1], img.shape[0]


def score(res):
    m = [p for p in res["players"] if p["correct"] is not None]
    return (sum(1 for p in m if p["correct"]), len(m))


def main():
    ann = pickle.load(open(os.path.join(ROOT, "outputs", "annotations.pkl"), "rb"))
    os.makedirs(OUT_IMG, exist_ok=True)
    predictor = WindowPredictor()
    seeded = {}
    ex = os.path.join(ROOT, "frontend", "assets", "group", "examples.json")
    if os.path.exists(ex):
        seeded = {k: (v["video"], v["clip"]) for k, v in json.load(open(ex)).items()}
    out = {"group": [], "player": []}

    for label in GROUP_TARGETS:
        cands = sorted((k for k, d in ann.items() if k[0] in TEST and d["category"] == label and len(d["players"]) >= 11),
                       key=lambda k: (k[0], int(k[1])))
        step = max(1, len(cands) // TRIES)
        order = ([seeded[label]] if label in seeded else []) + cands[::step][:TRIES]
        best = None
        for vid, clip in order:
            r = analyse(predictor, ann, vid, str(clip))
            if not r or r["scene"].replace("-", "_") != label.replace("-", "_"):
                print("group", label, vid, clip, "miss", r and r["scene"], flush=True)
                continue
            ok, n = score(r)
            print("group", label, vid, clip, "OK", ok, "/", n, flush=True)
            if n >= 8 and (best is None or ok / n > best[0]):
                best = (ok / n, vid, str(clip), r)
        if best:
            _, vid, clip, r = best
            img, w, h = save(r, f"group_{label.replace('-', '_')}")
            out["group"].append({"label": label, "video": vid, "clip": clip, "gt_scene": label, "pred_scene": r["scene"],
                                 "scene_conf": r["scene_conf"], "image": img, "width": w, "height": h, "players": r["players"],
                                 "clip_video": f"assets/group/{label.replace('-', '_')}.mp4" if seeded.get(label) == (vid, clip) else None})

    for action in PLAYER_TARGETS:
        cands = []
        for (vid, clip), d in ann.items():
            if vid not in TEST or len(d["players"]) < 10:
                continue
            tg = [p for p in d["players"] if p[0] == action and p[4] >= MIN_H]
            if len(tg) == 1:
                cands.append((tg[0][4], vid, clip, tg[0]))
        cands.sort(reverse=True)
        scored = []
        for _, vid, clip, tg in cands[:60]:
            img = cv2.imread(frame_path(vid, clip, int(clip)))
            if img is not None:
                scored.append((sharpness(img, tg[1:]), vid, clip, tg))
        scored.sort(reverse=True)
        best = None
        for _, vid, clip, tg in scored[:TRIES]:
            r = analyse(predictor, ann, vid, str(clip))
            if not r:
                continue
            gbox = (tg[1], tg[2], tg[1] + tg[3], tg[2] + tg[4])
            feat = max(r["players"], key=lambda p: iou(p["bbox"], gbox))
            hit = iou(feat["bbox"], gbox) >= 0.5 and feat["action"] == action
            ok, n = score(r)
            print("player", action, vid, clip, "featured", "OK" if hit else "miss", "players", ok, "/", n, "scene", r["scene"], flush=True)
            if hit and n >= 8 and (best is None or ok / n > best[0]):
                best = (ok / n, vid, str(clip), r, feat["id"])
        if best:
            _, vid, clip, r, fid = best
            img, w, h = save(r, f"player_{action}", featured=fid)
            out["player"].append({"label": action, "video": vid, "clip": clip, "gt_scene": ann[(vid, clip)]["category"], "pred_scene": r["scene"],
                                  "scene_conf": r["scene_conf"], "featured": fid, "image": img, "width": w, "height": h, "players": r["players"]})

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("done", len(out["group"]), "group", len(out["player"]), "player")


if __name__ == "__main__":
    main()
