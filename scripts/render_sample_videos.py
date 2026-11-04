"""Build the videos for the site's "Sample videos" section and the hero.

* Group activity: reuses the four verified clips in frontend/assets/group.
* Player activity: renders the full pipeline on test-split clips (never seen in
  training) and keeps a clip only when the detection that overlaps the dataset's
  ground-truth player (IoU >= 0.5) at the annotated centre frame is predicted with
  the ground-truth action. That player is ringed in the final render.
* Every clip is brought to about 8 s: raw clips are slowed with motion interpolation, longer ones are sped up.

    python scripts/render_sample_videos.py
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from render_group_examples import PLAYBACK_FPS, stitch  # noqa: E402
from src.data.annotations import TEST  # noqa: E402
from src.inference import annotator  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402

FRONT = os.path.join(ROOT, "frontend")
GROUP_DIR = os.path.join(FRONT, "assets", "group")
PLAYER_DIR = os.path.join(FRONT, "assets", "player")
SAMPLES = os.path.join(FRONT, "data", "samples.json")
SHOWCASE = os.path.join(FRONT, "data", "showcase.json")
OUT_JSON = os.path.join(FRONT, "data", "videos.json")
FFMPEG = annotator.FFMPEG_EXE
ACTIONS = ["spiking", "blocking", "setting", "digging"]
TARGET_SEC = 8.0
TRIES = 12


def duration(path):
    cap = cv2.VideoCapture(path)
    d = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
    cap.release()
    return d


def slow(src, dst):
    """Bring a clip to about TARGET_SEC: stretch short raw clips with motion interpolation, speed up clips that
    are already longer than 10 s, leave the rest alone."""
    d = duration(src)
    tmp = dst + ".tmp.mp4"
    if d < 6:
        vf = f"setpts={TARGET_SEC / d:.4f}*PTS,minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    elif d > 10:
        vf = f"setpts={TARGET_SEC / d:.4f}*PTS,fps=25"
    else:
        if src != dst:
            shutil.copy(src, dst)
        return
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", src, "-vf", vf, "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "medium", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, dst)


def poster(mp4, at_sec, dest):
    from PIL import Image
    png = dest + ".png"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{at_sec:.2f}", "-i", mp4, "-frames:v", "1", "-q:v", "2", png], check=True)
    Image.open(png).convert("RGB").save(dest + ".webp", "WEBP", quality=88, method=6)
    os.remove(png)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


class Recorder:
    """Hooks the annotator's draw calls to record per-frame boxes/actions and ring one track."""

    def __init__(self):
        self.frames, self._buf, self.ring = {}, [], None
        self._box, self._hud = annotator.draw_player_box_and_action, annotator.draw_top_broadcast_hud

    def __enter__(self):
        def box(frame, bbox, track_id, action="standing", team="neutral"):
            self._box(frame, bbox, track_id, action=action, team=team)
            self._buf.append((track_id, [float(v) for v in bbox], action))
            if self.ring is not None and track_id == self.ring:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (0, 255, 255), 4)

        def hud(frame, event=None, frame_idx=0, fps=25.0, n_players=0):
            self._hud(frame, event, frame_idx, fps, n_players)
            self.frames[frame_idx], self._buf = self._buf, []

        annotator.draw_player_box_and_action, annotator.draw_top_broadcast_hud = box, hud
        return self

    def __exit__(self, *_):
        annotator.draw_player_box_and_action, annotator.draw_top_broadcast_hud = self._box, self._hud


def render_player(predictor, ann, vid, clip, action, tmp):
    gt = [(p[1], p[2], p[1] + p[3], p[2] + p[4]) for p in ann[(vid, clip)]["players"] if p[0] == action]
    if not gt:
        return None
    src, out = os.path.join(tmp, "src.mp4"), os.path.join(tmp, "out.mp4")
    center, _ = stitch(vid, clip, src)
    rec = Recorder()
    with rec:
        annotator.render_annotated_pipeline(src, out, predictor=predictor)
    feat = None
    for tid, bbox, act in rec.frames.get(center, []):
        if act == action and any(iou(bbox, g) >= 0.5 for g in gt):
            feat = tid
            break
    if feat is None:
        return None
    rec2 = Recorder()
    rec2.ring = feat
    with rec2:
        annotator.render_annotated_pipeline(src, out, predictor=predictor)
    return out, center / PLAYBACK_FPS


def main():
    ann = pickle.load(open(os.path.join(ROOT, "outputs", "annotations.pkl"), "rb"))
    old = {s["label"]: s for s in json.load(open(SAMPLES))["player"]}
    predictor = WindowPredictor()
    tmp = tempfile.mkdtemp()
    os.makedirs(PLAYER_DIR, exist_ok=True)

    group = []
    for c in json.load(open(SHOWCASE)):
        mp4 = os.path.join(FRONT, c["video_file"])
        slow(mp4, mp4)
        poster(mp4, duration(mp4) / 2, mp4[:-4] + "_poster")
        group.append({"label": c["label"], "team": c["team"], "event": c["event"], "video_file": c["video_file"],
                      "poster": c["poster"], "source_video": c["source_video"], "source_clip": c["source_clip"],
                      "confidence": c["confidence"]})
        print("group", c["label"], "slowed", round(duration(mp4), 1), "s", flush=True)

    kept = {p["label"]: p for p in json.load(open(OUT_JSON))["player"]} if os.path.exists(OUT_JSON) else {}
    player = []
    for action in ACTIONS:
        if action in kept and os.path.exists(os.path.join(PLAYER_DIR, f"{action}.mp4")):
            player.append(kept[action])
            print("player", action, "kept", flush=True)
            continue
        cands = []
        if action in old:
            cands.append((old[action]["video"], str(old[action]["clip"])))
        extra = sorted((k for k, d in ann.items() if k[0] in TEST and any(p[0] == action for p in d["players"])),
                       key=lambda k: (k[0], int(k[1])))
        step = max(1, len(extra) // TRIES)
        cands += [k for k in extra[::step][:TRIES] if k not in cands]
        for vid, clip in cands:
            r = render_player(predictor, ann, vid, clip, action, tmp)
            print("player", action, vid, clip, "OK" if r else "miss", flush=True)
            if not r:
                continue
            out, t = r
            final = os.path.join(PLAYER_DIR, f"{action}.mp4")
            frac = t / duration(out)
            slow(out, final)
            poster(final, frac * duration(final), os.path.join(PLAYER_DIR, f"{action}_poster"))
            player.append({"label": action, "video_file": f"assets/player/{action}.mp4",
                           "poster": f"assets/player/{action}_poster.webp", "source_video": vid, "source_clip": clip})
            break

    json.dump({"group": group, "player": player}, open(OUT_JSON, "w"), indent=1)
    print("group:", [g["label"] for g in group], "player:", [p["label"] for p in player])


if __name__ == "__main__":
    main()
