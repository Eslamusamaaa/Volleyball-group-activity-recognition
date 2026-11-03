"""Render verified group-activity examples for the frontend.

For each target class, tries test-split clips (never seen in training) through
the full inference pipeline (YOLOv8n + ByteTrack + B3-A + B8-mt + annotated
render) and keeps the first clip whose predicted group activity at the
annotated centre frame equals the dataset's ground-truth label. Writes
frontend/assets/group/<label>.mp4, <label>.webp (centre frame) and
examples.json (clip ids, ground truth, prediction, confidence).

    python scripts/render_group_examples.py
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

from src.data.annotations import TEST  # noqa: E402
from src.inference.annotator import render_annotated_pipeline  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402

OUT = os.path.join(ROOT, "frontend", "assets", "group")
TARGETS = ["r_spike", "l_set", "r-pass", "l_winpoint"]
TRIES = 8
PLAYBACK_FPS = 8  # 41 frames at 25 fps is only 1.6 s; slow down so it can be read
FFMPEG = os.path.join(ROOT, ".venv", "Scripts", "ffmpeg.exe")


def stitch(vid, clip, path):
    d = os.path.join(ROOT, "volleyball_", "videos", str(vid), str(clip))
    frames = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    first = cv2.imread(os.path.join(d, frames[0]))
    h, w = first.shape[:2]
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (w, h))
    center = None
    for i, f in enumerate(frames):
        wr.write(cv2.imread(os.path.join(d, f)))
        if f == f"{clip}.jpg":
            center = i
    wr.release()
    return center, len(frames)


def main():
    ann = pickle.load(open(os.path.join(ROOT, "outputs", "annotations.pkl"), "rb"))
    os.makedirs(OUT, exist_ok=True)
    predictor = WindowPredictor()
    result = {}
    tmp = tempfile.mkdtemp()
    for label in TARGETS:
        team = "left" if label[0] == "l" else "right"
        event = label[2:]
        cands = sorted((k for k, d in ann.items() if k[0] in TEST and d["category"] == label and len(d["players"]) >= 11),
                       key=lambda k: (k[0], int(k[1])))
        step = max(1, len(cands) // TRIES)
        for vid, clip in cands[::step][:TRIES]:
            src = os.path.join(tmp, "src.mp4")
            center, n = stitch(vid, clip, src)
            out = os.path.join(tmp, "out.mp4")
            res = render_annotated_pipeline(src, out, predictor=predictor)
            t = center / PLAYBACK_FPS
            hit = [e for e in res["events"] if e["start_sec"] <= t <= e["end_sec"]]
            ok = bool(hit) and hit[0]["team"] == team and hit[0]["event"] == event
            print(label, vid, clip, "->", [(e["team"], e["event"], e["confidence"]) for e in hit], "OK" if ok else "miss", flush=True)
            if not ok:
                continue
            safe = label.replace("-", "_")
            final = os.path.join(OUT, f"{safe}.mp4")
            shutil.move(out, final)
            subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", final, "-frames:v", "1", "-q:v", "2", os.path.join(tmp, "c.png")], check=True)
            from PIL import Image
            Image.open(os.path.join(tmp, "c.png")).convert("RGB").save(os.path.join(OUT, f"{safe}.webp"), "WEBP", quality=94, method=6)
            result[label] = {"video": vid, "clip": str(clip), "ground_truth": label, "predicted": f"{hit[0]['team']} {hit[0]['event']}",
                             "confidence": hit[0]["confidence"], "file": f"assets/group/{safe}"}
            break
    json.dump(result, open(os.path.join(OUT, "examples.json"), "w"), indent=1)
    print("verified:", list(result))


if __name__ == "__main__":
    main()
