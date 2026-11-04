"""Find, for each of the 8 group-activity classes, the test-split clip where the model's own
banner shows the dataset's label for the longest stretch, so the site's clip for that class is
unambiguous from start to finish.

For every candidate (test videos only, never seen in training) the 41 frames go through the same
tracker + 9-frame sliding window as the renderer; a window "agrees" when its predicted scene class
equals the clip's dataset label with confidence >= MIN_CONF. Results go to outputs/clear_clips.json.

    python scripts/find_clear_clips.py
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
from src.inference.annotator import _resize_for_inference, _scale_dets  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402
from src.inference.tracker import PlayerTracker  # noqa: E402

LABELS = os.environ.get("LABELS", "r_spike,l_set,r-pass,l_winpoint,l-pass,r_set,l-spike,r_winpoint").split(",")
PER_LABEL = int(os.environ.get("PER_LABEL", 30))
MIN_CONF = 0.9
OUT = os.path.join(ROOT, "outputs", "clear_clips.json")
norm = lambda s: s.replace("-", "_")


def windows_for(vid, clip, predictor, tracker):
    d = os.path.join(ROOT, "volleyball_", "videos", str(vid), str(clip))
    frames = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    tracker.reset()
    buf, wins = deque(maxlen=9), []
    for idx, f in enumerate(frames):
        img = cv2.imread(os.path.join(d, f))
        small, scale = _resize_for_inference(img)
        small_dets = tracker.update(small)
        buf.append((idx, small, small_dets))
        if len(buf) == 9 and (idx - 8) % 9 == 0:
            try:
                p = predictor.predict_window(buf)
            except Exception:
                p = None
            wins.append({"start": buf[0][0], "end": idx, "cls": p["scene_class"] if p else None, "conf": round(p["scene_confidence"], 3) if p else 0.0})
    return wins, len(frames)


def score(wins, label):
    ok = [bool(w["cls"]) and norm(w["cls"]) == norm(label) and w["conf"] >= MIN_CONF for w in wins]
    best = run = 0
    for o in ok:
        run = run + 1 if o else 0
        best = max(best, run)
    mean = sum(w["conf"] for w, o in zip(wins, ok) if o) / max(1, sum(ok))
    return sum(ok), best, round(mean, 3)


def main():
    ann = pickle.load(open(os.path.join(ROOT, "outputs", "annotations.pkl"), "rb"))
    predictor, tracker = WindowPredictor(), PlayerTracker()
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for label in LABELS:
        cands = sorted((k for k, d in ann.items() if k[0] in TEST and d["category"] == label and len(d["players"]) >= 11),
                       key=lambda k: (k[0], int(k[1])))
        step = max(1, len(cands) // PER_LABEL)
        rows = results.get(label, [])
        done = {(r["video"], r["clip"]) for r in rows}
        for vid, clip in cands[::step][:PER_LABEL]:
            if (vid, str(clip)) in done:
                continue
            wins, n = windows_for(vid, clip, predictor, tracker)
            agree, run, mean = score(wins, label)
            rows.append({"video": vid, "clip": str(clip), "frames": n, "windows": wins, "agree": agree, "run": run, "mean_conf": mean})
            print(label, vid, clip, f"agree={agree}/{len(wins)} run={run} conf={mean}", flush=True)
            results[label] = rows
            json.dump(results, open(OUT, "w"))
        best = max(rows, key=lambda r: (r["agree"], r["run"], r["mean_conf"]))
        print("BEST", label, best["video"], best["clip"], best["agree"], best["run"], best["mean_conf"], flush=True)
    print("done")


if __name__ == "__main__":
    main()
