"""Render the annotated pipeline video for the group-activity classes that have
no clip yet, so the site can show one verified clip per class (8 in total).

For each missing class, test-split clips (never seen in training) go through
the full pipeline (YOLOv8n -> ByteTrack -> B3-A -> B8-mt -> annotated render).
A clip is kept only if the pipeline's event at the annotated centre frame
equals the dataset's ground-truth label. Also writes a poster image for every
clip, and frontend/data/showcase.json.

    python scripts/render_showcase_clips.py
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from render_group_examples import PLAYBACK_FPS, stitch  # noqa: E402
from src.data.annotations import TEST  # noqa: E402
from src.inference.annotator import render_annotated_pipeline  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402

OUT = os.path.join(ROOT, "frontend", "assets", "group")
SAMPLES = os.path.join(ROOT, "frontend", "data", "samples.json")
SHOWCASE = os.path.join(ROOT, "frontend", "data", "showcase.json")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
# The four classes shown on the site. Each already has a verified clip in
# frontend/assets/group; add a label here to render and verify another one.
ORDER = ["r_spike", "l_set", "r-pass", "l_winpoint", "l-pass", "r_set", "l-spike", "r_winpoint"]
TRIES = 10


def safe(label):
    return label.replace("-", "_")


def poster(mp4, at_sec, dest):
    from PIL import Image
    png = dest + ".png"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{at_sec:.2f}", "-i", mp4, "-frames:v", "1", "-q:v", "2", png], check=True)
    Image.open(png).convert("RGB").save(dest + ".webp", "WEBP", quality=88, method=6)
    os.remove(png)


def main():
    ann = pickle.load(open(os.path.join(ROOT, "outputs", "annotations.pkl"), "rb"))
    # Clips already rendered (and slowed) are kept as they are; only missing labels are rendered.
    existing = {c["label"]: c for c in json.load(open(SHOWCASE))} if os.path.exists(SHOWCASE) else {}
    predictor = WindowPredictor()
    tmp = tempfile.mkdtemp()
    items = {}

    for label in ORDER:
        team, event = ("left" if label[0] == "l" else "right"), label[2:]
        name = safe(label)
        if label in existing and os.path.exists(os.path.join(OUT, name + ".mp4")):
            c = existing[label]
            items[label] = {"video": c["source_video"], "clip": c["source_clip"], "confidence": c["confidence"]}
            print(label, "kept", flush=True)
            continue
        cands = sorted((k for k, d in ann.items() if k[0] in TEST and d["category"] == label and len(d["players"]) >= 11),
                       key=lambda k: (k[0], int(k[1])))
        step = max(1, len(cands) // TRIES)
        for vid, clip in cands[::step][:TRIES]:
            src, out = os.path.join(tmp, "src.mp4"), os.path.join(tmp, "out.mp4")
            center, _ = stitch(vid, clip, src)
            res = render_annotated_pipeline(src, out, predictor=predictor)
            t = center / PLAYBACK_FPS
            hit = [e for e in res["events"] if e["start_sec"] <= t <= e["end_sec"]]
            ok = bool(hit) and hit[0]["team"] == team and hit[0]["event"] == event
            print(label, vid, clip, [(e["team"], e["event"], e["confidence"]) for e in hit], "OK" if ok else "miss", flush=True)
            if not ok:
                continue
            final = os.path.join(OUT, name + ".mp4")
            shutil.move(out, final)
            poster(final, t, os.path.join(OUT, name + "_poster"))
            items[label] = {"video": vid, "clip": str(clip), "confidence": hit[0]["confidence"]}
            break

    showcase = [{"label": l, "team": "left" if l[0] == "l" else "right", "event": l[2:],
                 "video_file": f"assets/group/{safe(l)}.mp4", "poster": f"assets/group/{safe(l)}_poster.webp",
                 "source_video": items[l]["video"], "source_clip": items[l]["clip"], "confidence": items[l]["confidence"]}
                for l in ORDER if l in items]
    json.dump(showcase, open(SHOWCASE, "w"), indent=1)
    print("showcase:", [s["label"] for s in showcase], "missing:", [l for l in ORDER if l not in items])


if __name__ == "__main__":
    main()
