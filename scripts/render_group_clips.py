"""Render the annotated pipeline video for each group-activity sample in
frontend/data/samples.json and attach it only when the pipeline's event at the
centre frame equals the ground-truth label.

    python scripts/build_sample_data.py && python scripts/render_group_clips.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from render_group_examples import PLAYBACK_FPS, stitch  # noqa: E402
from src.inference.annotator import render_annotated_pipeline  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402

DATA = os.path.join(ROOT, "frontend", "data", "samples.json")
OUT = os.path.join(ROOT, "frontend", "assets", "group")


def main():
    data = json.load(open(DATA))
    predictor = WindowPredictor()
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    tmp = tempfile.mkdtemp()
    for s in data["group"]:
        label = s["gt_scene"]
        team, event = ("left" if label[0] == "l" else "right"), label[2:]
        src, out = os.path.join(tmp, "src.mp4"), os.path.join(tmp, "out.mp4")
        center, _ = stitch(s["video"], s["clip"], src)
        res = render_annotated_pipeline(src, out, predictor=predictor)
        t = center / PLAYBACK_FPS
        hit = [e for e in res["events"] if e["start_sec"] <= t <= e["end_sec"]]
        ok = bool(hit) and hit[0]["team"] == team and hit[0]["event"] == event
        print(label, s["clip"], [(e["team"], e["event"]) for e in hit], "OK" if ok else "mismatch - no clip attached", flush=True)
        s["clip_video"] = None
        if ok:
            name = f"assets/group/{label.replace('-', '_')}.mp4"
            shutil.move(out, os.path.join(ROOT, "frontend", name))
            s["clip_video"] = name
    json.dump(data, open(DATA, "w"), indent=1)


if __name__ == "__main__":
    main()
