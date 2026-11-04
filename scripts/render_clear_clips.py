"""Render the site's group-activity clips from outputs/clear_clips.json (scripts/find_clear_clips.py).

For each of the 8 classes the chosen test clip is the one whose model banner agrees with the dataset
label across the longest run of 9-frame windows that includes the annotated middle frame. The clip is
rendered with the full pipeline, trimmed to exactly that run (so the banner shows the class from the
first frame to the last) and slowed to about 12 s with motion interpolation. Also writes the posters,
frontend/data/showcase.json and the group list of frontend/data/videos.json.

    python scripts/render_clear_clips.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from render_group_examples import PLAYBACK_FPS, stitch  # noqa: E402
from src.inference import annotator  # noqa: E402
from src.inference.sliding_window import WindowPredictor  # noqa: E402

FRONT = os.path.join(ROOT, "frontend")
GROUP_DIR = os.path.join(FRONT, "assets", "group")
SCAN = os.path.join(ROOT, "outputs", "clear_clips.json")
SHOWCASE = os.path.join(FRONT, "data", "showcase.json")
VIDEOS = os.path.join(FRONT, "data", "videos.json")
FFMPEG = annotator.FFMPEG_EXE
TARGET_SEC = 12.0
MAX_SLOW = 4.5
MIN_CONF = 0.9
ORDER = ["r_spike", "l_set", "r-pass", "l_winpoint", "l-pass", "r_set", "l-spike", "r_winpoint"]
norm = lambda s: s.replace("-", "_")
# Hand-picked clips where the scan's best choice is not the most readable on screen (the best-agreeing
# l_winpoint clip showed players standing still; this one shows the left team celebrating the point).
OVERRIDE = {"l_winpoint": ("25", "36225")}
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]


def central_run(c, label):
    """(first, last) window indices of the agreeing run that contains the middle window, or None."""
    ok = [bool(w["cls"]) and norm(w["cls"]) == norm(label) and w["conf"] >= MIN_CONF for w in c["windows"]]
    mid = 2
    if len(ok) <= mid or not ok[mid]:
        return None
    a = b = mid
    while a > 0 and ok[a - 1]:
        a -= 1
    while b < len(ok) - 1 and ok[b + 1]:
        b += 1
    return a, b


def pick(rows, label):
    best = None
    for c in rows:
        if label in OVERRIDE and (str(c["video"]), str(c["clip"])) != OVERRIDE[label]:
            continue
        run = central_run(c, label)
        if not run:
            continue
        a, b = run
        confs = [w["conf"] for w in c["windows"][a:b + 1]]
        key = (b - a + 1, sum(confs) / len(confs))
        if best is None or key > best[0]:
            best = (key, c, run)
    return best


def trim_and_slow(src, dst, first_frame, n_frames):
    start, length = first_frame / PLAYBACK_FPS, n_frames / PLAYBACK_FPS
    factor = min(TARGET_SEC / length, MAX_SLOW)   # a short run (a spike lasts well under a second) is not stretched past 4.5x
    vf = f"setpts={factor:.4f}*PTS,minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    tmp = dst + ".tmp.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", src, "-vf", vf, "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "medium", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, dst)


def poster(mp4, dest):
    from PIL import Image
    png = dest + ".png"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "5.0", "-i", mp4, "-frames:v", "1", "-q:v", "2", png], check=True)
    Image.open(png).convert("RGB").save(dest + ".webp", "WEBP", quality=88, method=6)
    os.remove(png)


def main():
    scan = json.load(open(SCAN))
    predictor = WindowPredictor()
    tmp = tempfile.mkdtemp()
    done = {c["label"]: c for c in json.load(open(SHOWCASE))} if (ONLY and os.path.exists(SHOWCASE)) else {}
    showcase = []
    for label in ORDER:
        if ONLY and label not in ONLY:
            if label in done:
                showcase.append(done[label])
            continue
        picked = pick(scan[label], label)
        if not picked:
            print(label, "NO clip with the middle window agreeing", flush=True)
            continue
        (length, conf), c, (a, b) = picked
        vid, clip = c["video"], c["clip"]
        src, raw = os.path.join(tmp, "src.mp4"), os.path.join(tmp, "raw.mp4")
        stitch(vid, clip, src)
        res = annotator.render_annotated_pipeline(src, raw, predictor=predictor)
        first, last = c["windows"][a]["start"], c["windows"][b]["end"]
        name = label.replace("-", "_")
        final = os.path.join(GROUP_DIR, name + ".mp4")
        trim_and_slow(raw, final, first, last - first + 1)
        poster(final, os.path.join(GROUP_DIR, name + "_poster"))
        showcase.append({"label": label, "team": "left" if label[0] == "l" else "right", "event": label[2:],
                         "video_file": f"assets/group/{name}.mp4", "poster": f"assets/group/{name}_poster.webp",
                         "source_video": vid, "source_clip": clip, "confidence": round(conf, 3)})
        print(label, vid, clip, f"windows {a}-{b} frames {first}-{last} conf {conf:.3f}", flush=True)
    json.dump(showcase, open(SHOWCASE, "w"), indent=1)
    old = json.load(open(VIDEOS)) if os.path.exists(VIDEOS) else {"player": []}
    json.dump({"group": showcase, "player": old.get("player", [])}, open(VIDEOS, "w"), indent=1)
    print("done", [s["label"] for s in showcase])


if __name__ == "__main__":
    main()
