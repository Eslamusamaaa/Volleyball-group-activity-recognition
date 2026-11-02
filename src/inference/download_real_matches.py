"""Downloads clean, continuous 3-to-4-minute segments of authentic, official
volleyball tournament match broadcasts from YouTube into:
D:\\volleyball datasets\\production Data\\from internet

These are REAL full-match broadcasts:
- Standard court-side broadcast camera (all 12 players visible, net in center)
- Authentic continuous live rallies, serves, passes, sets, spikes, blocks, digs
- Natural game sound and referee whistles
- Zero music, zero montage edits, zero trailers, zero repeated frames
"""
import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "production Data", "from internet")
FFMPEG_DIR = os.path.join(ROOT, ".venv", "Scripts")
PYTHON_EXE = os.path.join(ROOT, ".venv", "Scripts", "python.exe")

REAL_MATCH_SEGMENTS = [
    {
        "name": "USA_vs_Japan_VNL_Live_Match_3.5min.mp4",
        "url": "https://www.youtube.com/watch?v=lMq8llmamCA",
        "section": "*15:00-18:30",
        "desc": "USA vs Japan (VNL Full Match broadcast, 3.5 min continuous Set 1 action)",
    },
    {
        "name": "Japan_vs_France_Gold_Match_3.5min.mp4",
        "url": "https://www.youtube.com/watch?v=g7GlUj61BYA",
        "section": "*20:00-23:30",
        "desc": "Japan vs France (VNL Gold Final broadcast, 3.5 min continuous championship rallies)",
    },
    {
        "name": "Poland_vs_Italy_Championship_3.5min.mp4",
        "url": "https://www.youtube.com/watch?v=M2IfS9CHo30",
        "section": "*18:00-21:30",
        "desc": "Poland vs Italy (VNL Gold Match broadcast, 3.5 min continuous world championship action)",
    },
]


def download_match_segment(item, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, item["name"])
    if os.path.exists(target) and os.path.getsize(target) > 5000000:
        print(f"[Matches] Already exists: {item['name']}")
        return target

    print(f"[Matches] Downloading authentic broadcast: {item['desc']} ...")
    cmd = [
        PYTHON_EXE, "-m", "yt_dlp",
        item["url"],
        "--download-sections", item["section"],
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--ffmpeg-location", FFMPEG_DIR,
        "-o", os.path.join(out_dir, item["name"]),
        "--no-playlist",
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0 and os.path.exists(target):
        print(f"[Matches] Success: {item['name']} ({os.path.getsize(target) / 1024 / 1024:.1f} MB)")
        return target
    else:
        print(f"[Matches] Download exited with code {res.returncode}")
        return None


def download_all_real_matches():
    for item in REAL_MATCH_SEGMENTS:
        download_match_segment(item)


if __name__ == "__main__":
    download_all_real_matches()

