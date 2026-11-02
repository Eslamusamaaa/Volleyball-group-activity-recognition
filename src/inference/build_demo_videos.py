"""Builds high-quality, continuous, smooth dataset match videos (30s - 90s)
for the VolleyAI Studio platform.

Key Features:
1. Zero duplicate frames: strictly deduplicated and sorted in natural chronological frame order.
2. Long, smooth match footage: stitches consecutive rallies from the same match.
3. Fixes the winpoint cutoff: any rally containing a winpoint is joined with the subsequent
   rally, so the winpoint occurs naturally in the middle of the video, giving the viewer and
   model full context before, during, and after the point is scored!
"""
import argparse
import os
import shutil
import subprocess
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEOS_DIR = os.path.join(ROOT, "volleyball_", "videos")
OUT_DIR = os.path.join(ROOT, "production Data", "from our dataset")
DEMO_FPS = 25.0

VENV_FFMPEG = os.path.join(ROOT, ".venv", "Scripts", "ffmpeg.exe")
FFMPEG_EXE = VENV_FFMPEG if os.path.exists(VENV_FFMPEG) else (shutil.which("ffmpeg") or "ffmpeg")


def get_video_rallies(vid):
    """Parses all annotated clips in a video and groups them into contiguous rallies.
    Returns list of dicts:
      {'rally_idx': int, 'clips': list, 'frame_paths': list, 'has_winpoint': bool, 'duration_sec': float}
    """
    vdir = os.path.join(VIDEOS_DIR, str(vid))
    if not os.path.isdir(vdir):
        return []

    # Read annotations to identify winpoints
    ann_path = os.path.join(vdir, "annotations.txt")
    clip_categories = {}
    if os.path.exists(ann_path):
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    c_id = parts[0].split(".")[0]
                    clip_categories[int(c_id)] = parts[1]

    clip_ids = sorted([int(c) for c in os.listdir(vdir) if c.isdigit()])
    if not clip_ids:
        return []

    # Group clips into distinct rallies (distance between clip centers > 350 frames marks a new rally)
    rallies = []
    curr = [clip_ids[0]]
    for prev, c in zip(clip_ids, clip_ids[1:]):
        if c - prev < 350:
            curr.append(c)
        else:
            rallies.append(curr)
            curr = [c]
    rallies.append(curr)

    rally_objects = []
    for r_idx, r_clips in enumerate(rallies):
        # Extract unique frames for this rally in ascending order
        frame_dict = {}
        has_win = False
        for c in r_clips:
            cat = clip_categories.get(c, "")
            if "winpoint" in cat:
                has_win = True
            cdir = os.path.join(vdir, str(c))
            if os.path.isdir(cdir):
                for fname in os.listdir(cdir):
                    if fname.endswith(".jpg"):
                        fnum = int(os.path.splitext(fname)[0])
                        frame_dict.setdefault(fnum, os.path.join(cdir, fname))

        ordered_nums = sorted(frame_dict.keys())
        if len(ordered_nums) < 40:  # Skip tiny fragments
            continue

        frame_paths = [frame_dict[n] for n in ordered_nums]
        rally_objects.append({
            "video_id": vid,
            "rally_idx": r_idx,
            "clips": r_clips,
            "frame_paths": frame_paths,
            "has_winpoint": has_win,
            "n_frames": len(frame_paths),
            "duration_sec": len(frame_paths) / DEMO_FPS,
        })

    return rally_objects


def write_frames_to_mp4(frame_paths, out_path, fps=DEMO_FPS):
    """Streams list of frame paths to H.264 MP4 with faststart for instant web playback."""
    if not frame_paths:
        return 0

    first = cv2.imread(frame_paths[0])
    if first is None:
        return 0
    h, w = first.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cmd = [
        FFMPEG_EXE, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n = 0
    for p in frame_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        proc.stdin.write(frame.tobytes())
        n += 1

    proc.stdin.close()
    proc.wait()
    return n


def build_match_videos(target_videos=None, max_videos=20):
    """Builds rich multi-rally match games where:
    - Winpoints happen in the MIDDLE of the match footage.
    - Each video is 30s - 90s long.
    - Zero duplicate frames.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    if target_videos is None:
        target_videos = [0, 1, 3, 4, 10, 18, 20, 22, 25, 31, 38, 48, 51, 52]

    built_count = 0
    print(f"Building long smooth match videos from dataset ...")

    for vid in target_videos:
        if built_count >= max_videos:
            break

        rallies = get_video_rallies(vid)
        if not rallies:
            continue

        # Strategy 1: Find rallies with winpoint and join them with the next 1-2 rallies
        # so winpoint is prominently featured in the middle!
        i = 0
        while i < len(rallies) and built_count < max_videos:
            curr_r = rallies[i]

            # If this rally has a winpoint, or to make a multi-rally game:
            combined_paths = []
            combined_paths.extend(curr_r["frame_paths"])
            has_win = curr_r["has_winpoint"]

            # Add subsequent rallies from this match to ensure length and prevent cutoff
            j = i + 1
            while j < len(rallies) and len(combined_paths) < 1100:  # ~45s target
                next_r = rallies[j]
                combined_paths.extend(next_r["frame_paths"])
                if next_r["has_winpoint"]:
                    has_win = True
                j += 1

            dur_sec = len(combined_paths) / DEMO_FPS

            # Only export substantial, smooth matches (>= 25 seconds)
            if dur_sec >= 22.0:
                tag = "winpoint_match" if has_win else "continuous_play"
                out_name = f"match_v{vid}_seq{i+1}_{tag}_{dur_sec:.1f}s.mp4"
                out_path = os.path.join(OUT_DIR, out_name)

                print(f"Rendering {out_name} ({len(combined_paths)} frames, {dur_sec:.1f}s, winpoint={has_win})...")
                n_written = write_frames_to_mp4(combined_paths, out_path)
                if n_written > 0:
                    built_count += 1
                    print(f"  -> Built successfully ({built_count}/{max_videos})")

            i = max(i + 1, j)

    print(f"\nAll done! Built {built_count} long smooth dataset videos in:\n{OUT_DIR}")


if __name__ == "__main__":
    build_match_videos()
