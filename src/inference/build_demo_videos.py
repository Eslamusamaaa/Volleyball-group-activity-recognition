"""Builds genuinely smooth, strictly-contiguous (zero gap, zero jump, zero duplicate)
rally videos from the dataset frames.

Every single frame in these videos is physically adjacent in time (cur == prev + 1).
No jumps, no cuts, no repeating frames, no stuttering.
"""
import os
import sys
import shutil
import subprocess
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.annotations import load_annotations
VIDEOS_DIR = os.path.join(ROOT, "volleyball_", "videos")
OUT_DIR = os.path.join(ROOT, "production Data", "from our dataset")
DEMO_FPS = 25.0

VENV_FFMPEG = os.path.join(ROOT, ".venv", "Scripts", "ffmpeg.exe")
FFMPEG_EXE = VENV_FFMPEG if os.path.exists(VENV_FFMPEG) else (shutil.which("ffmpeg") or "ffmpeg")


def write_frames_to_mp4(frame_paths, out_path, fps=DEMO_FPS):
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
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart",
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


def build_pure_contiguous_dataset_videos():
    os.makedirs(OUT_DIR, exist_ok=True)
    ann = load_annotations(VIDEOS_DIR)

    all_segments = []
    for vid in sorted(int(d) for d in os.listdir(VIDEOS_DIR) if d.isdigit()):
        vdir = os.path.join(VIDEOS_DIR, str(vid))
        clip_ids = sorted(int(c) for c in os.listdir(vdir) if c.isdigit())
        frames_by_num = {}
        for clip in clip_ids:
            cdir = os.path.join(vdir, str(clip))
            for fname in os.listdir(cdir):
                if fname.endswith(".jpg"):
                    fnum = int(os.path.splitext(fname)[0])
                    frames_by_num.setdefault(fnum, os.path.join(cdir, fname))
        nums = sorted(frames_by_num)
        if not nums:
            continue

        # Strictly contiguous run: cur == prev + 1 (ZERO frame gap)
        run = [nums[0]]
        for prev, cur in zip(nums, nums[1:]):
            if cur == prev + 1:
                run.append(cur)
            else:
                all_segments.append((vid, run, [frames_by_num[n] for n in run]))
                run = [cur]
        all_segments.append((vid, run, [frames_by_num[n] for n in run]))

    # Sort by duration
    all_segments.sort(key=lambda s: -len(s[1]))

    # Pick top 20 longest contiguous runs
    top_segments = all_segments[:25]
    print(f"Building {len(top_segments)} strictly contiguous, zero-gap match videos...")

    for i, (vid, fnums, fpaths) in enumerate(top_segments):
        start_f, end_f = fnums[0], fnums[-1]
        dur_sec = len(fnums) / DEMO_FPS

        # Find annotations in this window
        clips_in_seg = [(int(c), data['category']) for (v, c), data in ann.items()
                        if v == vid and start_f <= int(c) <= end_f]
        clips_in_seg.sort()
        cats = [cat for _, cat in clips_in_seg]
        has_win = any('winpoint' in c for c in cats)

        tag = "winpoint" if has_win else "rally"
        out_name = f"match_v{vid}_seg{i+1}_{tag}_{len(cats)}plays_{dur_sec:.1f}s.mp4"
        out_path = os.path.join(OUT_DIR, out_name)

        print(f"[{i+1}/{len(top_segments)}] Rendering {out_name} ({len(fnums)} frames, {dur_sec:.1f}s, plays={len(cats)})...")
        write_frames_to_mp4(fpaths, out_path)

    print(f"\nAll done! Saved {len(top_segments)} strictly smooth videos in:\n{OUT_DIR}")


if __name__ == "__main__":
    build_pure_contiguous_dataset_videos()
