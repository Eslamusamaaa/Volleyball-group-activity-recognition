"""Build a small, usable set of test videos for the Section 10 app from
this project's own real dataset frames -- NOT a substitute for a
genuinely new raw match video (the pipeline hasn't been tested on one
yet, see HANDOFF.md), but a legitimate way to exercise the full pipeline
on real volleyball footage without downloading copyrighted broadcast
content.

A video's clip folders are NOT disjoint windows -- each clip folder holds
~41 frames centered on its own annotated frame, and neighboring clips'
windows overlap heavily (e.g. clip 3646 spans frames 3626-3666, clip 3656
spans 3636-3676: frames 3636-3666 exist in both folders). Concatenating
clip folders in sequence replays the same physical frames over and over,
which is exactly the kind of degenerate, repeated-frame input the
sliding-window model was never trained on -- and simply dumping every
genuinely-contiguous segment from every one of the 55 videos produces
thousands of 1-7 second files, which isn't usable either.

Fixed by: de-duplicating every video's frames by frame number, splitting
at gaps into contiguous runs, then keeping only the N LONGEST runs across
the WHOLE dataset (default 50) -- a reasonable number of genuinely smooth,
no-duplicate, no-jump-cut clips, biased toward the longest (most useful
for testing) ones available anywhere in the data.
"""
import argparse
import os
import shutil
import subprocess

import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEOS_DIR = os.path.join(ROOT, "volleyball_", "videos")
OUT_DIR = os.path.join(ROOT, "production Data", "from our dataset")
DEMO_FPS = 25.0  # assumed playback rate for this synthetic stitch

FFMPEG_EXE = os.path.join(ROOT, ".venv", "Scripts", "ffmpeg.exe")
if not os.path.exists(FFMPEG_EXE):
    FFMPEG_EXE = shutil.which("ffmpeg") or "ffmpeg"


def dataset_segments(video_id):
    """-> list of (video_id, seg_idx, [frame_paths in order]) for every
    genuinely contiguous (zero-gap) run of real frames in this video."""
    video_dir = os.path.join(VIDEOS_DIR, str(video_id))
    if not os.path.isdir(video_dir):
        return []
    clip_ids = sorted((int(c) for c in os.listdir(video_dir) if c.isdigit()))

    frames_by_number = {}
    for clip in clip_ids:
        clip_dir = os.path.join(video_dir, str(clip))
        for fname in os.listdir(clip_dir):
            if not fname.endswith(".jpg"):
                continue
            fnum = int(os.path.splitext(fname)[0])
            frames_by_number.setdefault(fnum, os.path.join(clip_dir, fname))

    ordered_nums = sorted(frames_by_number)
    if not ordered_nums:
        return []

    segments, run = [], [ordered_nums[0]]
    for prev, cur in zip(ordered_nums, ordered_nums[1:]):
        if cur == prev + 1:
            run.append(cur)
        else:
            segments.append(run)
            run = [cur]
    segments.append(run)

    return [(video_id, i, [frames_by_number[n] for n in seg]) for i, seg in enumerate(segments)]


def write_segment(frame_paths, out_path):
    first = cv2.imread(frame_paths[0])
    if first is None:
        return 0
    h, w = first.shape[:2]
    cmd = [
        FFMPEG_EXE, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(DEMO_FPS),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=50, help="how many of the longest segments to keep, dataset-wide")
    ap.add_argument("videos", nargs="*", type=int, help="limit the scan to these video IDs (default: all 55)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    video_ids = args.videos or sorted(int(d) for d in os.listdir(VIDEOS_DIR) if d.isdigit())

    print(f"scanning {len(video_ids)} videos for contiguous runs ...")
    all_segments = []
    for vid in video_ids:
        all_segments.extend(dataset_segments(vid))
    print(f"found {len(all_segments)} contiguous runs total")

    all_segments.sort(key=lambda s: -len(s[2]))
    top = all_segments[: args.top_n]

    for vid, seg_idx, paths in top:
        secs = len(paths) / DEMO_FPS
        out_path = os.path.join(OUT_DIR, f"video_{vid}_seg{seg_idx}_{secs:.1f}s.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
            print(f"video {vid} seg {seg_idx}: already built ({secs:.1f}s), skipping")
            continue
        n = write_segment(paths, out_path)
        print(f"video {vid} seg {seg_idx}: {n} contiguous frames, {secs:.1f}s -> {out_path}")

    print(f"done -> {OUT_DIR} ({len(top)} videos kept, longest {len(top[0][2]) / DEMO_FPS:.1f}s, "
          f"shortest {len(top[-1][2]) / DEMO_FPS:.1f}s)")


if __name__ == "__main__":
    main()
