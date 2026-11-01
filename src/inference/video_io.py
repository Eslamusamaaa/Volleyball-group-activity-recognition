"""Frame/FPS access for the Section 10 inference pipeline.

Two sources are supported with the same interface:
  - a real video file (.mp4/.avi/...), via cv2.VideoCapture
  - a directory of already-extracted numbered frame .jpg files (the format
    this project's own dataset ships in), for testing the pipeline without
    a real raw video on disk.

Project brief Section 7.3: frame rate is not uniform across all 55 source
videos and must never be assumed -- always read it from the file itself.
"""
import os
import cv2


def get_fps(video_path, default=25.0):
    """Real FPS of a video file, read with cv2 (Section 7.3). Falls back to
    `default` only if the file doesn't report a usable value (some codecs
    report 0), and says so -- never silently assumes a number otherwise."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        print(f"warning: {video_path} reported no usable FPS, falling back to {default}")
        return default
    return fps


def iter_video_frames(video_path):
    """Yield (frame_index, frame_bgr) for every frame of a real video file,
    in order, starting at 0."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def iter_frame_dir(dir_path):
    """Yield (frame_index, frame_bgr) for a directory of numbered .jpg frames
    (e.g. one of this dataset's own clip folders), sorted numerically --
    Section 7.1's exact pitfall: filenames are numeric strings and a plain
    lexicographic sort misorders them once digit counts differ."""
    names = [n for n in os.listdir(dir_path) if n.lower().endswith(".jpg")]
    names.sort(key=lambda n: int(os.path.splitext(n)[0]))
    for i, n in enumerate(names):
        frame = cv2.imread(os.path.join(dir_path, n))
        if frame is None:
            continue
        yield i, frame


def frame_source(path):
    """Dispatch: a directory -> iter_frame_dir, a file -> iter_video_frames."""
    if os.path.isdir(path):
        return iter_frame_dir(path)
    return iter_video_frames(path)
