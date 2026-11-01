"""Section 10, stage 4: turn the sliding window's per-window predictions
into a merged, timestamped event list -- the actual final deliverable
(video in, JSON timeline out).
"""
import argparse
import json

from .video_io import get_fps
from .sliding_window import WindowPredictor, iter_windows


def build_timeline(video_path, n_frames=9, stride=9, device=None):
    fps = get_fps(video_path)
    predictor = WindowPredictor(device=device)

    raw = []
    for start_f, end_f, pred in iter_windows(video_path, predictor, n_frames=n_frames, stride=stride):
        if pred is None:
            continue
        raw.append({"start_frame": start_f, "end_frame": end_f, **pred})

    events = []
    for w in raw:
        if events and events[-1]["scene_class"] == w["scene_class"]:
            # same predicted event as the previous window -- extend it
            # instead of reporting a duplicate (brief Section 10, stage 4)
            events[-1]["end_frame"] = w["end_frame"]
            events[-1]["confidence"] = max(events[-1]["confidence"], w["scene_confidence"])
            events[-1]["windows"] += 1
        else:
            events.append({
                "start_frame": w["start_frame"], "end_frame": w["end_frame"],
                "team": w["team"], "event": w["event"], "scene_class": w["scene_class"],
                "confidence": w["scene_confidence"], "windows": 1,
            })

    for e in events:
        e["start_sec"] = round(e.pop("start_frame") / fps, 2)
        e["end_sec"] = round((e.pop("end_frame") + 1) / fps, 2)
        e.pop("scene_class")
        e.pop("windows")

    return {"fps": fps, "events": events}


def main():
    ap = argparse.ArgumentParser(description="Video -> timestamped event timeline (project brief Section 10).")
    ap.add_argument("video", help="path to a video file, or a dataset clip folder for testing")
    ap.add_argument("--n-frames", type=int, default=9)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = ap.parse_args()

    result = build_timeline(args.video, n_frames=args.n_frames, stride=args.stride)
    out = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out} ({len(result['events'])} events)")
    else:
        print(out)


if __name__ == "__main__":
    main()
