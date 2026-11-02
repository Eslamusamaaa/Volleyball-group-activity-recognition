"""Section 10, stage 4: turn the sliding window's per-window predictions
into a merged, timestamped event list -- the actual final deliverable
(video in, JSON timeline out).
"""
import argparse
import json

from .video_io import get_fps
from .sliding_window import WindowPredictor, iter_windows


def merge_windows_into_events(raw, min_confidence=0.40, max_winpoint_frames=100):
    """raw: list of {"start_frame", "end_frame", "scene_class", "team",
    "event", "scene_confidence", "players", ...} in time order.
    Merges consecutive windows that predicted the same scene class into one
    continuous event instead of reporting duplicates.
    
    Filters out low-confidence idle windows (< min_confidence) to avoid
    falsely labeling dead ball periods or walking as winpoints.
    """
    events = []
    for w in raw:
        conf = w.get("scene_confidence", 0.0)
        # Filter weak/uncertain predictions during non-action lulls
        if conf < min_confidence:
            continue

        is_winpoint = "winpoint" in w.get("event", "").lower()

        if events and events[-1]["scene_class"] == w["scene_class"]:
            # If it's a winpoint, avoid expanding past a realistic celebration duration (~3-4 sec)
            curr_dur_frames = w["end_frame"] - events[-1]["start_frame"]
            if is_winpoint and curr_dur_frames > max_winpoint_frames:
                # Cap the winpoint and start a fresh state for subsequent activity
                events.append({
                    "start_frame": w["start_frame"], "end_frame": w["end_frame"],
                    "team": w["team"], "event": w["event"], "scene_class": w["scene_class"],
                    "confidence": conf, "windows": 1,
                })
            else:
                events[-1]["end_frame"] = w["end_frame"]
                events[-1]["confidence"] = max(events[-1]["confidence"], conf)
                events[-1]["windows"] += 1
        else:
            events.append({
                "start_frame": w["start_frame"], "end_frame": w["end_frame"],
                "team": w["team"], "event": w["event"], "scene_class": w["scene_class"],
                "confidence": conf, "windows": 1,
            })
    return events


def build_timeline(video_path, n_frames=9, stride=9, device=None, min_confidence=0.40):
    fps = get_fps(video_path)
    predictor = WindowPredictor(device=device)

    raw = []
    for start_f, end_f, pred in iter_windows(video_path, predictor, n_frames=n_frames, stride=stride):
        if pred is None:
            continue
        raw.append({"start_frame": start_f, "end_frame": end_f, **pred})

    events = merge_windows_into_events(raw, min_confidence=min_confidence)
    for e in events:
        e["start_sec"] = round(e.pop("start_frame") / fps, 2)
        e["end_sec"] = round((e.pop("end_frame") + 1) / fps, 2)
        e.pop("scene_class", None)
        e.pop("windows", None)

    return {"fps": fps, "events": events}


def main():
    ap = argparse.ArgumentParser(description="Video -> timestamped event timeline (project brief Section 10).")
    ap.add_argument("video", help="path to a video file, or a dataset clip folder for testing")
    ap.add_argument("--n-frames", type=int, default=9)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--min-conf", type=float, default=0.40)
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = ap.parse_args()

    result = build_timeline(args.video, n_frames=args.n_frames, stride=args.stride, min_confidence=args.min_conf)
    out = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out} ({len(result['events'])} events)")
    else:
        print(out)


if __name__ == "__main__":
    main()
