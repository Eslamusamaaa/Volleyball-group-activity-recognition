"""Burn the detected events directly into a copy of the video -- player
boxes drawn live, plus a caption banner ("RIGHT TEAM: SPIKE", "POINT --
LEFT TEAM!") that updates as the game moves through different events.

Two passes over the video: pass 1 tracks players and runs the sliding
window to get events (same as timeline.py); pass 2 re-reads the video and
draws, since holding every raw frame in memory for a multi-minute video
would use too much RAM -- only the lightweight per-frame box list and the
merged event list are kept between passes.
"""
import argparse
import os
from collections import deque

import cv2

from .video_io import get_fps, iter_video_frames
from .tracker import PlayerTracker
from .sliding_window import WindowPredictor
from .timeline import merge_windows_into_events

TEAM_COLOR = {"left": (74, 148, 255), "right": (255, 176, 66)}  # BGR
BOX_COLOR = (90, 220, 90)
BANNER_BG = (20, 20, 20)


def _draw_banner(frame, text, highlight=False):
    h, w = frame.shape[:2]
    bar_h = 56 if highlight else 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), BANNER_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    color = (60, 220, 255) if highlight else (235, 235, 235)
    scale = 1.0 if highlight else 0.8
    cv2.putText(frame, text, (16, h - bar_h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def _event_text(event):
    team_label = f"{event['team'].upper()} TEAM"
    if event["event"] == "winpoint":
        return f"POINT -- {team_label}!", True
    return f"{team_label}: {event['event'].upper()} ({event['confidence'] * 100:.0f}%)", False


def render(video_path, out_path, n_frames=9, stride=9, device=None):
    fps = get_fps(video_path)
    predictor = WindowPredictor(device=device)
    tracker = PlayerTracker()
    tracker.reset()

    frame_dets = []
    buf = deque(maxlen=n_frames)
    raw_windows = []
    print("pass 1/2: tracking + predicting ...")
    for idx, frame in iter_video_frames(video_path):
        dets = tracker.update(frame)
        frame_dets.append(dets)
        buf.append((idx, frame, dets))
        if len(buf) == n_frames and (idx - n_frames + 1) % stride == 0:
            pred = predictor.predict_window(buf)
            if pred:
                raw_windows.append({"start_frame": buf[0][0], "end_frame": buf[-1][0], **pred})

    events = merge_windows_into_events(raw_windows)
    print(f"  {len(events)} events found")

    print("pass 2/2: rendering annotated video ...")
    writer = None
    event_idx = 0
    for idx, frame in iter_video_frames(video_path):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        while event_idx < len(events) - 1 and idx > events[event_idx]["end_frame"]:
            event_idx += 1
        active = (events[event_idx] if events and
                  events[event_idx]["start_frame"] <= idx <= events[event_idx]["end_frame"] else None)

        dets = frame_dets[idx] if idx < len(frame_dets) else []
        box_color = TEAM_COLOR.get(active["team"], BOX_COLOR) if active else BOX_COLOR
        for d in dets:
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        if active:
            text, highlight = _event_text(active)
            _draw_banner(frame, text, highlight)

        writer.write(frame)
    if writer is not None:
        writer.release()
    print(f"wrote {out_path}")
    return events


def main():
    ap = argparse.ArgumentParser(description="Render a copy of the video with live event captions burned in.")
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="default: <video>_annotated.mp4 next to the input")
    ap.add_argument("--n-frames", type=int, default=9)
    ap.add_argument("--stride", type=int, default=9)
    args = ap.parse_args()

    out_path = args.out or os.path.splitext(args.video)[0] + "_annotated.mp4"
    render(args.video, out_path, n_frames=args.n_frames, stride=args.stride)


if __name__ == "__main__":
    main()
