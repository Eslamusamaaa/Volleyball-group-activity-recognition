"""Annotated video generator for the Volleyball Activity Recognition Platform.
Renders:
1. Per-player detection bounding boxes with individual classified actions (SPIKING, SETTING, BLOCKING, DIGGING, etc.)
2. Prominent top-center broadcast scoreboard HUD for team-level group activity & winpoints
3. Clean match clock and telemetry
Directly outputs an H.264 MP4 that streams seamlessly in web browsers.
"""
import os
import shutil
import subprocess
from collections import deque
import cv2
import numpy as np

from .video_io import get_fps, iter_video_frames
from .tracker import PlayerTracker
from .sliding_window import WindowPredictor
from .timeline import merge_windows_into_events

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VENV_FFMPEG = os.path.join(ROOT, ".venv", "Scripts", "ffmpeg.exe")
FFMPEG_EXE = VENV_FFMPEG if os.path.exists(VENV_FFMPEG) else (shutil.which("ffmpeg") or "ffmpeg")

# BGR color palette for teams
TEAM_COLORS = {
    "left":  (255, 175, 40),   # Electric Cyan / Sky Blue
    "right": (40, 130, 255),   # Sunset Orange / Amber
}
DEFAULT_BOX_COLOR = (70, 220, 120)   # Neon Emerald
HUD_BG_DARK   = (15, 18, 26)
HUD_BORDER    = (50, 65, 90)
ACCENT_POINT  = (0, 215, 255)        # Vivid Gold

# Specific colors per player individual action
ACTION_BADGE_COLORS = {
    "spiking":  (35, 50, 245),    # Vibrant Crimson Red
    "blocking": (210, 60, 180),   # Electric Violet / Magenta
    "setting":  (245, 190, 30),   # Neon Sky Blue
    "digging":  (50, 215, 90),    # Emerald Green
    "jumping":  (10, 215, 255),   # Glowing Gold
    "falling":  (110, 100, 240),  # Rose
    "moving":   (200, 160, 60),   # Cyan Slate
    "waiting":  (160, 150, 140),  # Muted Slate
    "standing": (140, 140, 140),  # Neutral Gray
}

# Clean action display names
ACTION_DISPLAY_NAMES = {
    "spiking":  "SPIKE",
    "blocking": "BLOCK",
    "setting":  "SET",
    "digging":  "DIG",
    "jumping":  "JUMP",
    "falling":  "DIVE",
    "moving":   "MOVE",
    "waiting":  "READY",
    "standing": "STAND",
}

# Inference resolution scale
INFER_MAX_W = 640
INFER_MAX_H = 480


def _resize_for_inference(frame):
    h, w = frame.shape[:2]
    scale = min(INFER_MAX_W / w, INFER_MAX_H / h, 1.0)
    if scale >= 1.0:
        return frame, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale


def _scale_dets(dets, scale):
    if scale == 1.0:
        return dets
    inv = 1.0 / scale
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        out.append({**d, "bbox": [x1 * inv, y1 * inv, x2 * inv, y2 * inv]})
    return out


def draw_top_broadcast_hud(frame, event=None, frame_idx=0, fps=25.0, n_players=0):
    """Draws a prominent, high-visibility broadcast scoreboard banner at the TOP CENTER."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Time string
    time_sec = frame_idx / fps if fps > 0 else 0
    mins  = int(time_sec // 60)
    secs  = int(time_sec % 60)
    millis = int((time_sec - int(time_sec)) * 10)
    time_str = f"{mins:02d}:{secs:02d}.{millis:01d}"

    # Top banner bar background
    bar_h = 56
    cv2.rectangle(overlay, (0, 0), (w, bar_h), HUD_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    # Bottom accent line of top bar
    cv2.line(frame, (0, bar_h), (w, bar_h), HUD_BORDER, 2)

    # 1. Left side: Match Clock badge
    cv2.rectangle(frame, (16, 10), (135, 46), (28, 35, 48), -1)
    cv2.rectangle(frame, (16, 10), (135, 46), HUD_BORDER, 1)
    cv2.putText(frame, time_str, (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 230, 245), 2, cv2.LINE_AA)

    # 2. Right side: Branding & Player count
    brand_text = f"VOLLEYAI STUDIO  |  {n_players} PLAYERS"
    cv2.putText(frame, brand_text, (w - 320, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 170, 200), 1, cv2.LINE_AA)

    # 3. CENTER: Prominent Main Event Pill Banner
    if event:
        team     = event.get("team", "unknown").upper()
        act_raw  = event.get("event", "rally").lower()
        conf     = int(event.get("confidence", 1.0) * 100)
        is_point = ("winpoint" in act_raw or "point" in act_raw)

        if is_point:
            # Huge, glowing gold championship point banner
            center_text = f"★  POINT SCORED  —  {team} TEAM  ★"
            (tw, th), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            pill_w = tw + 48
            pill_x = (w - pill_w) // 2
            # Gold pill
            cv2.rectangle(frame, (pill_x, 6), (pill_x + pill_w, 50), ACCENT_POINT, -1)
            cv2.rectangle(frame, (pill_x, 6), (pill_x + pill_w, 50), (255, 255, 255), 2)
            cv2.putText(frame, center_text, (pill_x + 24, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (10, 15, 20), 2, cv2.LINE_AA)
        else:
            # High-visibility team tactical play pill
            team_color = TEAM_COLORS.get(event.get("team", "").lower(), DEFAULT_BOX_COLOR)
            display_act = act_raw.replace("-", " ").replace("_", " ").upper()
            center_text = f"{team} TEAM  •  {display_act}  ({conf}%)"
            (tw, th), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
            pill_w = tw + 44
            pill_x = (w - pill_w) // 2
            # Team color pill with dark text or white text
            cv2.rectangle(frame, (pill_x, 8), (pill_x + pill_w, 48), team_color, -1)
            cv2.rectangle(frame, (pill_x, 8), (pill_x + pill_w, 48), (255, 255, 255), 1)
            cv2.putText(frame, center_text, (pill_x + 22, 37),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (15, 20, 25), 2, cv2.LINE_AA)
    else:
        # Neutral active rally tracking
        center_text = "🏐  LIVE RALLY IN PROGRESS  •  TRACKING ALL PLAYERS"
        (tw, th), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        pill_w = tw + 36
        pill_x = (w - pill_w) // 2
        cv2.rectangle(frame, (pill_x, 10), (pill_x + pill_w, 46), (26, 33, 46), -1)
        cv2.rectangle(frame, (pill_x, 10), (pill_x + pill_w, 46), HUD_BORDER, 1)
        cv2.putText(frame, center_text, (pill_x + 18, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 185, 215), 2, cv2.LINE_AA)


def draw_player_box_and_action(frame, bbox, track_id, action="standing", team="neutral"):
    """Draws bounding box with a floating action badge above the player:
    e.g. [ P#4 • SPIKE ] in vivid color-coded badges!
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    action_clean = action.lower() if action else "standing"
    badge_color = ACTION_BADGE_COLORS.get(action_clean, TEAM_COLORS.get(team, DEFAULT_BOX_COLOR))
    display_action = ACTION_DISPLAY_NAMES.get(action_clean, action_clean.upper())

    # Thicker, prominent bounding box for key active players
    is_key_action = action_clean in ("spiking", "blocking", "setting", "digging", "jumping")
    box_thickness = 3 if is_key_action else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), badge_color, box_thickness)

    # Corner brackets for futuristic sports broadcast look
    c_len = min(14, (x2 - x1) // 3, (y2 - y1) // 3)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)

    # Floating Action Tag above player box
    tag = f"P#{track_id} • {display_action}"
    font_scale = 0.50 if is_key_action else 0.44
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

    pill_top = max(4, y1 - th - 10)
    pill_bottom = max(th + 10, y1 - 2)
    pill_right = min(frame.shape[1] - 4, x1 + tw + 12)

    # Solid pill background
    cv2.rectangle(frame, (x1, pill_top), (pill_right, pill_bottom), badge_color, -1)
    cv2.rectangle(frame, (x1, pill_top), (pill_right, pill_bottom), (20, 20, 20), 1)

    # Black text for high contrast on bright pills
    text_color = (10, 15, 20)
    cv2.putText(frame, tag, (x1 + 6, pill_bottom - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)


def render_annotated_pipeline(video_path, out_path, predictor=None,
                              n_frames=9, stride=9, progress_callback=None):
    """Full 2-pass pipeline:
    Pass 1: tracks players, runs hierarchical B8-mt model for group activities + per-player actions.
    Pass 2: renders high-visibility top broadcast HUD + individual player action badges via ffmpeg H.264.
    """
    fps = get_fps(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if predictor is None:
        predictor = WindowPredictor()

    tracker = PlayerTracker()
    tracker.reset()

    frame_dets = []
    buf = deque(maxlen=n_frames)
    raw_windows = []
    # Map frame index to dict of {track_id: action_str}
    frame_player_actions = {}

    # ------------------------------------------------------------------ Pass 1
    for idx, frame in iter_video_frames(video_path):
        small, scale = _resize_for_inference(frame)
        small_dets = tracker.update(small)
        dets = _scale_dets(small_dets, scale)
        frame_dets.append(dets)

        buf.append((idx, small, small_dets))

        if len(buf) == n_frames and (idx - n_frames + 1) % stride == 0:
            try:
                pred = predictor.predict_window(buf)
            except StopIteration:
                pred = None
            except Exception:
                pred = None

            if pred:
                w_start = buf[0][0]
                w_end = buf[-1][0]
                raw_windows.append({"start_frame": w_start, "end_frame": w_end, **pred})
                # Store per-player actions across this window
                for player in pred.get("players", []):
                    tid = player.get("track_id")
                    act = player.get("action", "standing")
                    for f in range(w_start, w_end + 1):
                        if f not in frame_player_actions:
                            frame_player_actions[f] = {}
                        frame_player_actions[f][tid] = act

        if progress_callback and total_frames > 0 and idx % 10 == 0:
            progress_callback(
                min(45, int(45 * idx / total_frames)),
                "Tracking & Action Recognition"
            )

    events = merge_windows_into_events(raw_windows)

    # ------------------------------------------------------------------ Pass 2
    first_frame = None
    for _, f in iter_video_frames(video_path):
        first_frame = f
        break

    if first_frame is None:
        raise RuntimeError("No frames found in video")

    h, w = first_frame.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    cmd = [
        FFMPEG_EXE, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    event_idx = 0
    n_events = len(events)
    # Track latest known action per player for persistence
    latest_player_actions = {}

    for idx, frame in iter_video_frames(video_path):
        while event_idx < n_events - 1 and idx > events[event_idx]["end_frame"]:
            event_idx += 1
        active = (
            events[event_idx]
            if events and events[event_idx]["start_frame"] <= idx <= events[event_idx]["end_frame"]
            else None
        )

        dets = frame_dets[idx] if idx < len(frame_dets) else []
        active_team = active["team"] if active else "neutral"

        # Update latest known actions for current frame
        if idx in frame_player_actions:
            latest_player_actions.update(frame_player_actions[idx])

        # Draw player bounding boxes + classified actions
        for d in dets:
            tid = d.get("track_id", "")
            action = latest_player_actions.get(tid, "standing")
            draw_player_box_and_action(frame, d["bbox"], tid, action=action, team=active_team)

        # Draw Prominent Top Broadcast Scoreboard HUD
        draw_top_broadcast_hud(frame, active, idx, fps, n_players=len(dets))

        proc.stdin.write(frame.tobytes())

        if progress_callback and total_frames > 0 and idx % 20 == 0:
            progress_callback(min(98, 45 + int(53 * idx / total_frames)), "Rendering Broadcast Video")

    proc.stdin.close()
    proc.wait()

    if progress_callback:
        progress_callback(100, "Completed")

    # Format events for API response
    formatted_events = []
    for e in events:
        formatted_events.append({
            "team":        e["team"],
            "event":       e["event"],
            "confidence":  round(e["confidence"], 3),
            "start_sec":   round(e["start_frame"] / fps, 2),
            "end_sec":     round((e["end_frame"] + 1) / fps, 2),
            "start_frame": e["start_frame"],
            "end_frame":   e["end_frame"],
        })

    return {
        "annotated_path": out_path,
        "fps":            fps,
        "total_frames":   total_frames,
        "duration_sec":   round(total_frames / fps, 2) if fps > 0 else 0,
        "events":         formatted_events,
    }
