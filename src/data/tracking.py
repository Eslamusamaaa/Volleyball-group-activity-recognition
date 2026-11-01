"""Load real per-frame player tracks from volleyball_tracking_annotation and
align them to the ground-truth players in annotations.txt, so the 9-frame
window can use each player's ACTUAL moving box per frame instead of reusing
the center frame's box (the paper tracks real tracklets with a Dlib tracker;
this matches that instead of the Section 7.2 choice (a) simplification).
"""
import os


def load_tracks(tracking_root, video, clip):
    """Return {track_id: {frame: (x, y, w, h)}} for one clip.

    On-disk format is "track_id x1 y1 x2 y2 frame ... action" (box corners,
    NOT x/y/width/height despite looking similar at a glance) -- verified by
    comparing against annotations.txt's ground-truth boxes for the same
    player: interpreting columns 2-5 as corners (then converting to w/h)
    lines up with the GT box, interpreting them as x/y/w/h directly does not
    (gives an implausibly tall box, ~4x too tall).
    """
    path = os.path.join(tracking_root, str(video), str(clip), f"{clip}.txt")
    tracks = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tid = int(parts[0])
            x1, y1, x2, y2 = map(int, parts[1:5])
            frame = int(parts[5])
            tracks.setdefault(tid, {})[frame] = (x1, y1, x2 - x1, y2 - y1)
    return tracks


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_players_to_tracks(gt_players, tracks, center_frame):
    """Match each ground-truth player (annotations.txt, box only at the
    center frame) to the track_id whose box at the center frame has the
    highest IoU overlap. Returns a list (same order as gt_players) of
    track_id or None if no track overlaps at all.
    """
    matches = []
    used = set()
    for action, x, y, w, h in gt_players:
        best_tid, best_iou = None, 0.0
        for tid, frames in tracks.items():
            if tid in used or center_frame not in frames:
                continue
            iou = _iou((x, y, w, h), frames[center_frame])
            if iou > best_iou:
                best_iou, best_tid = iou, tid
        if best_tid is not None and best_iou > 0.3:
            used.add(best_tid)
            matches.append(best_tid)
        else:
            matches.append(None)
    return matches


def player_box_per_frame(tracks, track_id, frame, fallback_box):
    """The player's real tracked box at `frame`, or `fallback_box` (the
    ground-truth center-frame box) if that track has no entry for this
    frame (tracker lost them, or frame outside the tracked range)."""
    if track_id is None:
        return fallback_box
    frames = tracks.get(track_id, {})
    return frames.get(frame, fallback_box)


# Per the official reference repo's data notes: these 8 videos are
# 1920x1080, every other video in the dataset is 1280x720.
_HD_VIDEOS = {2, 37, 38, 39, 40, 41, 44, 45}


def frame_size(video):
    return (1920, 1080) if int(video) in _HD_VIDEOS else (1280, 720)


def motion_features(tracks, track_id, frame_nums, fallback_box, video):
    """Cheap, free explicit-motion signal for a player across a frame window:
    frame-to-frame deltas (dx, dy, dw, dh), normalized by image size / the
    player's own box size. No CNN involved -- pure arithmetic on the
    already-loaded tracked boxes, aimed at the motion-defined action classes
    (jumping, moving, falling, digging) that a single-frame CNN feature
    struggles to tell apart from static ones (standing, waiting).

    Returns a (len(frame_nums), 4) list of [dx, dy, dw, dh] float deltas,
    where entry 0 is always [0, 0, 0, 0] (no previous frame to diff against).
    """
    img_w, img_h = frame_size(video)
    boxes = [player_box_per_frame(tracks, track_id, f, fallback_box) for f in frame_nums]
    feats = [[0.0, 0.0, 0.0, 0.0]]
    for i in range(1, len(boxes)):
        x0, y0, w0, h0 = boxes[i - 1]
        x1, y1, w1, h1 = boxes[i]
        dx = (x1 - x0) / img_w
        dy = (y1 - y0) / img_h
        dw = (w1 - w0) / max(w0, 1)
        dh = (h1 - h0) / max(h0, 1)
        feats.append([dx, dy, dw, dh])
    return feats
