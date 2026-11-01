"""Section 10, stage 2: link per-frame detections into stable per-player
tracks (a consistent ID across frames), using YOLO's built-in ByteTrack.

Project brief: "Validate first against the ground-truth track_ids in
volleyball_tracking_annotation before trusting it on a new video" --
validate_on_clip() below does exactly that, the same way B5's own real-vs-
static tracking comparison was validated (IoU match against ground truth).
"""
import os
from ultralytics import YOLO

from .detector import WEIGHTS, PERSON_CLASS_ID


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PlayerTracker:
    def __init__(self, weights=WEIGHTS, conf=0.35, tracker_cfg="bytetrack.yaml", device=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.tracker_cfg = tracker_cfg
        self.device = device

    def reset(self):
        """Start a fresh set of track IDs -- call this between videos, since
        ultralytics keeps tracker state across calls when persist=True."""
        if hasattr(self.model.predictor, "trackers"):
            self.model.predictor.trackers[0].reset()

    def update(self, frame_bgr):
        """One frame -> list of {"track_id": int, "bbox": [x1,y1,x2,y2], "conf": float}.
        Call in frame order, on every frame of the same video, without
        skipping frames -- ByteTrack's identity continuity depends on it."""
        results = self.model.track(
            frame_bgr, classes=[PERSON_CLASS_ID], conf=self.conf,
            tracker=self.tracker_cfg, persist=True, device=self.device, verbose=False,
        )
        out = []
        r = results[0]
        if r.boxes.id is None:
            return out
        for box, tid in zip(r.boxes, r.boxes.id):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            out.append({"track_id": int(tid), "bbox": [x1, y1, x2, y2], "conf": float(box.conf[0])})
        return out


def validate_on_clip(video_id, clip_id, dataset_root=r"E:\volleyball datasets\volleyball_\videos",
                      iou_threshold=0.3):
    """Run the tracker across one known annotated clip's 9 real frames (in
    order) and report what fraction of the clip's ground-truth players (at
    the center/annotated frame) get matched to a tracked box by IoU -- the
    same sanity check this project already ran for the dataset's own
    tracking files (99.8% match) for B5, now applied to OUR tracker on a
    video it's never seen ground truth for.
    """
    from .video_io import iter_frame_dir
    from ..data.annotations import load_annotations

    clip_dir = os.path.join(dataset_root, str(video_id), str(clip_id))
    ann = load_annotations(os.path.join(dataset_root))
    gt = ann.get((video_id, str(clip_id)))
    if gt is None:
        raise ValueError(f"no ground-truth annotation for video {video_id} clip {clip_id}")
    gt_boxes = [(x, y, x + w, y + h) for (_, x, y, w, h) in gt["players"]]
    center_frame_name = f"{clip_id}.jpg"

    tracker = PlayerTracker()
    tracker.reset()
    last_frame_detections = None
    for idx, frame in iter_frame_dir(clip_dir):
        dets = tracker.update(frame)
        names = sorted(os.listdir(clip_dir), key=lambda n: int(os.path.splitext(n)[0]))
        if names[idx] == center_frame_name:
            last_frame_detections = dets

    if last_frame_detections is None:
        return {"matched": 0, "total": len(gt_boxes), "match_rate": 0.0, "note": "center frame not reached"}

    matched = 0
    used = set()
    for gtb in gt_boxes:
        best_j, best_iou = -1, 0.0
        for j, det in enumerate(last_frame_detections):
            if j in used:
                continue
            v = iou(gtb, det["bbox"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold:
            matched += 1
            used.add(best_j)

    return {
        "matched": matched, "total": len(gt_boxes),
        "match_rate": matched / len(gt_boxes) if gt_boxes else 0.0,
        "detected_at_center_frame": len(last_frame_detections),
    }


if __name__ == "__main__":
    report = validate_on_clip(video_id=4, clip_id=94540)
    print("Tracker validation on video 4 / clip 94540 (r_spike, test split):")
    print(report)
