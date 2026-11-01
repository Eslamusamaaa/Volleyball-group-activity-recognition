"""Section 10, stage 1: automatic player (person) detection on raw video
frames -- a genuinely new video has no ground-truth boxes, unlike the
training clips, so this replaces the annotation file's boxes at inference
time.

COCO-pretrained YOLOv8n, no volleyball-specific fine-tuning (project brief
Section 8/10: try the off-the-shelf detector first, only fine-tune if
quality turns out too poor -- not assumed up front).
"""
import os
from ultralytics import YOLO

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "checkpoints", "yolov8n.pt")
PERSON_CLASS_ID = 0  # COCO class 0 = "person"


class PlayerDetector:
    def __init__(self, weights=WEIGHTS, conf=0.35, device=None):
        """weights: path to a YOLO .pt file; downloaded automatically by
        ultralytics on first use if the path doesn't exist yet.
        conf: minimum detection confidence to keep a box."""
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def detect(self, frame_bgr):
        """One frame (H, W, 3) BGR, as read by cv2. Returns a list of
        {"bbox": [x1, y1, x2, y2], "conf": float}, person-class only,
        sorted by confidence descending."""
        results = self.model.predict(
            frame_bgr, classes=[PERSON_CLASS_ID], conf=self.conf,
            device=self.device, verbose=False,
        )
        boxes = []
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append({"bbox": [x1, y1, x2, y2], "conf": float(b.conf[0])})
        boxes.sort(key=lambda d: -d["conf"])
        return boxes


if __name__ == "__main__":
    import sys
    from .video_io import frame_source

    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\volleyball datasets\volleyball_\videos\4\94540"
    det = PlayerDetector()
    for idx, frame in frame_source(path):
        boxes = det.detect(frame)
        print(f"frame {idx}: {len(boxes)} players, top conf {boxes[0]['conf']:.2f}" if boxes
              else f"frame {idx}: 0 players")
        if idx >= 8:
            break
