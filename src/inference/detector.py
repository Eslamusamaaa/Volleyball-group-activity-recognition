"""Section 10, stage 1: automatic player detection on raw video frames --
a genuinely new video has no ground-truth boxes, unlike the training
clips, so this replaces the annotation file's boxes at inference time.

First pass was the stock COCO-pretrained YOLOv8n (person class), per the
project brief's "try off-the-shelf first" instruction. Measured quality
empirically (see outputs/results/tracker_validation.json): the generic
"person" class also fires on courtside referees/coaches/ball kids, and
the plain detector itself (before any tracking) only matched 77.8% of
ground-truth players on an 18-clip sweep across 6 videos. Fine-tuned
YOLOv8n on this project's own ~2,150 train-split clips' ground-truth
player boxes instead (single class: "player", see prepare_yolo_dataset.py
+ train_detector.py) -- 95.3% precision / 94.2% recall / 0.98 mAP50 on
the held-out val-split images. That fine-tuned checkpoint is now the
default here.
"""
import os
from ultralytics import YOLO

FINETUNED_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "checkpoints",
                                  "yolov8n_player", "weights", "best.pt")
COCO_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "checkpoints", "yolov8n.pt")
WEIGHTS = FINETUNED_WEIGHTS if os.path.exists(FINETUNED_WEIGHTS) else COCO_WEIGHTS
PERSON_CLASS_ID = 0  # class 0 in both: COCO "person", and the fine-tuned model's only class "player"


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
