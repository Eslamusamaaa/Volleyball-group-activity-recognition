"""Fine-tune the Section 10 person detector on this project's own
ground-truth player boxes (prepare_yolo_dataset.py builds the data first),
so it learns "volleyball player" specifically instead of generic COCO
"person" -- which also fires on courtside referees/coaches/ball kids.

4 GB VRAM target (RTX 3050 laptop, same constraint as every other baseline
in this project) -- YOLOv8n is small enough that this fits comfortably.
"""
import os
from ultralytics import YOLO

ROOT = r"E:\volleyball datasets"
DATA_YAML = os.path.join(ROOT, "outputs", "yolo_player_data", "data.yaml")
BASE_WEIGHTS = os.path.join(ROOT, "outputs", "checkpoints", "yolov8n.pt")
RUN_NAME = "yolov8n_player"


def main():
    model = YOLO(BASE_WEIGHTS)
    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        project=os.path.join(ROOT, "outputs", "checkpoints"),
        name=RUN_NAME,
        patience=8,
        amp=True,
        workers=2,  # this machine has 16GB RAM; 8 workers (ultralytics default)
                    # crashed mid-validation with a Windows cv2 OutOfMemoryError
        cache=False,
    )


if __name__ == "__main__":
    main()
