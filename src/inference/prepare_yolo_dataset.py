"""Export this project's own ground-truth player boxes (center frame of
every clip) into YOLO training format, to fine-tune the Section 10 person
detector specifically on volleyball players -- instead of a generic COCO
"person" class that also fires on courtside referees/coaches/ball kids.

Single class: "player". Train/val images come only from the official
TRAIN/VAL video splits (never TEST), same convention as every other
baseline in this project.
"""
import os
import shutil
from PIL import Image

from ..data.annotations import load_annotations, TRAIN, VAL

ROOT = r"E:\volleyball datasets"
VIDEOS_DIR = os.path.join(ROOT, "volleyball_", "videos")
OUT = os.path.join(ROOT, "outputs", "yolo_player_data")


def export_split(data, video_ids, split_name):
    img_dir = os.path.join(OUT, "images", split_name)
    lbl_dir = os.path.join(OUT, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n = 0
    for (vid, clip), ann in data.items():
        if vid not in video_ids:
            continue
        src_img = os.path.join(VIDEOS_DIR, str(vid), clip, f"{clip}.jpg")
        if not os.path.exists(src_img):
            continue
        with Image.open(src_img) as im:
            W, H = im.size

        name = f"{vid}_{clip}"
        dst_img = os.path.join(img_dir, f"{name}.jpg")
        if not os.path.exists(dst_img):
            shutil.copyfile(src_img, dst_img)

        lines = []
        for (_, x, y, w, h) in ann["players"]:
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines))
        n += 1
    print(f"{split_name}: {n} images exported to {img_dir}")
    return n


def main():
    cache = os.path.join(ROOT, "outputs", "annotations.pkl")
    data = load_annotations(VIDEOS_DIR) if not os.path.exists(cache) else __import__("pickle").load(open(cache, "rb"))

    export_split(data, set(TRAIN), "train")
    export_split(data, set(VAL), "val")

    yaml_path = os.path.join(OUT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {OUT}
train: images/train
val: images/val
names:
  0: player
""")
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
