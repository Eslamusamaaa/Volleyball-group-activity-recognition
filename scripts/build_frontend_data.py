"""Build frontend/data/research.json from the real experiment outputs.

Every number shown in the research pages comes from this file, which is parsed
from outputs/results/*.json and *_test_report.txt (plus the YOLO results.csv and
tracker_validation.json). Re-run after retraining:

    python scripts/build_frontend_data.py
"""
import csv
import json
import os
import re
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "outputs", "results")
OUT_DIR = os.path.join(ROOT, "frontend", "data")
IMG_DIR = os.path.join(ROOT, "frontend", "assets", "research")

# id, label, json stem, report stem, accuracy key, paper target (README ladder), short description
LADDER = [
    ("B1", "Whole image", "scene_finetune_b1", "scene_finetune_b1", "test_acc", 66.7,
     "Center frame to fine-tuned ResNet50. No players, no time."),
    ("B2", "Frozen person crops", "person_pooled_b2", "person_pooled_b2", "test_acc", 64.6,
     "Frozen ImageNet ResNet50 on player crops, max-pooled."),
    ("B3", "Fine-tuned person crops", "person_pooled_b3", "person_pooled_b3", "test_acc", 68.1,
     "B3-A action backbone, frozen, player features max-pooled."),
    ("B4", "Image LSTM", "b4", "b4", "test_scene_acc", 63.1,
     "Whole-frame features over 9 frames into one LSTM."),
    ("B5", "Per-player LSTM", "b5_tracked", "b5_tracked", "test_scene_acc", 67.6,
     "LSTM per player over 9 frames, then max-pool. Real tracking."),
    ("B6", "Pool, then LSTM", "b6_extended3", "b6_extended3", "test_scene_acc", 74.7,
     "Pool players per frame, then a single LSTM."),
    ("B7", "Hierarchical", "b7", "b7", "test_scene_acc", 80.2,
     "Player LSTM1, pool all 12 players, scene LSTM2."),
    ("B8", "Team-separated pooling", "b8", "b8", "test_scene_acc", 81.9,
     "B7 with each team pooled separately and concatenated."),
    ("B8-mt", "B8 + action head", "b8mt_lam1.0", "b8mt_lam1.0", "test_scene_acc", None,
     "B8 plus a per-player 9-way action head. Final model."),
]


def parse_report(path):
    """Return list of sections: {accuracy, classes:[{name,p,r,f1,support}], labels, matrix}."""
    text = open(path, encoding="utf-8").read()
    sections = []
    for block in re.split(r"\n(?=accuracy:)", text):
        m = re.match(r"accuracy:\s*([\d.]+)", block.strip())
        if not m:
            continue
        sec = {"accuracy": float(m.group(1)), "classes": [], "labels": [], "matrix": []}
        lines = block.splitlines()
        i = next(k for k, l in enumerate(lines) if l.strip().startswith("class"))
        i += 1
        while i < len(lines) and lines[i].strip():
            parts = lines[i].split()
            sec["classes"].append({"name": parts[0], "precision": float(parts[1]),
                                   "recall": float(parts[2]), "f1": float(parts[3]),
                                   "support": int(parts[4])})
            i += 1
        j = next(k for k, l in enumerate(lines) if l.startswith("confusion matrix"))
        for l in lines[j + 2:]:
            if not l.strip():
                break
            sec["matrix"].append([int(x) for x in l.split()[1:]])
        sec["labels"] = [c["name"] for c in sec["classes"]]
        sections.append(sec)
    return sections


def load(stem):
    return json.load(open(os.path.join(RES, stem + ".json")))


def main():
    data = {"ladder": [], "models": {}}
    for mid, label, stem, rep, key, paper, desc in LADDER:
        j = load(stem)
        secs = parse_report(os.path.join(RES, rep + "_test_report.txt"))
        entry = {
            "id": mid, "label": label, "description": desc, "paper_target": paper,
            "test_acc": round(j[key] * 100, 2),
            "val_acc": round(j["best_val_acc"] * 100, 2),
            "epochs": j.get("epochs"),
            "curve": [{"epoch": r["epoch"], "train_acc": r["train_acc"], "val_acc": r["val_acc"],
                       "train_loss": r["train_loss"], "val_loss": r["val_loss"]} for r in j["log"]],
            "scene": secs[0],
        }
        if "test_person_acc" in j:
            entry["person_acc"] = round(j["test_person_acc"] * 100, 2)
            entry["person"] = secs[1]
        data["ladder"].append(entry)

    # B3-A person-action backbone
    b3a = load("person_finetune_b3a")
    data["b3a"] = {"test_acc": round(b3a["test_acc"] * 100, 2),
                   "curve": [{"epoch": r["epoch"], "train_acc": r["train_acc"], "val_acc": r["val_acc"]} for r in b3a["log"]],
                   "person": parse_report(os.path.join(RES, "person_finetune_b3a_test_report.txt"))[0]}

    # Follow-up experiments
    fu = []
    for stem, label in [("b8mt_lam1.0", "lambda = 1.0 (final)"), ("b8mt_lam2.0", "lambda = 2.0"),
                        ("b8mt_motion", "+ motion features")]:
        j = load(stem)
        rep = parse_report(os.path.join(RES, stem + "_test_report.txt"))
        fu.append({"name": label, "scene_acc": round(j["test_scene_acc"] * 100, 2),
                   "person_acc": round(j["test_person_acc"] * 100, 2),
                   "recall": {c["name"]: c["recall"] for c in rep[1]["classes"]}})
    data["followups"] = fu

    # Detector (best epoch by mAP50) and tracker
    rows = list(csv.DictReader(open(os.path.join(ROOT, "outputs", "checkpoints",
                                                 "yolov8n_player", "results.csv"))))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    data["detector"] = {
        "model": "YOLOv8n (fine-tuned, single class: player)", "best_epoch": int(best["epoch"]),
        "precision": round(float(best["metrics/precision(B)"]) * 100, 1),
        "recall": round(float(best["metrics/recall(B)"]) * 100, 1),
        "map50": round(float(best["metrics/mAP50(B)"]) * 100, 1),
        "map50_95": round(float(best["metrics/mAP50-95(B)"]) * 100, 1),
        "curve": [{"epoch": int(r["epoch"]), "map50": float(r["metrics/mAP50(B)"])} for r in rows],
    }
    tv = load("tracker_validation")
    data["tracker"] = {
        "match_rate": round(tv["average_match_rate"] * 100, 1), "n_clips": tv["n_clips"],
        "per_clip": [{"clip": c["clip"], "video": c["video"], "matched": c["matched"], "total": c["total"]}
                     for c in tv["per_clip"]],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "research.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    # Real project figures -> frontend assets (optimised JPEG/WebP when PIL is available)
    os.makedirs(IMG_DIR, exist_ok=True)
    figs = ["b8mt_pipeline", "b8_pipeline", "b7_pipeline", "b5_pipeline", "b4_pipeline", "b3_pipeline",
            "b2_pipeline", "b1_pipeline", "b6_pipeline", "explainer_static_vs_tracked_box",
            "explainer_9_of_41_frames"]
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for name in figs:
        src = os.path.join(RES, name + ".png")
        if not os.path.exists(src):
            continue
        if Image:
            im = Image.open(src).convert("RGB")
            im.save(os.path.join(IMG_DIR, name + ".webp"), "WEBP", lossless=True, method=6)
        else:
            shutil.copy(src, os.path.join(IMG_DIR, name + ".png"))
    print("wrote research.json and", len(os.listdir(IMG_DIR)), "figures")


if __name__ == "__main__":
    main()
