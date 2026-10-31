"""Load Volleyball dataset annotations (scene + per-player) and official splits.

Single source of truth for labels: reuses the parsing logic already verified in
explore_data.py (on-disk field order is "X Y W H action", not "action X Y W H").
"""
import os
import pickle
from collections import Counter

TRAIN = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
VAL = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
SPLIT = {v: s for s, vids in (("train", TRAIN), ("val", VAL), ("test", TEST)) for v in vids}

SCENE_CLASSES = sorted(["l-pass", "r-pass", "r_set", "l-spike", "l_set", "r_spike", "l_winpoint", "r_winpoint"])
SCENE_TO_IDX = {c: i for i, c in enumerate(SCENE_CLASSES)}

ACTION_CLASSES = sorted(["standing", "moving", "waiting", "blocking", "digging", "setting", "falling", "spiking", "jumping"])
ACTION_TO_IDX = {c: i for i, c in enumerate(ACTION_CLASSES)}

EXPECTED_SCENE_COUNTS = {
    "l-pass": 826, "r-pass": 801, "r_set": 644, "l-spike": 642,
    "l_set": 632, "r_spike": 623, "l_winpoint": 367, "r_winpoint": 295,
}
EXPECTED_ACTION_COUNTS = {
    "standing": 38686, "moving": 5120, "waiting": 3600, "blocking": 2458,
    "digging": 2333, "setting": 1332, "falling": 1241, "spiking": 1216, "jumping": 341,
}
EXPECTED_TOTAL_CLIPS = 4830


def load_annotations(root):
    """Return {(video, clip): {'category': str, 'players': [(action, x, y, w, h), ...]}}."""
    data = {}
    for vid in sorted(int(d) for d in os.listdir(root) if d.isdigit()):
        ann_path = os.path.join(root, str(vid), "annotations.txt")
        with open(ann_path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                clip = parts[0].split(".")[0]
                players = []
                for i in range(2, len(parts), 5):
                    x, y, w, h = map(int, parts[i:i + 4])
                    action = parts[i + 4]
                    players.append((action, x, y, w, h))
                data[(vid, clip)] = {"category": parts[1], "players": players}
    return data


def load_or_cache(root, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    data = load_annotations(root)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data


def verify_counts(data):
    """Check parsed annotations match the documented dataset statistics (Section 3.6).

    Returns (ok, report_lines).
    """
    lines = []
    ok = True

    if len(data) != EXPECTED_TOTAL_CLIPS:
        ok = False
    lines.append(f"Total clips: {len(data)} (expected {EXPECTED_TOTAL_CLIPS})")

    cats = Counter(d["category"] for d in data.values())
    for cls, expected in EXPECTED_SCENE_COUNTS.items():
        got = cats.get(cls, 0)
        if got != expected:
            ok = False
        lines.append(f"  scene {cls:12s} {got:5d} (expected {expected})")

    actions = Counter(p[0] for d in data.values() for p in d["players"])
    for cls, expected in EXPECTED_ACTION_COUNTS.items():
        got = actions.get(cls, 0)
        if got != expected:
            ok = False
        lines.append(f"  action {cls:10s} {got:6d} (expected {expected})")

    return ok, lines


def split_keys(data, split_name):
    """Return sorted list of (video, clip) keys belonging to 'train'/'val'/'test'."""
    return sorted(k for k in data if SPLIT[k[0]] == split_name)


if __name__ == "__main__":
    root = r"E:\volleyball datasets\volleyball_\videos"
    cache = r"E:\volleyball datasets\outputs\annotations.pkl"
    data = load_or_cache(root, cache)
    ok, report = verify_counts(data)
    print("\n".join(report))
    print("\nMATCH" if ok else "\nMISMATCH — counts differ from Section 3.6, investigate before proceeding")
