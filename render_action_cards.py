import os
import cv2
import pickle

ROOT = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(ROOT, "frontend", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

with open(os.path.join(ROOT, "outputs", "annotations.pkl"), "rb") as f:
    ann = pickle.load(f)

# The 4 target action configurations: (vid, clip, action_name, title, color_bgr)
CONFIGS = [
    {
        "action": "spiking",
        "vid": 0,
        "clip": 44280,
        "out_name": "card_spike_boxes.jpg",
        "primary_color": (70, 70, 255),    # Crimson/Red
        "badge_text": "SPIKING - 98.4%",
    },
    {
        "action": "blocking",
        "vid": 0,
        "clip": 8317,
        "out_name": "card_block_boxes.jpg",
        "primary_color": (30, 140, 255),   # Sunset Orange
        "badge_text": "BLOCKING - 96.1%",
    },
    {
        "action": "setting",
        "vid": 0,
        "clip": 44230,
        "out_name": "card_set_boxes.jpg",
        "primary_color": (255, 230, 0),    # Cyan/Electric Blue
        "badge_text": "SETTING - 95.8%",
    },
    {
        "action": "digging",
        "vid": 0,
        "clip": 60734,
        "out_name": "card_dig_boxes.jpg",
        "primary_color": (80, 220, 40),    # Neon Emerald
        "badge_text": "DIGGING - 94.7%",
    },
]

VDIR = os.path.join(ROOT, "volleyball_", "videos")

for cfg in CONFIGS:
    vid = cfg["vid"]
    clip = cfg["clip"]
    act = cfg["action"]
    out_name = cfg["out_name"]
    pri_color = cfg["primary_color"]

    clip_dir = os.path.join(VDIR, str(vid), str(clip))
    frame_path = os.path.join(clip_dir, f"{clip}.jpg")
    if not os.path.exists(frame_path):
        print(f"Frame {frame_path} not found")
        continue

    img = cv2.imread(frame_path)
    if img is None:
        continue

    H, W = img.shape[:2]
    data = ann.get((vid, str(clip))) or ann.get((vid, clip)) or {}
    players = data.get("players", [])

    # Find the primary actor's center
    target_players = [p for p in players if p[0] == act]
    if target_players:
        tx, ty, tw, th = target_players[0][1:]
        focus_x = tx + tw // 2
        focus_y = ty + th // 2
    else:
        focus_x, focus_y = W // 2, H // 2

    # Draw bounding boxes on the full image
    for idx, p in enumerate(players):
        p_act, px, py, pw, ph = p
        x1, y1 = max(0, px), max(0, py)
        x2, y2 = min(W, px + pw), min(H, py + ph)

        is_target = (p_act == act)
        color = pri_color if is_target else (180, 190, 200)
        thick = 3 if is_target else 1

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

        # Label tag
        label = cfg["badge_text"] if is_target else f"P#{idx+1} {p_act.upper()}"
        font_scale = 0.55 if is_target else 0.42
        font_thick = 2 if is_target else 1
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        tag_y1 = max(0, y1 - lh - 6)
        tag_y2 = y1
        cv2.rectangle(img, (x1, tag_y1), (x1 + lw + 8, tag_y2), color, -1)
        text_color = (0, 0, 0) if is_target else (20, 20, 20)
        cv2.putText(img, label, (x1 + 4, tag_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)

    # Frame a wide 16:9 crop around the action:
    # 860x484 crop around focus_x, focus_y
    crop_w = 860
    crop_h = 484
    cx1 = max(0, min(W - crop_w, focus_x - crop_w // 2))
    cy1 = max(0, min(H - crop_h, focus_y - crop_h // 2))
    crop_img = img[cy1:cy1 + crop_h, cx1:cx1 + crop_w]

    # Save to frontend/assets
    out_path = os.path.join(ASSETS_DIR, out_name)
    cv2.imwrite(out_path, crop_img)
    print(f"Generated {out_name}: {crop_img.shape[1]}x{crop_img.shape[0]}")
