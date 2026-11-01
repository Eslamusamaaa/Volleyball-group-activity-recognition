"""Regenerate the baseline pipeline diagrams (outputs/results/b*_pipeline.png)
using REAL frames/crops from one example clip instead of abstract icons.

Supersedes the icon-only diagrams in make_pipeline_diagrams.py (kept for the
results_panel() confusion-matrix/accuracy panels, which this script doesn't
touch). Run once; everything needed is re-extracted from the dataset on disk,
so it's reproducible without any external/scratch state.

Example clip: video 4, clip 94540 (test split) -- Brazil vs France, Rio 2015
World League Finals, ground-truth label r_spike. Chosen for a clean spike +
two blockers + a mover, giving every action-highlight baseline something real
to show.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = r"E:\volleyball datasets"
VID, CLIP, CENTER = 4, "94540", 94540
FRAME_DIR = os.path.join(ROOT, "volleyball_", "videos", str(VID), CLIP)
TRACK_FILE = os.path.join(ROOT, "volleyball_tracking_annotation", "volleyball_tracking_annotation",
                           str(VID), CLIP, f"{CLIP}.txt")
ANN_FILE = os.path.join(ROOT, "volleyball_", "videos", str(VID), "annotations.txt")
OUT = os.path.join(ROOT, "outputs", "results")

# ---------- design tokens (shared with make_pipeline_diagrams.py) ----------
BG = (15, 17, 23)
PANEL = (19, 23, 33)
LINE = (38, 44, 61)
LINE_SOFT = (28, 33, 46)
TEXT = (230, 232, 239)
SUBTEXT = (154, 162, 177)
GOOD = (54, 179, 126)
BAD = (255, 92, 122)
TEAM = {"L": (255, 140, 66), "R": (60, 170, 255)}
ACCENT = {
    "b1": (79, 140, 255), "b2": (255, 138, 79), "b3": (54, 179, 126),
    "b4": (166, 107, 255), "b5": (79, 209, 255), "b6": (255, 92, 122),
    "b7": (240, 180, 41), "b8": (72, 209, 204), "b8mt": (255, 122, 182),
}

F_TITLE = ImageFont.truetype("arialbd.ttf", 30)
F_BADGE = ImageFont.truetype("arialbd.ttf", 34)
F_PAPER = ImageFont.truetype("consola.ttf", 15)
F_LABEL = ImageFont.truetype("arialbd.ttf", 18)
F_SUB = ImageFont.truetype("arial.ttf", 15)
F_CAP = ImageFont.truetype("consola.ttf", 13)
F_NUM = ImageFont.truetype("arialbd.ttf", 15)
F_BIG = ImageFont.truetype("arialbd.ttf", 40)
F_DELTA = ImageFont.truetype("consola.ttf", 16)


def rrect(draw, box, radius, outline, width=2, fill=None):
    draw.rounded_rectangle(box, radius=radius, outline=outline, width=width, fill=fill)


def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ============================================================
# 1. Pull real assets for the example clip
# ============================================================
def extract_assets():
    players = []
    category = None
    with open(ANN_FILE) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].split(".")[0] != CLIP:
                continue
            category = parts[1]
            for i in range(2, len(parts), 5):
                x, y, w, h = map(int, parts[i:i + 4])
                players.append({"x": x, "y": y, "w": w, "h": h, "action": parts[i + 4]})
            break
    players.sort(key=lambda p: p["x"] + p["w"] / 2)
    for i, p in enumerate(players):
        p["team"] = "L" if i < 6 else "R"

    tracks = {}
    with open(TRACK_FILE) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tid = int(parts[0])
            x1, y1, x2, y2 = map(int, parts[1:5])
            frame = int(parts[5])
            label = parts[9] if len(parts) > 9 else ""
            tracks.setdefault(tid, {})[frame] = (x1, y1, x2, y2, label)

    spiker_gt = next(p for p in players if p["action"] == "spiking")
    scx, scy = spiker_gt["x"] + spiker_gt["w"] / 2, spiker_gt["y"] + spiker_gt["h"] / 2
    best_tid, best_d = None, 1e9
    for tid, fr in tracks.items():
        if CENTER not in fr:
            continue
        x1, y1, x2, y2, _ = fr[CENTER]
        d = ((x1 + x2) / 2 - scx) ** 2 + ((y1 + y2) / 2 - scy) ** 2
        if d < best_d:
            best_d, best_tid = d, tid

    frames = list(range(CENTER - 4, CENTER + 5))
    center_im = Image.open(os.path.join(FRAME_DIR, f"{CENTER}.jpg")).convert("RGB")
    return {
        "category": category, "players": players, "tracks": tracks,
        "spiker_tid": best_tid, "spiker_gt": spiker_gt, "frames": frames,
        "center_im": center_im,
    }


A = extract_assets()


def frame_im(fr):
    return Image.open(os.path.join(FRAME_DIR, f"{fr}.jpg")).convert("RGB")


def fit(im, box_w, box_h, cover=False):
    """Resize into (box_w, box_h). contain (default): letterbox, nothing cropped.
    cover: scale to fill the box and center-crop the overflow -- for portrait
    player crops, where letterboxing would leave large black bars."""
    if cover:
        r = max(box_w / im.width, box_h / im.height)
        nw, nh = max(1, int(im.width * r)), max(1, int(im.height * r))
        im2 = im.resize((nw, nh))
        left, top = (nw - box_w) // 2, (nh - box_h) // 2
        return im2.crop((left, top, left + box_w, top + box_h))
    r = min(box_w / im.width, box_h / im.height)
    nw, nh = max(1, int(im.width * r)), max(1, int(im.height * r))
    im2 = im.resize((nw, nh))
    canvas = Image.new("RGB", (box_w, box_h), (0, 0, 0))
    canvas.paste(im2, ((box_w - nw) // 2, (box_h - nh) // 2))
    return canvas


def img_annotated_center(all_boxes=True):
    im = A["center_im"].copy()
    draw = ImageDraw.Draw(im)
    highlight = {"spiking", "blocking", "setting", "digging", "jumping", "falling"}
    order = [p for p in A["players"] if p["action"] in highlight]
    order.sort(key=lambda p: p["x"])
    slot = {id(p): i for i, p in enumerate(order)}
    for p in A["players"]:
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        color = TEAM[p["team"]]
        lw = 5 if p["action"] in highlight else 2
        draw.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        if p["action"] in highlight:
            label = p["action"]
            try:
                font_px = ImageFont.truetype("arial.ttf", 20)
            except Exception:
                font_px = ImageFont.load_default()
            tw = draw.textlength(label, font=font_px)
            above = slot[id(p)] % 2 == 0
            ly = (y - 26) if above else (y + h + 3)
            draw.rectangle([x, ly, x + tw + 10, ly + 24], fill=color)
            draw.text((x + 5, ly + 2), label, fill=(10, 10, 15), font=font_px)
    return im


def img_player_crop(p, pad=6):
    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
    box = (max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad)
    return A["center_im"].crop(box)


def img_filmstrip():
    thumbs = []
    for fr in A["frames"]:
        im = frame_im(fr)
        im.thumbnail((200, 113))
        thumbs.append((fr, im))
    w, h = thumbs[0][1].size
    gap = 4
    strip = Image.new("RGB", (w * len(thumbs) + gap * (len(thumbs) - 1), h), BG)
    d = ImageDraw.Draw(strip)
    for i, (fr, im) in enumerate(thumbs):
        x = i * (w + gap)
        strip.paste(im, (x, 0))
        tag = "t=0" if fr == CENTER else f"{fr - CENTER:+d}"
        d.rectangle([x, 0, x + 34, 15], fill=(0, 0, 0))
        d.text((x + 3, 1), tag, fill=(255, 255, 255), font=F_CAP)
    return strip


def img_spiker_sequence():
    frames_box = A["tracks"].get(A["spiker_tid"], {})
    crops = []
    for fr in A["frames"]:
        if fr not in frames_box:
            continue
        x1, y1, x2, y2, _ = frames_box[fr]
        im = frame_im(fr)
        pad = 8
        box = (max(0, x1 - pad), max(0, y1 - pad), min(im.width, x2 + pad), min(im.height, y2 + pad))
        crops.append((fr, im.crop(box)))
    th = 150
    resized = []
    for fr, im in crops:
        r = th / im.height
        resized.append((fr, im.resize((max(1, int(im.width * r)), th))))
    gap = 4
    total_w = sum(im.width for _, im in resized) + gap * (len(resized) - 1)
    strip = Image.new("RGB", (total_w, th + 16), BG)
    d = ImageDraw.Draw(strip)
    x = 0
    for fr, im in resized:
        strip.paste(im, (x, 16))
        tag = "t=0" if fr == CENTER else f"{fr - CENTER:+d}"
        d.text((x + 2, 1), tag, fill=TEXT, font=F_CAP)
        x += im.width + gap
    return strip


def img_team_split():
    def block(team):
        ps = [p for p in A["players"] if p["team"] == team]
        th = 150
        cell = 92
        gap = 5
        blk = Image.new("RGB", (cell * len(ps) + gap * (len(ps) - 1), th + 20), BG)
        bd = ImageDraw.Draw(blk)
        for i, p in enumerate(ps):
            im = img_player_crop(p, pad=4)
            r = th / im.height
            im = im.resize((min(cell, int(im.width * r)), th))
            x = i * (cell + gap)
            blk.paste(im, (x + (cell - im.width) // 2, 0))
            bd.text((x, th + 2), p["action"][:9], fill=TEAM[team], font=F_CAP)
        return blk

    left, right = block("L"), block("R")
    net_w = 22
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + net_w + right.width, h), BG)
    out.paste(left, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([left.width, 0, left.width + net_w, h], fill=LINE)
    d.line([left.width + net_w // 2, 0, left.width + net_w // 2, h], fill=TEXT, width=2)
    out.paste(right, (left.width + net_w, 0))
    return out


def img_all_players_grid():
    th, cell, cols, pad, gap = 120, 96, 6, 6, 22
    rows = 2
    w = cols * cell + (cols - 1) * pad
    h = rows * (th + gap) + rows * 4
    grid = Image.new("RGB", (w, h), BG)
    gd = ImageDraw.Draw(grid)
    for i, p in enumerate(A["players"]):
        im = img_player_crop(p, pad=4)
        r = th / im.height
        im = im.resize((min(cell, int(im.width * r)), th))
        col, row = i % cols, i // cols
        cx = col * (cell + pad)
        cy = row * (th + gap + 4)
        gd.rectangle([cx - 2, cy - 2, cx + cell + 2, cy + th + 2], outline=TEAM[p["team"]], width=3)
        grid.paste(im, (cx + (cell - im.width) // 2, cy))
        gd.text((cx, cy + th + 3), p["action"][:9], fill=TEAM[p["team"]], font=F_CAP)
    return grid


def img_static_vs_tracked():
    first_fr = A["frames"][0]
    im = frame_im(first_fr)
    d = ImageDraw.Draw(im)
    gx, gy, gw, gh = (A["spiker_gt"][k] for k in ("x", "y", "w", "h"))
    d.rectangle([gx, gy, gx + gw, gy + gh], outline=BAD, width=5)
    d.text((gx, gy - 24), "static: reused center-frame box", fill=BAD, font=F_SUB)
    tbox = A["tracks"].get(A["spiker_tid"], {}).get(first_fr)
    if tbox:
        tx1, ty1, tx2, ty2, _ = tbox
        d.rectangle([tx1, ty1, tx2, ty2], outline=GOOD, width=5)
        d.text((tx1, ty2 + 6), "tracked: actual box at t-4", fill=GOOD, font=F_SUB)
    return im


# pre-build the handful of real-image assets every baseline draws from
ASSETS = {
    "center_annotated": img_annotated_center(),
    "spiker_single": img_player_crop(next(p for p in A["players"] if p["action"] == "spiking")),
    "player_grid": img_all_players_grid(),
    "filmstrip": img_filmstrip(),
    "spiker_seq": img_spiker_sequence(),
    "team_split": img_team_split(),
    "static_vs_tracked": img_static_vs_tracked(),
}


# ============================================================
# 2. Composition primitives
# ============================================================
STAGE_W_DEFAULT = 236
STAGE_H = 300
PHOTO_H = 140
MARGIN = 36
GAP = 46
TOP = 118
BOTTOM_PAD = 34


def draw_arrow(draw, x0, y0, x1, y1, color, width=4):
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    ah = 9
    draw.polygon([(x1, y0), (x1 - ah, y0 - ah * 0.6), (x1 - ah, y0 + ah * 0.6)], fill=color)


def draw_stage(draw, canvas, x, y, w, h, accent, num, label, sub_lines, image=None, caption=None, photo_h=PHOTO_H,
               cover=False):
    rrect(draw, [x, y, x + w, y + h], 18, outline=accent, width=2, fill=PANEL)
    # numbered chip
    chip_r = 16
    cx, cy = x + 24, y + 24
    draw.ellipse([cx - chip_r, cy - chip_r, cx + chip_r, cy + chip_r], fill=accent)
    tw = draw.textlength(str(num), font=F_NUM)
    draw.text((cx - tw / 2, cy - 9), str(num), fill=BG, font=F_NUM)
    cursor_y = y + 52
    if image is not None:
        iw = w - 24
        fitted = fit(image, iw, photo_h, cover=cover)
        canvas.paste(fitted, (x + 12, cursor_y))
        draw.rectangle([x + 12, cursor_y, x + 12 + iw, cursor_y + photo_h], outline=LINE_SOFT, width=1)
        cursor_y += photo_h + 6
        if caption:
            draw.text((x + 14, cursor_y), caption, fill=SUBTEXT, font=F_CAP)
            cursor_y += 20
    cursor_y += 6
    for line in wrap_text(draw, label, F_LABEL, w - 24):
        draw.text((x + 14, cursor_y), line, fill=TEXT, font=F_LABEL)
        cursor_y += 23
    cursor_y += 4
    for s in sub_lines:
        for line in wrap_text(draw, s, F_SUB, w - 24):
            draw.text((x + 14, cursor_y), line, fill=SUBTEXT, font=F_SUB)
            cursor_y += 19
    return cursor_y


def pipeline_diagram(path, baseline_id, paper_note, title, stages, accent, result_text, delta_text, delta_good,
                      stage_w=None):
    n = len(stages)
    sw = stage_w or STAGE_W_DEFAULT
    widths = [st.get("w", sw) for st in stages]
    width = MARGIN * 2 + sum(widths) + (n - 1) * GAP
    height = TOP + STAGE_H + BOTTOM_PAD
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    # header
    draw.text((MARGIN, 18), baseline_id, fill=accent, font=F_BADGE)
    bw = draw.textlength(baseline_id, font=F_BADGE)
    draw.rounded_rectangle([MARGIN + bw + 16, 24, MARGIN + bw + 16 + draw.textlength(paper_note, font=F_PAPER) + 20, 52],
                            radius=8, outline=LINE, width=1)
    draw.text((MARGIN + bw + 26, 30), paper_note, fill=SUBTEXT, font=F_PAPER)
    draw.text((MARGIN, 62), title, fill=TEXT, font=F_SUB)

    # scoreboard top-right (two lines: result readout, then delta below)
    rw = max(300, int(draw.textlength(result_text, font=F_LABEL)) + 28,
             int(draw.textlength(delta_text, font=F_DELTA)) + 28)
    rx = width - MARGIN - rw
    draw.rounded_rectangle([rx, 10, rx + rw, 66], radius=10, outline=LINE, width=1, fill=PANEL)
    draw.text((rx + 14, 17), result_text, fill=TEXT, font=F_LABEL)
    dcolor = GOOD if delta_good else BAD
    dtw = draw.textlength(delta_text, font=F_DELTA)
    draw.text((rx + rw - dtw - 14, 42), delta_text, fill=dcolor, font=F_DELTA)

    x = MARGIN
    y = TOP
    for i, st in enumerate(stages):
        w = st.get("w", sw)
        draw_stage(draw, canvas, x, y, w, STAGE_H, accent, st["num"], st["label"], st.get("sub", []),
                   image=st.get("image"), caption=st.get("caption"), photo_h=st.get("photo_h", PHOTO_H),
                   cover=st.get("cover", False))
        if i < n - 1:
            nx = x + w + GAP
            draw_arrow(draw, x + w + 6, y + STAGE_H / 2, nx - 6, y + STAGE_H / 2, accent)
        x += w + GAP

    canvas.save(path, quality=92)
    print("saved", path)


# ============================================================
# 3. One call per baseline
# ============================================================

# ---- B1 ----
pipeline_diagram(
    f"{OUT}/b1_pipeline.png", "B1-tuned", "paper: fine-tuned AlexNet -> this repo: ResNet50",
    "Whole center frame -> fine-tuned ResNet50 -> 8-way scene class. No players, no time.",
    [
        {"num": 1, "label": "Center frame", "sub": ["one whole-scene image", "1280x720, raw input"],
         "image": A["center_im"], "caption": "t=0, raw"},
        {"num": 2, "label": "ResNet50", "sub": ["fine-tuned end-to-end", "every weight trainable"]},
        {"num": 3, "label": "Global pool", "sub": ["spatial average", "-> one 2048-d vector"]},
        {"num": 4, "label": "Linear classifier", "sub": ["-> 8 scene classes"]},
    ], ACCENT["b1"], "72.2% test acc.", "+5.5 pts vs. paper 66.7%", True)

# ---- B2 ----
pipeline_diagram(
    f"{OUT}/b2_pipeline.png", "B2", 'paper name: "Person Classification"',
    "Every player crop -> FROZEN ImageNet ResNet50 -> max-pool -> small trained classifier -> scene class.",
    [
        {"num": 1, "label": "All player crops", "sub": ["up to 12 boxes,", "same center frame"],
         "image": ASSETS["player_grid"], "caption": "orange = left team, blue = right", "w": 300, "photo_h": 110},
        {"num": 2, "label": "ResNet50 -- frozen", "sub": ["plain ImageNet weights,", "never fine-tuned"]},
        {"num": 3, "label": "Max-pool players", "sub": ["12 x 2048-d", "-> 1 x 2048-d"]},
        {"num": 4, "label": "Small classifier", "sub": ["(only trained part)", "-> 8 scene classes"]},
    ], ACCENT["b2"], "42.2% test acc.", "-22.4 pts vs. paper 64.6%", False)

# ---- B3 ----
pipeline_diagram(
    f"{OUT}/b3_pipeline.png", "B3", "two steps: A (teach a backbone), then B+C (reuse B2's pipeline)",
    "Step A fine-tunes on person action; steps B+C freeze it and pool across players -> scene class.",
    [
        {"num": "1A", "label": "One player crop", "sub": ["target: 9-way person", "action, not scene"],
         "image": ASSETS["spiker_single"], "caption": 'label: "spiking"', "photo_h": 150, "cover": True},
        {"num": "2A", "label": "ResNet50 fine-tuned", "sub": ["on person-action labels", "(step B3-A): 80.6% 9-way acc."]},
        {"num": "1BC", "label": "Freeze, re-run on all 12", "sub": ["now-fine-tuned backbone,", "every player crop"],
         "image": ASSETS["player_grid"], "caption": "same 12 crops as B2", "w": 300, "photo_h": 100},
        {"num": "2BC", "label": "Max-pool + classify", "sub": ["identical to B2's", "steps 3-4 -> scene class"]},
    ], ACCENT["b3"], "65.7% test acc. (B+C)", "-2.4 pts vs. paper 68.1%", False, stage_w=236)

# ---- B4 ----
pipeline_diagram(
    f"{OUT}/b4_pipeline.png", "B4", "paper's \"Implementation #1\" -- image-level temporal",
    "9-frame window of whole images -> B1's frozen backbone (cached) -> one LSTM -> scene class.",
    [
        {"num": 1, "label": "9-frame window", "sub": ["whole images, centered", "on the annotated frame"],
         "image": ASSETS["filmstrip"], "caption": "t = -4 ... 0 ... +4", "w": 420, "photo_h": 90},
        {"num": 2, "label": "B1 backbone", "sub": ["frozen, features", "cached to disk"]},
        {"num": 3, "label": "Single LSTM", "sub": ["over the 9-step", "sequence of vectors"]},
        {"num": 4, "label": "Linear classifier", "sub": ["on final hidden state", "-> 8 scene classes"]},
    ], ACCENT["b4"], "73.4% test acc.", "+10.3 pts vs. paper 63.1%", True)

# ---- B5 ----
pipeline_diagram(
    f"{OUT}/b5_pipeline.png", "B5", 'paper\'s "Temporal on crops" -- players arrive here',
    "Per-player tracked crop sequence -> B3-A backbone -> LSTM1 per player -> pool -> scene class.",
    [
        {"num": 1, "label": "Per-player tracked crop", "sub": ["this player's own box,", "each of the 9 frames"],
         "image": ASSETS["spiker_seq"], "caption": "the spiker's own box, t = -4...+4", "w": 460, "photo_h": 110},
        {"num": 2, "label": "B3-A backbone", "sub": ["frozen, cached --", "2048-d per player/frame"]},
        {"num": 3, "label": "LSTM1 per player", "sub": ["shared weights,", "-> final hidden state"]},
        {"num": 4, "label": "Max-pool players", "sub": ["across all 12", "final hidden states"]},
        {"num": 5, "label": "Linear classifier", "sub": ["-> 8 scene classes,", "no scene-level LSTM yet"]},
    ], ACCENT["b5"], "79.0% test acc. (tracked)", "+11.4 pts vs. paper 67.6%", True)

# ---- B6 ----
pipeline_diagram(
    f"{OUT}/b6_pipeline.png", "B6", "same ingredients as B5, pooling and LSTM order reversed",
    "Pool players per frame first -> single LSTM over the pooled 9-step sequence -> scene class.",
    [
        {"num": 1, "label": "Each of 9 frames", "sub": ["all player crops", "in that frame"],
         "image": ASSETS["filmstrip"], "caption": "same 9-frame window as B4", "w": 420, "photo_h": 90},
        {"num": 2, "label": "B3-A backbone", "sub": ["frozen, cached,", "per player per frame"]},
        {"num": 3, "label": "Max-pool per frame", "sub": ["12 players -> 1 vector,", "9x -> 9-step sequence"]},
        {"num": 4, "label": "Single LSTM", "sub": ["over pooled sequence --", "no per-player LSTM1"]},
        {"num": 5, "label": "Linear classifier", "sub": ["-> 8 scene classes"]},
    ], ACCENT["b6"], "73.2% test acc. (65 ep.)", "-1.5 pts vs. paper 74.7%", False)

# ---- B7 ----
pipeline_diagram(
    f"{OUT}/b7_pipeline.png", "B7", "full two-stage hierarchical model, minus team pooling",
    "LSTM1 (full 9-step output) -> pool per frame -> LSTM2 (scene-level) -> scene class.",
    [
        {"num": 1, "label": "Per-player tracked crop", "sub": ["same real-tracked", "sequences as B5"],
         "image": ASSETS["spiker_seq"], "caption": "9-frame per-player sequence", "w": 400, "photo_h": 100},
        {"num": 2, "label": "B3-A backbone", "sub": ["frozen, cached"]},
        {"num": 3, "label": "LSTM1 per player", "sub": ["FULL 9-step output kept,", "not just the last state"]},
        {"num": 4, "label": "Pool at each frame", "sub": ["all players' state at t", "-> 9-step pooled seq."]},
        {"num": 5, "label": "LSTM2 (scene-level)", "sub": ["over the pooled", "9-step sequence"]},
        {"num": 6, "label": "Linear classifier", "sub": ["-> 8 scene classes"]},
    ], ACCENT["b7"], "78.8% test acc.", "-1.4 pts vs. paper 80.2%", False, stage_w=216)

# ---- B8 ----
pipeline_diagram(
    f"{OUT}/b8_pipeline.png", "B8", "paper's best config -- identical to B7, one pooling change",
    "Like B7, but pool each TEAM's 6 players separately and concat, instead of mixing all 12.",
    [
        {"num": "1-3", "label": "Same as B7", "sub": ["tracked crops -> B3-A ->", "LSTM1 per player, full output"]},
        {"num": 4, "label": "Pool PER TEAM", "sub": ["X=pool team L, Y=pool team R,", "concat(X, Y)"],
         "image": ASSETS["team_split"], "caption": "net-divided pooling: left / right team", "w": 420, "photo_h": 110},
        {"num": 5, "label": "LSTM2 (scene-level)", "sub": ["over the 9-step", "team-separated sequence"]},
        {"num": 6, "label": "Linear classifier", "sub": ["-> 8 scene classes"]},
    ], ACCENT["b8"], "88.8% test acc. (best)", "+6.9 pts vs. paper 81.9%", True, stage_w=236)

# ---- B8-mt ----
pipeline_diagram(
    f"{OUT}/b8mt_pipeline.png", "B8-mt", "B8 + a mandatory extension beyond the paper -- final model",
    "Like B8, plus a per-player 9-way action head on LSTM1's output, trained jointly with the scene loss.",
    [
        {"num": "1-3", "label": "Same as B8", "sub": ["tracked crops -> B3-A ->", "LSTM1 per player, full output"]},
        {"num": "4a", "label": "Action head (per player)", "sub": ["directly on LSTM1 output", "-> 9-way action/player"],
         "image": ASSETS["spiker_single"], "caption": 'this player -> "spiking"', "photo_h": 110, "cover": True},
        {"num": "4b", "label": "Team pool -> LSTM2", "sub": ["identical to B8's branch,", "unchanged"],
         "image": ASSETS["team_split"], "caption": "B8's pooling branch", "w": 300, "photo_h": 100},
        {"num": 5, "label": "loss = loss_scene", "sub": ["+ lambda * loss_person", "(lambda = 1.0)"]},
    ], ACCENT["b8mt"], "87.4% scene / 75.7% act.", "joint, one forward pass", True, stage_w=236)

print("ALL DONE")
