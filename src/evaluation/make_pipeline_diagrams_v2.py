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
BG = (13, 15, 20)
PANEL = (19, 23, 33)
LINE = (40, 47, 66)
LINE_SOFT = (28, 33, 46)
TEXT = (232, 235, 242)
SUBTEXT = (160, 168, 184)
GOOD = (58, 191, 134)
BAD = (255, 99, 128)
TEAM = {"L": (255, 148, 74), "R": (66, 176, 255)}
ACCENT = {
    "b1": (86, 148, 255), "b2": (255, 148, 86), "b3": (60, 191, 134),
    "b4": (176, 118, 255), "b5": (86, 217, 255), "b6": (255, 99, 128),
    "b7": (247, 189, 59), "b8": (79, 222, 216), "b8mt": (255, 133, 193),
}

F_BADGE = ImageFont.truetype("arialbd.ttf", 46)
F_PAPER = ImageFont.truetype("consola.ttf", 17)
F_MECH = ImageFont.truetype("arial.ttf", 19)
F_LABEL = ImageFont.truetype("arialbd.ttf", 22)
F_SUB = ImageFont.truetype("arial.ttf", 17)
F_CAP = ImageFont.truetype("consola.ttf", 15)
F_TAG = ImageFont.truetype("arialbd.ttf", 13)
F_NUM = ImageFont.truetype("arialbd.ttf", 18)
F_SCORE = ImageFont.truetype("arialbd.ttf", 24)
F_DELTA = ImageFont.truetype("consola.ttf", 18)
F_FOOT_LBL = ImageFont.truetype("arialbd.ttf", 14)
F_FOOT = ImageFont.truetype("arial.ttf", 17)


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


def img_player_crop(p, pad=6):
    x, y, w, h = p["x"], p["y"], p["w"], p["h"]
    box = (max(0, x - pad), max(0, y - pad), x + w + pad, y + h + pad)
    return A["center_im"].crop(box)


def img_tracked_crop_at(frame, pad=10):
    """Crop the spiker's real tracked box at a specific frame (not necessarily
    the clip's center frame) -- same real track, same ground-truth label
    ('spiking' holds for every frame in this clip's track), just picked for
    whichever frame actually shows the swing instead of the landing."""
    x1, y1, x2, y2, _ = A["tracks"][A["spiker_tid"]][frame]
    im = frame_im(frame)
    box = (max(0, x1 - pad), max(0, y1 - pad), min(im.width, x2 + pad), min(im.height, y2 + pad))
    return im.crop(box)


def img_filmstrip():
    thumbs = []
    for fr in A["frames"]:
        im = frame_im(fr)
        im.thumbnail((300, 169))
        thumbs.append((fr, im))
    w, h = thumbs[0][1].size
    gap = 5
    strip = Image.new("RGB", (w * len(thumbs) + gap * (len(thumbs) - 1), h), BG)
    d = ImageDraw.Draw(strip)
    for i, (fr, im) in enumerate(thumbs):
        x = i * (w + gap)
        strip.paste(im, (x, 0))
        tag = "t=0" if fr == CENTER else f"{fr - CENTER:+d}"
        d.rectangle([x, 0, x + 42, 19], fill=(0, 0, 0))
        d.text((x + 4, 2), tag, fill=(255, 255, 255), font=F_CAP)
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
    th = 230
    resized = []
    for fr, im in crops:
        r = th / im.height
        resized.append((fr, im.resize((max(1, int(im.width * r)), th))))
    gap = 5
    total_w = sum(im.width for _, im in resized) + gap * (len(resized) - 1)
    strip = Image.new("RGB", (total_w, th + 20), BG)
    d = ImageDraw.Draw(strip)
    x = 0
    for fr, im in resized:
        strip.paste(im, (x, 20))
        tag = "t=0" if fr == CENTER else f"{fr - CENTER:+d}"
        d.text((x + 2, 2), tag, fill=TEXT, font=F_CAP)
        x += im.width + gap
    return strip


def img_team_split():
    def block(team):
        ps = [p for p in A["players"] if p["team"] == team]
        th = 230
        cell = 142
        gap = 6
        blk = Image.new("RGB", (cell * len(ps) + gap * (len(ps) - 1), th + 24), BG)
        bd = ImageDraw.Draw(blk)
        for i, p in enumerate(ps):
            im = img_player_crop(p, pad=4)
            r = th / im.height
            im = im.resize((min(cell, int(im.width * r)), th))
            x = i * (cell + gap)
            blk.paste(im, (x + (cell - im.width) // 2, 0))
            bd.text((x, th + 3), p["action"][:9], fill=TEAM[team], font=F_CAP)
        return blk

    left, right = block("L"), block("R")
    net_w = 28
    h = max(left.height, right.height)
    out = Image.new("RGB", (left.width + net_w + right.width, h), BG)
    out.paste(left, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([left.width, 0, left.width + net_w, h], fill=LINE)
    d.line([left.width + net_w // 2, 0, left.width + net_w // 2, h], fill=TEXT, width=2)
    out.paste(right, (left.width + net_w, 0))
    return out


def img_all_players_grid():
    th, cell, cols, pad, gap = 165, 128, 6, 8, 28
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
        gd.text((cx, cy + th + 4), p["action"][:9], fill=TEAM[p["team"]], font=F_CAP)
    return grid


# pre-build the handful of real-image assets every baseline draws from
ASSETS = {
    # frame 94538 (t=-2): mid-swing, arm cocked back -- the clearest "spiking"
    # frame in this clip; the exact center frame (t=0) catches the landing
    # instead, same real player, same real "spiking" ground-truth track.
    "spiker_single": img_tracked_crop_at(94538),
    "player_grid": img_all_players_grid(),
    "filmstrip": img_filmstrip(),
    "spiker_seq": img_spiker_sequence(),
    "team_split": img_team_split(),
}


# ============================================================
# 2. Composition primitives
# ============================================================
STAGE_W_DEFAULT = 300
STAGE_H = 540
PHOTO_H = 270
MARGIN = 52
GAP = 54
TOP = 158
FOOT_H = 96
BOTTOM_PAD = 40


def draw_arrow(draw, x0, y0, x1, y1, color, width=5):
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    ah = 11
    draw.polygon([(x1, y0), (x1 - ah, y0 - ah * 0.65), (x1 - ah, y0 + ah * 0.65)], fill=color)


def draw_tag_pill(draw, x, y, text, color):
    tw = draw.textlength(text, font=F_TAG)
    draw.rounded_rectangle([x, y, x + tw + 16, y + 22], radius=11, fill=color)
    draw.text((x + 8, y + 4), text, fill=BG, font=F_TAG)
    return tw + 16


def draw_stage(draw, canvas, x, y, w, h, accent, num, label, sub_lines, image=None, caption=None,
               photo_h=PHOTO_H, cover=False, tag=None):
    draw.rounded_rectangle([x, y, x + w, y + h], radius=20, outline=accent, width=2, fill=PANEL)
    # numbered chip, top-left
    chip_r = 19
    cx, cy = x + 28, y + 28
    draw.ellipse([cx - chip_r, cy - chip_r, cx + chip_r, cy + chip_r], fill=accent)
    tw = draw.textlength(str(num), font=F_NUM)
    draw.text((cx - tw / 2, cy - 11), str(num), fill=BG, font=F_NUM)

    cursor_y = y + 60
    if image is not None:
        if tag:
            draw_tag_pill(draw, x + 16, cursor_y, tag, accent)
            cursor_y += 30
        iw = w - 32
        fitted = fit(image, iw, photo_h, cover=cover)
        canvas.paste(fitted, (x + 16, cursor_y))
        draw.rectangle([x + 16, cursor_y, x + 16 + iw, cursor_y + photo_h], outline=LINE_SOFT, width=2)
        cursor_y += photo_h + 8
        if caption:
            for line in wrap_text(draw, caption, F_CAP, w - 34):
                draw.text((x + 17, cursor_y), line, fill=SUBTEXT, font=F_CAP)
                cursor_y += 20
            cursor_y += 4
    cursor_y += 10
    for line in wrap_text(draw, label, F_LABEL, w - 32):
        draw.text((x + 17, cursor_y), line, fill=TEXT, font=F_LABEL)
        cursor_y += 28
    cursor_y += 6
    for s in sub_lines:
        for line in wrap_text(draw, s, F_SUB, w - 32):
            draw.text((x + 17, cursor_y), line, fill=SUBTEXT, font=F_SUB)
            cursor_y += 23
    return cursor_y


def draw_dashed_rect(draw, box, color, dash=10, gap=7, width=2):
    x0, y0, x1, y1 = box
    x = x0
    while x < x1:
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=width)
        x += dash + gap
    y = y0
    while y < y1:
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=width)
        y += dash + gap


def pipeline_diagram(path, baseline_id, paper_note, title, stages, accent, result_text, delta_text, delta_good,
                      ablation, stage_w=None):
    n = len(stages)
    sw = stage_w or STAGE_W_DEFAULT
    widths = [st.get("w", sw) for st in stages]
    width = MARGIN * 2 + sum(widths) + (n - 1) * GAP
    height = TOP + STAGE_H + 28 + FOOT_H + BOTTOM_PAD
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    # top accent bar
    draw.rectangle([0, 0, width, 6], fill=accent)

    # header
    draw.text((MARGIN, 26), baseline_id, fill=accent, font=F_BADGE)
    bw = draw.textlength(baseline_id, font=F_BADGE)
    pill_x = MARGIN + bw + 22
    pill_w = draw.textlength(paper_note, font=F_PAPER) + 24
    draw.rounded_rectangle([pill_x, 36, pill_x + pill_w, 66], radius=10, outline=LINE, width=1)
    draw.text((pill_x + 12, 43), paper_note, fill=SUBTEXT, font=F_PAPER)
    draw.text((MARGIN, 80), title, fill=TEXT, font=F_MECH)

    # scoreboard top-right (two lines: result readout, then delta below)
    rw = max(340, int(draw.textlength(result_text, font=F_SCORE)) + 36,
             int(draw.textlength(delta_text, font=F_DELTA)) + 36)
    rx = width - MARGIN - rw
    draw.rounded_rectangle([rx, 18, rx + rw, 78], radius=12, outline=LINE, width=1, fill=PANEL)
    draw.text((rx + 16, 24), result_text, fill=TEXT, font=F_SCORE)
    dcolor = GOOD if delta_good else BAD
    dtw = draw.textlength(delta_text, font=F_DELTA)
    draw.text((rx + rw - dtw - 16, 54), delta_text, fill=dcolor, font=F_DELTA)

    x = MARGIN
    y = TOP
    for i, st in enumerate(stages):
        w = st.get("w", sw)
        draw_stage(draw, canvas, x, y, w, STAGE_H, accent, st["num"], st["label"], st.get("sub", []),
                   image=st.get("image"), caption=st.get("caption"), photo_h=st.get("photo_h", PHOTO_H),
                   cover=st.get("cover", False), tag=st.get("tag"))
        if i < n - 1:
            nx = x + w + GAP
            draw_arrow(draw, x + w + 8, y + STAGE_H / 2, nx - 8, y + STAGE_H / 2, accent)
        x += w + GAP

    # footer: what this baseline isolates, as a standalone callout baked into the figure
    fy0 = TOP + STAGE_H + 28
    fy1 = fy0 + FOOT_H
    draw_dashed_rect(draw, [MARGIN, fy0, width - MARGIN, fy1], accent)
    lbl = "ISOLATES"
    draw.rounded_rectangle([MARGIN + 16, fy0 + 14, MARGIN + 16 + draw.textlength(lbl, font=F_FOOT_LBL) + 20, fy0 + 38],
                            radius=10, fill=accent)
    draw.text((MARGIN + 26, fy0 + 19), lbl, fill=BG, font=F_FOOT_LBL)
    text_x = MARGIN + 16 + draw.textlength(lbl, font=F_FOOT_LBL) + 40
    lines = wrap_text(draw, ablation, F_FOOT, width - MARGIN - text_x - 24)
    ty = fy0 + (FOOT_H - len(lines) * 24) / 2
    for line in lines:
        draw.text((text_x, ty), line, fill=TEXT, font=F_FOOT)
        ty += 24

    canvas.save(path)
    print("saved", path, canvas.size)


# ============================================================
# 3. One call per baseline
# ============================================================

# ---- B1 ----
pipeline_diagram(
    f"{OUT}/b1_pipeline.png", "B1-tuned", "paper: fine-tuned AlexNet -> this repo: ResNet50",
    "Whole center frame -> fine-tuned ResNet50 -> 8-way scene class. No players, no time.",
    [
        {"num": 1, "label": "Center frame", "sub": ["One whole-scene RGB image,", "1280x720, no crops, no boxes."],
         "image": A["center_im"], "caption": "t = 0 (the annotated frame)", "tag": "REAL FRAME"},
        {"num": 2, "label": "ResNet50", "sub": ["Fine-tuned end-to-end:", "every weight in the backbone",
                                                 "is trainable, not just the head."]},
        {"num": 3, "label": "Global pool", "sub": ["Spatial average over the", "last feature map ->",
                                                     "one 2048-d vector."]},
        {"num": 4, "label": "Linear classifier", "sub": ["Single trained layer", "-> 8-way scene class",
                                                           "(l-pass, r_spike, ...)"]},
    ], ACCENT["b1"], "72.2% test acc.", "+5.5 pts vs. paper 66.7%", True,
    "the no-player, no-time floor -- how far does a single static image get you on its own?")

# ---- B2 ----
pipeline_diagram(
    f"{OUT}/b2_pipeline.png", "B2", 'paper name: "Person Classification"',
    "Every player crop -> FROZEN ImageNet ResNet50 -> max-pool -> small trained classifier -> scene class.",
    [
        {"num": 1, "label": "All player crops", "sub": ["Every player's bounding-box", "crop in the clip (up to 12),",
                                                          "same center frame as B1."],
         "image": ASSETS["player_grid"], "caption": "orange = left team, blue = right team",
         "w": 440, "photo_h": 200, "tag": "REAL CROPS x12"},
        {"num": 2, "label": "ResNet50 -- frozen", "sub": ["Plain off-the-shelf ImageNet", "weights. Never fine-tuned",
                                                            "on this dataset at all."]},
        {"num": 3, "label": "Max-pool players", "sub": ["12 x 2048-d feature vectors", "-> one 2048-d vector",
                                                          "for the whole clip."]},
        {"num": 4, "label": "Small classifier", "sub": ["The only part that trains", "-> 8-way scene class."]},
    ], ACCENT["b2"], "42.2% test acc.", "-22.4 pts vs. paper 64.6%", False,
    "person-level input on its own, with zero task-specific training -- the gap to B3 is what fine-tuning is worth.")

# ---- B3 ----
pipeline_diagram(
    f"{OUT}/b3_pipeline.png", "B3", "two steps: A (teach a backbone), then B+C (reuse B2's pipeline)",
    "Step A fine-tunes on person action; steps B+C freeze it and pool across players -> scene class.",
    [
        {"num": "1A", "label": "One player crop", "sub": ["Target for step A: 9-way", "PERSON action, not the",
                                                            "8-way scene class."],
         "image": ASSETS["spiker_single"], "caption": 'ground-truth label: "spiking" (t = -2, mid-swing)',
         "photo_h": 230, "tag": "REAL CROP"},
        {"num": "2A", "label": "ResNet50 fine-tuned", "sub": ["On person-action labels", "(this sub-step = \"B3-A\"):",
                                                                "80.6% 9-way accuracy."]},
        {"num": "1BC", "label": "Freeze, re-run on all 12", "sub": ["Same now-fine-tuned backbone,",
                                                                     "every player crop in the clip."],
         "image": ASSETS["player_grid"], "caption": "same 12 crops as B2, different backbone",
         "w": 440, "photo_h": 190, "tag": "REAL CROPS x12"},
        {"num": "2BC", "label": "Max-pool + classify", "sub": ["Identical to B2's steps 3-4", "-> 8-way scene class."]},
    ], ACCENT["b3"], "65.7% test acc. (B+C)", "-2.4 pts vs. paper 68.1%", False,
    "whether fine-tuning the backbone (even on a proxy task) beats person-level input alone (B2).")

# ---- B4 ----
pipeline_diagram(
    f"{OUT}/b4_pipeline.png", "B4", "paper's \"Implementation #1\" -- image-level temporal",
    "9-frame window of whole images -> B1's frozen backbone (cached) -> one LSTM -> scene class.",
    [
        {"num": 1, "label": "9-frame window", "sub": ["Whole images, centered on", "the annotated frame --",
                                                         "still no player boxes."],
         "image": ASSETS["filmstrip"], "caption": "t = -4 ... 0 (center) ... +4",
         "w": 620, "photo_h": 175, "tag": "9 REAL FRAMES"},
        {"num": 2, "label": "B1 backbone", "sub": ["Frozen; features cached", "to disk beforehand, no",
                                                     "backbone training here."]},
        {"num": 3, "label": "Single LSTM", "sub": ["Runs over the 9-step", "sequence of 2048-d", "frame vectors."]},
        {"num": 4, "label": "Linear classifier", "sub": ["On the LSTM's final", "hidden state ->",
                                                           "8-way scene class."]},
    ], ACCENT["b4"], "73.4% test acc.", "+10.3 pts vs. paper 63.1%", True,
    "whether adding time alone (still no players) helps over B1 -- the paper found it hurt; here it helps.")

# ---- B5 ----
pipeline_diagram(
    f"{OUT}/b5_pipeline.png", "B5", 'paper\'s "Temporal on crops" -- players arrive here',
    "Per-player tracked crop sequence -> B3-A backbone -> LSTM1 per player -> pool -> scene class.",
    [
        {"num": 1, "label": "Per-player tracked crop", "sub": ["This player's OWN box in", "each of the 9 frames --",
                                                                 "not one box reused 9x."],
         "image": ASSETS["spiker_seq"], "caption": "the spiker's real tracked box, t = -4 ... +4",
         "w": 680, "photo_h": 185, "tag": "9 REAL TRACKED CROPS"},
        {"num": 2, "label": "B3-A backbone", "sub": ["Frozen, cached --", "2048-d feature per", "player per frame."]},
        {"num": 3, "label": "LSTM1 per player", "sub": ["Shared weights, one run", "per player -> keeps only",
                                                          "the final hidden state."]},
        {"num": 4, "label": "Max-pool players", "sub": ["Across all 12 players'", "final hidden states ->",
                                                          "one clip-level vector."]},
        {"num": 5, "label": "Linear classifier", "sub": ["-> 8-way scene class.", "No scene-level LSTM", "yet (that's B7)."]},
    ], ACCENT["b5"], "79.0% test acc. (tracked)", "+11.4 pts vs. paper 67.6%", True,
    "whether time helps once players are already in the picture (B3 had players, no time).")

# ---- B6 ----
pipeline_diagram(
    f"{OUT}/b6_pipeline.png", "B6", "same ingredients as B5, pooling and LSTM order reversed",
    "Pool players per frame first -> single LSTM over the pooled 9-step sequence -> scene class.",
    [
        {"num": 1, "label": "Each of 9 frames", "sub": ["All player crops in that", "frame, read independently",
                                                          "frame by frame."],
         "image": ASSETS["filmstrip"], "caption": "same 9-frame window as B4, now read per-player",
         "w": 620, "photo_h": 175, "tag": "9 REAL FRAMES"},
        {"num": 2, "label": "B3-A backbone", "sub": ["Frozen, cached,", "per player per frame",
                                                       "(same features as B5)."]},
        {"num": 3, "label": "Max-pool per frame", "sub": ["12 players -> 1 vector,", "repeated 9x -> a",
                                                            "9-step pooled sequence."]},
        {"num": 4, "label": "Single LSTM", "sub": ["Over the pooled sequence --", "NO per-player LSTM1",
                                                     "like B5 has."]},
        {"num": 5, "label": "Linear classifier", "sub": ["-> 8-way scene class."]},
    ], ACCENT["b6"], "73.2% test acc. (65 ep.)", "-1.5 pts vs. paper 74.7%", False,
    "whether pooling-before-LSTM (B6) vs. LSTM-before-pooling (B5) changes what the model actually learns.")

# ---- B7 ----
pipeline_diagram(
    f"{OUT}/b7_pipeline.png", "B7", "full two-stage hierarchical model, minus team pooling",
    "LSTM1 (full 9-step output) -> pool per frame -> LSTM2 (scene-level) -> scene class.",
    [
        {"num": 1, "label": "Per-player tracked crop", "sub": ["Same real-tracked per-player", "sequences as B5."],
         "image": ASSETS["spiker_seq"], "caption": "9-frame per-player sequence (same as B5)",
         "w": 560, "photo_h": 160, "tag": "9 REAL TRACKED CROPS"},
        {"num": 2, "label": "B3-A backbone", "sub": ["Frozen, cached."]},
        {"num": 3, "label": "LSTM1 per player", "sub": ["Keeps the FULL 9-step", "output this time, not",
                                                          "just the last state."]},
        {"num": 4, "label": "Pool at each frame", "sub": ["All players' hidden state", "at frame t -> a 9-step",
                                                            "pooled sequence."]},
        {"num": 5, "label": "LSTM2 (scene-level)", "sub": ["A second LSTM, over the", "pooled 9-step sequence."]},
        {"num": 6, "label": "Linear classifier", "sub": ["-> 8-way scene class."]},
    ], ACCENT["b7"], "78.8% test acc.", "-1.4 pts vs. paper 80.2%", False,
    "whether a second, scene-level LSTM on top of the per-player one adds anything beyond B5/B6.", stage_w=258)

# ---- B8 ----
pipeline_diagram(
    f"{OUT}/b8_pipeline.png", "B8", "paper's best config -- identical to B7, one pooling change",
    "Like B7, but pool each TEAM's 6 players separately and concat, instead of mixing all 12.",
    [
        {"num": "1-3", "label": "Same as B7", "sub": ["Tracked crops -> B3-A ->", "LSTM1 per player, full",
                                                        "9-step output."]},
        {"num": 4, "label": "Pool PER TEAM", "sub": ["X = pool team L's 6 players,", "Y = pool team R's 6 players,",
                                                       "scene vector = concat(X, Y)."],
         "image": ASSETS["team_split"], "caption": "net-divided pooling: left team | right team",
         "w": 620, "photo_h": 185, "tag": "REAL CROPS, TEAM-SPLIT"},
        {"num": 5, "label": "LSTM2 (scene-level)", "sub": ["Over the 9-step", "team-separated sequence."]},
        {"num": 6, "label": "Linear classifier", "sub": ["-> 8-way scene class."]},
    ], ACCENT["b8"], "88.8% test acc. (best)", "+6.9 pts vs. paper 81.9%", True,
    "whether keeping which-side-of-the-court information (instead of mixing both teams) fixes winpoint confusion.",
    stage_w=272)

# ---- B8-mt ----
pipeline_diagram(
    f"{OUT}/b8mt_pipeline.png", "B8-mt", "B8 + a mandatory extension beyond the paper -- final model",
    "Like B8, plus a per-player 9-way action head on LSTM1's output, trained jointly with the scene loss.",
    [
        {"num": "1-3", "label": "Same as B8", "sub": ["Tracked crops -> B3-A ->", "LSTM1 per player, full",
                                                        "9-step output."]},
        {"num": "4a", "label": "Action head (per player)", "sub": ["Attached directly to LSTM1's", "output, before any pooling",
                                                                     "-> 9-way action per player."],
         "image": ASSETS["spiker_single"], "caption": 'this player -> "spiking" (t = -2, mid-swing)',
         "photo_h": 185, "tag": "REAL CROP"},
        {"num": "4b", "label": "Team pool -> LSTM2", "sub": ["Identical to B8's branch,", "completely unchanged."],
         "image": ASSETS["team_split"], "caption": "B8's team-split pooling branch, reused as-is",
         "w": 460, "photo_h": 160, "tag": "REAL CROPS, TEAM-SPLIT"},
        {"num": 5, "label": "Joint loss", "sub": ["loss = loss_scene", "+ lambda * loss_person", "(lambda = 1.0)"]},
    ], ACCENT["b8mt"], "87.4% scene / 75.7% act.", "joint, one forward pass", True,
    "nothing -- this is the deliverable: one forward pass producing both the scene label and every player's action.",
    stage_w=262)

print("ALL DONE")
