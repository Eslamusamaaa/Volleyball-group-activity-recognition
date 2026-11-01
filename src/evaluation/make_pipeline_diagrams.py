"""Regenerate nicer pipeline diagrams + combined accuracy/confusion-matrix
panels for B1-B6. Run once, outputs land in outputs/results/.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

CLASSES = ['l-pass', 'l-spike', 'l_set', 'l_winpoint', 'r-pass', 'r_set', 'r_spike', 'r_winpoint']

# ---------- shared design tokens ----------
BG = "#0f1117"
PANEL = "#171a23"
GRID = "#2a2f3d"
TEXT = "#e6e8ef"
SUBTEXT = "#9aa2b1"
ACCENT = {
    "b1": "#4f8cff", "b2": "#ff8a4f", "b3": "#36b37e",
    "b4": "#a66bff", "b5": "#4fd1ff", "b6": "#ff5c7a",
}


INV_SQRT2 = 0.70710678

def _icon_court_frame(ax, cx, cy, r, color, asp):
    """A little volleyball court seen from above: rectangle + net line + posts."""
    rx, ry = r, r * asp
    w, h = rx * 1.9, ry * 1.15
    ax.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                                    edgecolor=color, lw=1.4, zorder=3))
    ax.plot([cx, cx], [cy - h / 2 - ry * 0.12, cy + h / 2 + ry * 0.12], color=color, lw=1.6, zorder=3)
    for s in (-1, 1):
        ax.plot([cx + s * w * 0.28], [cy], marker="o", markersize=3.2, color=color, zorder=3)


def _icon_volleyball(ax, cx, cy, r, color, asp):
    """A volleyball: circle with curved seam arcs, like the real ball panels."""
    rx, ry = r, r * asp
    ax.add_patch(patches.Ellipse((cx, cy), rx * 2, ry * 2, fill=False, edgecolor=color, lw=1.5, zorder=3))
    t = np.linspace(-1, 1, 40)
    # vertical-ish seam
    ax.plot(cx + np.sin(t * 1.3) * rx * 0.55, cy + t * ry, color=color, lw=1.1, zorder=3)
    # two diagonal seams
    ax.plot(cx + t * rx * 0.85, cy + np.sin(t * 1.3 + 1.0) * ry * 0.55, color=color, lw=1.1, zorder=3)
    ax.plot(cx + t * rx * 0.85, cy - np.sin(t * 1.3 + 1.0) * ry * 0.35, color=color, lw=1.1, zorder=3)


def _icon_player(ax, cx, cy, r, color, asp):
    """Simple player token: head circle + rounded body, upright."""
    rx, ry = r, r * asp
    ax.add_patch(patches.Ellipse((cx, cy + ry * 0.5), rx * 0.62, ry * 0.62,
                                  facecolor=color, edgecolor="none", zorder=3))
    body = patches.Rectangle((cx - rx * 0.45, cy - ry * 0.75), rx * 0.9, ry * 1.05,
                              facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(body)
    # rounded cap on top of the body so it doesn't look like a plain block
    ax.add_patch(patches.Ellipse((cx, cy - ry * 0.75 + ry * 1.05), rx * 0.9, ry * 0.3,
                                  facecolor=color, edgecolor="none", zorder=3))


def _icon_pulse(ax, cx, cy, r, color, asp):
    """A little motion/heartbeat line, standing in for 'sequence over time'."""
    rx, ry = r, r * asp
    xs = np.linspace(-rx, rx, 60)
    ys = np.zeros_like(xs)
    mid = np.abs(xs) < rx * 0.5
    ys[mid] = np.sin((xs[mid] / (rx * 0.5)) * np.pi * 1.5) * ry * 0.65
    ax.plot(cx + xs, cy + ys, color=color, lw=1.6, zorder=3)
    for dx in (-1, 1):
        ax.add_patch(patches.Ellipse((cx + dx * rx * 0.82, cy), rx * 0.09, ry * 0.09,
                                      facecolor=color, edgecolor="none", zorder=3))


def _icon_huddle(ax, cx, cy, r, color, asp):
    """Two overlapping circles = pooling players' info together."""
    rx, ry = r, r * asp
    for dx in (-0.24, 0.24):
        ax.add_patch(patches.Ellipse((cx + dx * rx, cy), rx * 0.95, ry * 0.95,
                                      fill=False, edgecolor=color, lw=1.4, zorder=3))


def _icon_scoreboard(ax, cx, cy, r, color, asp):
    """A little scoreboard flag = final classification result."""
    rx, ry = r, r * asp
    ax.plot([cx - rx * 0.6, cx - rx * 0.6], [cy - ry * 0.75, cy + ry * 0.8], color=color, lw=1.6, zorder=3)
    flag = patches.Polygon([[cx - rx * 0.6, cy + ry * 0.8], [cx + rx * 0.65, cy + ry * 0.48],
                             [cx - rx * 0.6, cy + ry * 0.16]], closed=True, facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(flag)


ICONS = {
    "input": _icon_court_frame,
    "backbone": _icon_volleyball,
    "temporal": _icon_pulse,
    "pool": _icon_huddle,
    "output": _icon_scoreboard,
    "player": _icon_player,
}


def pipeline_diagram(path, title, steps, accent, figsize=(13, 3.2)):
    """steps: list of (label_lines:list[str], kind) where kind picks a
    hand-drawn volleyball-themed icon (ball / court / player / pulse / huddle / flag)."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    # data-unit x and y spans are both 1, but the figure itself is far wider
    # than it is tall, so 1 x-unit != 1 y-unit physically. This factor lets
    # icon helpers draw in "physically equal" radii (rx data-units in x,
    # rx*asp data-units in y) so circles/icons don't come out squashed.
    asp = figsize[0] / figsize[1]

    margin = 0.045
    gap = 0.03
    w = (1 - 2 * margin - gap * (n - 1)) / n
    h = 0.5
    y = 0.16

    for i, (lines, kind) in enumerate(steps):
        x = margin + i * (w + gap)
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
                              linewidth=1.8, edgecolor=accent, facecolor=PANEL, zorder=2)
        ax.add_patch(box)
        chip_rx = 0.016
        ax.add_patch(patches.Ellipse((x + 0.026, y + h - 0.045), chip_rx * 2, chip_rx * 2 * asp,
                                      facecolor=accent, edgecolor="none", zorder=4))
        ax.text(x + 0.026, y + h - 0.045, str(i + 1), ha="center", va="center",
                fontsize=8.5, color=BG, weight="bold", zorder=5)
        icon_fn = ICONS.get(kind)
        if icon_fn:
            rx = min(w * 0.26, 0.085 / asp)
            icon_fn(ax, x + w / 2, y + h - 0.175, rx, accent, asp)
        ty = y + h - 0.30
        for line in lines:
            ax.text(x + w / 2, ty, line, ha="center", va="center",
                    fontsize=9.3, color=TEXT, zorder=3)
            ty -= 0.075
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.006, y + h / 2), xytext=(x + w + 0.006, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.0, color=accent, shrinkA=0, shrinkB=0,
                                         mutation_scale=16),
                        zorder=1)

    ax.text(0.5, 0.925, title, ha="center", va="center", fontsize=12.5, color=TEXT, weight="bold")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    print("saved", path)


OUT = "outputs/results"

# ---------------- B1 ----------------
pipeline_diagram(f"{OUT}/b1_pipeline.png", "B1-tuned — whole image → fine-tuned ResNet50 → scene class",
    [(["Center frame", "(whole scene)"], "input"),
     (["ResNet50", "(fine-tuned,", "all weights)"], "backbone"),
     (["Global pool", "2048-d vector"], "pool"),
     (["Linear classifier", "→ 8 scene classes"], "output")], ACCENT["b1"])

# ---------------- B2 ----------------
pipeline_diagram(f"{OUT}/b2_pipeline.png", "B2 — frozen backbone, pooled person crops → scene class",
    [(["All player crops", "(up to 12)"], "player"),
     (["ResNet50", "(FROZEN,", "ImageNet only)"], "backbone"),
     (["Max-pool", "across players"], "pool"),
     (["Small classifier", "(trained)", "→ 8 scene classes"], "output")], ACCENT["b2"])

# ---------------- B3 ----------------
pipeline_diagram(f"{OUT}/b3_pipeline.png", "B3 — fine-tuned person backbone, pooled → scene class",
    [(["Each player crop"], "player"),
     (["ResNet50 FINE-TUNED", "on person action", "(= step A)"], "backbone"),
     (["FROZEN after:", "extract + pool", "across players"], "pool"),
     (["Small classifier", "→ 8 scene classes"], "output")], ACCENT["b3"])

# ---------------- B4 ----------------
pipeline_diagram(f"{OUT}/b4_pipeline.png", "B4 — image-level LSTM over B1 features → scene class",
    [(["9-frame window", "(whole images)"], "input"),
     (["B1 backbone", "(frozen, cached)"], "backbone"),
     (["Single LSTM", "over 9 steps"], "temporal"),
     (["Linear classifier", "→ 8 scene classes"], "output")], ACCENT["b4"])

# ---------------- B5 ----------------
pipeline_diagram(f"{OUT}/b5_pipeline.png", "B5 — per-player LSTM, pooled at the end → scene class",
    [(["Each player,", "real tracked box", "per frame (9)"], "player"),
     (["B3-A backbone", "(frozen, cached)"], "backbone"),
     (["LSTM1 per player", "(9-step sequence)"], "temporal"),
     (["Max-pool final", "hidden states"], "pool"),
     (["Linear classifier", "→ 8 scene classes"], "output")], ACCENT["b5"], figsize=(14.5, 3.0))

# ---------------- B6 ----------------
pipeline_diagram(f"{OUT}/b6_pipeline.png", "B6 — pool players per frame, then single LSTM → scene class",
    [(["Each frame (9):", "all player crops"], "player"),
     (["B3-A backbone", "(frozen, cached)"], "backbone"),
     (["Max-pool per frame", "→ 9-step sequence"], "pool"),
     (["Single LSTM", "over 9 steps"], "temporal"),
     (["Linear classifier", "→ 8 scene classes"], "output")], ACCENT["b6"], figsize=(14.5, 3.0))

print("ALL DONE")
