import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

BLUE = "#2a78d6"      # V
ORANGE = "#eb6834"    # M
AQUA = "#1baf7a"       # C

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.2)
ax.axis("off")


def box(x, y, w, h, color, title, subtitle, lines):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=2.0, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center",
            fontsize=15, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.78, subtitle, ha="center", va="center",
            fontsize=9.5, color=color, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 1.22 - i * 0.32, line, ha="center", va="center",
                fontsize=8.7, color=SECONDARY)


def arrow(x0, y0, x1, y1, color=MUTED, label=None, label_dy=0.18):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.8, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + label_dy, label, ha="center", va="bottom",
                fontsize=8.3, color=SECONDARY, style="italic")


# --- Boxes ---
box(0.3, 1.0, 2.7, 3.6, BLUE, "V", "ConvVAE (vision)",
    ["64x64 grayscale frame", "-> latent z (32-d)", "cactus & bird shapes", "encoded as z"])

box(3.75, 1.0, 3.1, 3.6, ORANGE, "M", "MDN-RNN (memory)",
    ["GRU(256) over [z, a]", "predicts P(z_t+1 | z_t, a_t, h_t)", "5-Gaussian mixture + done head",
     "hallucinates the next frame's z"])

box(7.6, 1.0, 3.1, 3.6, AQUA, "C", "Linear Controller",
    ["input: [z, h]  (288-d)", "-> run / jump / duck", "trained with CMA-ES", "entirely inside M's dream"])

ax.text(5.5, 4.95, "World Model — Dino Runner  (V -> M -> C)", ha="center", va="center",
        fontsize=15, fontweight="bold", color=INK)

# z from V to M
arrow(3.05, 3.2, 3.7, 3.2, color=BLUE, label="z")
# h feedback M -> C
arrow(6.85, 3.2, 7.55, 3.2, color=ORANGE, label="z, h")
# action feedback C -> M (loop back below)
a = FancyArrowPatch((7.7, 1.15), (3.9, 1.15), arrowstyle="-|>", mutation_scale=16,
                     linewidth=1.8, color=AQUA, connectionstyle="arc3,rad=-0.25")
ax.add_patch(a)
ax.text(5.7, 0.55, "action a_t fed back into M each dream step", ha="center", va="center",
        fontsize=8.6, color=SECONDARY, style="italic")

plt.tight_layout()
out = __file__.replace("make_architecture_diagram.py", "architecture_diagram.png")
plt.savefig(out, facecolor=SURFACE)
print(f"Saved {out}")
