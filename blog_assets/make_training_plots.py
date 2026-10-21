import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel(xlabel, color=SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=SECONDARY, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def line_chart(x, series, colors, labels, title, xlabel, ylabel, out_name, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax, title, xlabel, ylabel)
    if logy:
        ax.set_yscale("log")
    for y, c, lab in zip(series, colors, labels):
        ax.plot(x, y, color=c, linewidth=2.2, solid_capstyle="round", label=lab)
        ax.plot(x[-1], y[-1], "o", color=c, markersize=5)
    if len(series) > 1:
        leg = ax.legend(frameon=False, loc="upper right", fontsize=9.5)
        for text in leg.get_texts():
            text.set_color(SECONDARY)
    plt.tight_layout()
    out = os.path.join(HERE, out_name)
    plt.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


def make_vae_plot():
    p = os.path.join(ROOT, "checkpoints", "vae_training_history.npz")
    if not os.path.exists(p):
        print("skip VAE plot (no history yet)")
        return
    d = np.load(p)
    line_chart(d["epoch"], [d["loss"]], [BLUE], ["loss / frame"],
               "V: VAE reconstruction loss", "Epoch", "Avg loss per frame",
               "vae_loss_curve.png")
    line_chart(d["epoch"], [d["recon"], d["kl"]], [BLUE, ORANGE], ["Reconstruction", "KL divergence"],
               "V: VAE loss breakdown", "Epoch", "Avg per frame (log scale)",
               "vae_loss_breakdown.png", logy=True)


def make_mdnrnn_plot():
    p = os.path.join(ROOT, "checkpoints", "mdnrnn_training_history.npz")
    if not os.path.exists(p):
        print("skip MDN-RNN plot (no history yet)")
        return
    d = np.load(p)
    line_chart(d["epoch"], [d["nll"]], [BLUE], ["negative log-likelihood"],
               "M: MDN-RNN next-latent prediction loss", "Epoch", "Avg NLL",
               "mdnrnn_nll_curve.png")
    line_chart(d["epoch"], [d["bce"]], [ORANGE], ["collision (done) BCE"],
               "M: collision-prediction loss", "Epoch", "Avg BCE",
               "mdnrnn_done_curve.png")


def make_controller_plot():
    p = os.path.join(ROOT, "checkpoints", "controller_training_history.npz")
    if not os.path.exists(p):
        print("skip controller plot (no history yet)")
        return
    d = np.load(p)
    line_chart(d["gen"], [d["best"], d["mean"]], [BLUE, ORANGE], ["Best of generation", "Population mean"],
               "C: CMA-ES fitness in dream rollouts", "Generation", "Dream survival steps",
               "controller_fitness_curve.png")


def make_eval_histogram():
    p = os.path.join(ROOT, "eval_scores.npy")
    if not os.path.exists(p):
        print("skip eval histogram (no eval scores yet)")
        return
    scores = np.load(p)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_ax(ax, "Real-env survival across evaluation episodes", "Steps survived", "Episodes")
    ax.hist(scores, bins=20, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(scores.mean(), color=ORANGE, linewidth=2, linestyle="--")
    ax.text(scores.mean(), ax.get_ylim()[1] * 0.95, f" mean={scores.mean():.0f}",
            color=ORANGE, fontsize=9.5, va="top")
    plt.tight_layout()
    out = os.path.join(HERE, "eval_histogram.png")
    plt.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_vae_plot()
    make_mdnrnn_plot()
    make_controller_plot()
    make_eval_histogram()
