"""Shared plot style and small helpers so every figure looks consistent."""
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path("figures")

# One fixed colour per model, used in every figure.
COLORS = {
    "reference": "#222222",
    "mse": "#d62728",
    "mae": "#9467bd",
    "single": "#ff7f0e",
    "mdn": "#1f77b4",
    "branch": "#2ca02c",
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
        "figure.constrained_layout.use": True,
    })


def savefig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")


def plot_history(ax, hist, title, ylabel):
    ax.plot(hist["train"], label="train", lw=1.2)
    ax.plot(hist["val"], label="validation", lw=1.2)
    ax.axvline(hist["best_epoch"], color="k", ls=":", lw=1, label=f"best val (epoch {hist['best_epoch']})")
    ax.set(xlabel="epoch", ylabel=ylabel, title=title)
    ax.legend()
