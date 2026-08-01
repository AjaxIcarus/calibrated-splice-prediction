from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def annotate(ax, bars):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.2e}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
            rotation=18,
        )


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    }
)

# Primary sampled-cache comparison. Values are the frozen flank-400 epoch-8
# point estimates under target-position weighting.
methods = ["Uncal.", "Genome-\nweighted", "OSAI-style"]
ece = np.array([0.00163185, 0.00002899, 0.00001168])
nll = np.array([0.00183797, 0.00026765, 0.00025782])
colors = ["#777777", "#2676b8", "#d95f02"]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), constrained_layout=True)
for ax, values, title, ylabel in [
    (axes[0], ece, "Multiclass ECE", "ECE (log scale)"),
    (axes[1], nll, "Negative log-likelihood", "NLL (log scale)"),
]:
    bars = ax.bar(methods, values, color=colors, width=0.68)
    ax.set_yscale("log")
    ax.set_ylim(values.min() / 1.6, values.max() * 2.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=7.5)
    style_axis(ax)
    annotate(ax, bars)
fig.suptitle("Flank-400 calibration under target-position weighting", fontweight="bold")
fig.savefig(OUT / "flank400_primary_calibration.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Exhaustive chr9 gene/transcript-position comparison. The two metrics are
# reported separately because ECE and NLL can rank prior-mismatched methods
# differently.
methods = ["Uncal.", "Global", "Unweighted", "Genome-\nweighted", "OSAI-style"]
ece = np.array([0.0017323952, 0.0028382773, 0.0000813442, 0.0000370374, 0.0000032078])
nll = np.array([0.0019442024, 0.0030600559, 0.0024446312, 0.0002797268, 0.0002671157])
colors = ["#777777", "#bdbdbd", "#7b6fd0", "#2676b8", "#d95f02"]

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
for ax, values, title, ylabel in [
    (axes[0], ece, "Multiclass ECE", "ECE (log scale)"),
    (axes[1], nll, "Negative log-likelihood", "NLL (log scale)"),
]:
    bars = ax.bar(methods, values, color=colors, width=0.72)
    ax.set_yscale("log")
    ax.set_ylim(values.min() / 1.7, values.max() * 2.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=7, rotation=18)
    style_axis(ax)
    annotate(ax, bars)
fig.suptitle("Flank-400 exhaustive chr9 gene/transcript-position evaluation", fontweight="bold")
fig.savefig(OUT / "flank400_chr9_calibration.png", dpi=300, bbox_inches="tight")
plt.close(fig)
