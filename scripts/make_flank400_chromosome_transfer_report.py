from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]

METHODS = [
    "uncalibrated",
    "fixed_global_T1p1",
    "unweighted_vectorT",
    "genome_weighted_vectorT",
    "openspliceai_style_vectorT",
]

METHOD_LABELS = {
    "uncalibrated": "Uncalibrated",
    "fixed_global_T1p1": "Global T=1.1",
    "unweighted_vectorT": "Unweighted vector-T",
    "genome_weighted_vectorT": "Genome-weighted vector-T",
    "openspliceai_style_vectorT": "OpenSpliceAI-style vector-T",
}

COLORS = {
    "uncalibrated": "#000000",
    "fixed_global_T1p1": "#D55E00",
    "unweighted_vectorT": "#CC79A7",
    "genome_weighted_vectorT": "#0072B2",
    "openspliceai_style_vectorT": "#009E73",
}

MARKERS = {
    "uncalibrated": "o",
    "fixed_global_T1p1": "X",
    "unweighted_vectorT": "^",
    "genome_weighted_vectorT": "s",
    "openspliceai_style_vectorT": "D",
}

OFFSETS = {
    method: offset
    for method, offset in zip(
        METHODS,
        np.linspace(-0.24, 0.24, len(METHODS)),
    )
}

PANEL_PATH = Path(
    "results/flank400_chromosome_transfer_"
    "seed11_epoch11_by_chromosome.csv"
)

SUMMARY_PATH = Path(
    "results/flank400_chromosome_transfer_"
    "seed11_epoch11_summary_with_pooled_ece.csv"
)

FIGURE_PNG = Path(
    "figures/flank400_chromosome_transfer_"
    "seed11_epoch11.png"
)

FIGURE_PDF = Path(
    "figures/flank400_chromosome_transfer_"
    "seed11_epoch11.pdf"
)

TABLE_CSV = Path(
    "results/flank400_chromosome_transfer_"
    "seed11_epoch11_manuscript_table.csv"
)

TABLE_TEX = Path(
    "results/flank400_chromosome_transfer_"
    "seed11_epoch11_manuscript_table.tex"
)


def require_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


panel = pd.read_csv(PANEL_PATH)
summary = pd.read_csv(SUMMARY_PATH)

require_columns(
    panel,
    [
        "chrom",
        "method",
        "multiclass_ece",
        "multiclass_nll",
        "total_positions",
    ],
    "chromosome panel",
)

require_columns(
    summary,
    [
        "method",
        "total_positions",
        "multiclass_ece_pooled",
        "multiclass_nll_pooled",
        "pooled_ece_reduction_percent",
        "pooled_nll_reduction_percent",
    ],
    "pooled summary",
)

if len(panel) != len(CHROMS) * len(METHODS):
    raise ValueError(
        f"Expected {len(CHROMS) * len(METHODS)} panel rows, "
        f"found {len(panel)}"
    )

if len(summary) != len(METHODS):
    raise ValueError(
        f"Expected {len(METHODS)} summary rows, found {len(summary)}"
    )

if set(panel["chrom"]) != set(CHROMS):
    raise ValueError(
        f"Unexpected chromosome set: {sorted(panel['chrom'].unique())}"
    )

if set(panel["method"]) != set(METHODS):
    raise ValueError(
        f"Unexpected panel methods: {sorted(panel['method'].unique())}"
    )

if set(summary["method"]) != set(METHODS):
    raise ValueError(
        f"Unexpected summary methods: "
        f"{sorted(summary['method'].unique())}"
    )

if panel.duplicated(["chrom", "method"]).any():
    raise ValueError("Duplicate chromosome/method rows detected")

if summary["method"].duplicated().any():
    raise ValueError("Duplicate summary method rows detected")

for chrom in CHROMS:
    counts = panel.loc[
        panel["chrom"] == chrom,
        "total_positions",
    ].unique()

    if len(counts) != 1:
        raise ValueError(
            f"{chrom}: total positions differ across methods"
        )

panel_total = int(
    panel.drop_duplicates("chrom")["total_positions"].sum()
)

summary_totals = summary["total_positions"].unique()

if len(summary_totals) != 1:
    raise ValueError("Pooled totals differ across methods")

if panel_total != int(summary_totals[0]):
    raise ValueError(
        f"Panel total {panel_total} does not match "
        f"summary total {int(summary_totals[0])}"
    )

if panel_total != 416_140_000:
    raise ValueError(f"Unexpected pooled total: {panel_total}")

for metric in ["multiclass_ece", "multiclass_nll"]:
    best_rows = panel.loc[
        panel.groupby("chrom")[metric].idxmin()
    ]

    if not best_rows["method"].eq(
        "openspliceai_style_vectorT"
    ).all():
        raise ValueError(
            f"OpenSpliceAI-style vector-T is not best for every "
            f"chromosome on {metric}"
        )

summary_indexed = summary.set_index("method").loc[METHODS]

for metric in [
    "multiclass_ece_pooled",
    "multiclass_nll_pooled",
]:
    if (
        summary_indexed[metric].idxmin()
        != "openspliceai_style_vectorT"
    ):
        raise ValueError(
            f"Unexpected pooled best method for {metric}"
        )

panel_indexed = panel.set_index(["method", "chrom"])
x_labels = CHROMS + ["Pooled"]
x_base = np.arange(len(x_labels), dtype=np.float64)

plot_values = {}

for method in METHODS:
    chromosome_ece = [
        float(
            panel_indexed.loc[
                (method, chrom),
                "multiclass_ece",
            ]
        )
        for chrom in CHROMS
    ]

    chromosome_nll = [
        float(
            panel_indexed.loc[
                (method, chrom),
                "multiclass_nll",
            ]
        )
        for chrom in CHROMS
    ]

    pooled = summary_indexed.loc[method]

    plot_values[method] = {
        "ece": chromosome_ece
        + [float(pooled["multiclass_ece_pooled"])],
        "nll": chromosome_nll
        + [float(pooled["multiclass_nll_pooled"])],
    }

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.2, 4.1),
    sharex=True,
)

panel_specs = [
    (
        axes[0],
        "ece",
        "Multiclass ECE",
        "(a) Reliability error",
    ),
    (
        axes[1],
        "nll",
        "Multiclass NLL",
        "(b) Negative log-likelihood",
    ),
]

legend_handles = []

for ax, metric_key, ylabel, title in panel_specs:
    ax.axvspan(
        4.5,
        5.5,
        color="#F0F0F0",
        zorder=0,
    )
    ax.axvline(
        4.5,
        color="#B0B0B0",
        linewidth=0.8,
        linestyle="--",
        zorder=1,
    )

    for method in METHODS:
        handle = ax.scatter(
            x_base + OFFSETS[method],
            plot_values[method][metric_key],
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            s=43,
            linewidths=0.8,
            edgecolors=(
                "white"
                if method != "uncalibrated"
                else "#000000"
            ),
            zorder=3,
        )

        if ax is axes[0]:
            legend_handles.append(handle)

    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.55, 5.55)
    ax.grid(
        axis="y",
        which="both",
        color="#D9D9D9",
        linewidth=0.6,
        alpha=0.8,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.legend(
    legend_handles,
    [METHOD_LABELS[m] for m in METHODS],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
)

fig.suptitle(
    "Calibration transfer across held-out chromosomes "
    "(seed 11, epoch 11)",
    y=1.01,
    fontsize=11,
)

fig.tight_layout(rect=(0, 0.15, 1, 1))

FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    FIGURE_PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(fig)

table = summary_indexed.reset_index()[
    [
        "method",
        "total_positions",
        "multiclass_ece_pooled",
        "multiclass_nll_pooled",
        "pooled_ece_reduction_percent",
        "pooled_nll_reduction_percent",
    ]
].copy()

table["method"] = table["method"].map(METHOD_LABELS)

table = table.rename(
    columns={
        "multiclass_ece_pooled": "pooled_ece",
        "multiclass_nll_pooled": "pooled_nll",
        "pooled_ece_reduction_percent":
            "ece_reduction_percent",
        "pooled_nll_reduction_percent":
            "nll_reduction_percent",
    }
)

table.to_csv(TABLE_CSV, index=False)

latex = table.copy()

latex["total_positions"] = latex["total_positions"].map(
    lambda value: f"{int(value):,}"
)

latex["pooled_ece"] = latex["pooled_ece"].map(
    lambda value: f"{value:.3e}"
)

latex["pooled_nll"] = latex["pooled_nll"].map(
    lambda value: f"{value:.3e}"
)

latex["ece_reduction_percent"] = latex[
    "ece_reduction_percent"
].map(lambda value: f"{value:.2f}")

latex["nll_reduction_percent"] = latex[
    "nll_reduction_percent"
].map(lambda value: f"{value:.2f}")

best_mask = (
    latex["method"]
    == METHOD_LABELS["openspliceai_style_vectorT"]
)

for column in ["pooled_ece", "pooled_nll"]:
    latex.loc[best_mask, column] = (
        "\\textbf{"
        + latex.loc[best_mask, column]
        + "}"
    )

latex = latex.rename(
    columns={
        "method": "Method",
        "total_positions": "Positions",
        "pooled_ece": "Pooled ECE",
        "pooled_nll": "Pooled NLL",
        "ece_reduction_percent": "ECE reduction (\\%)",
        "nll_reduction_percent": "NLL reduction (\\%)",
    }
)

latex_text = latex.to_latex(
    index=False,
    escape=False,
    column_format="lrrrrr",
    caption=(
        "Five-chromosome calibration transfer for the "
        "seed-11, epoch-11 model. Reductions are relative "
        "to the uncalibrated model; negative values indicate "
        "worsening."
    ),
    label="tab:flank400_chromosome_transfer",
)

TABLE_TEX.write_text(latex_text, encoding="utf-8")

print("\nVALIDATION")
print(f"Chromosomes: {', '.join(CHROMS)}")
print(f"Methods: {len(METHODS)}")
print(f"Total held-out positions: {panel_total:,}")
print(
    "Best method on ECE and NLL for every chromosome: "
    "OpenSpliceAI-style vector-T"
)

print("\nMANUSCRIPT TABLE")
print(
    table[
        [
            "method",
            "pooled_ece",
            "pooled_nll",
            "ece_reduction_percent",
            "nll_reduction_percent",
        ]
    ].to_string(index=False)
)

print("\nWROTE")
print(FIGURE_PNG)
print(FIGURE_PDF)
print(TABLE_CSV)
print(TABLE_TEX)
