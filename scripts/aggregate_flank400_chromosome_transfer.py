#!/usr/bin/env python3
import argparse
from pathlib import Path

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
EXPECTED_POSITIONS = 416_140_000

parser = argparse.ArgumentParser()
parser.add_argument("--run-label", required=True)
parser.add_argument("--output-prefix")
args = parser.parse_args()

prefix = Path(
    args.output_prefix
    or f"results/flank400_chromosome_transfer_{args.run_label}"
)

metric_frames = []
bin_frames = []

for chrom in CHROMS:
    out_dir = Path(
        f"results/chromosome_eval_flank400_"
        f"{args.run_label}_{chrom}"
    )
    marker = out_dir / "_SUCCESS"
    metrics_path = out_dir / f"{chrom}_streaming_metrics.csv"
    bins_path = (
        out_dir / f"{chrom}_multiclass_reliability_bins.csv"
    )

    for path in [marker, metrics_path, bins_path]:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")

    metrics = pd.read_csv(metrics_path)
    bins = pd.read_csv(bins_path)

    if len(metrics) != 5 or set(metrics["method"]) != set(METHODS):
        raise SystemExit(f"Invalid method table: {metrics_path}")

    if set(metrics["chrom"].astype(str)) != {chrom}:
        raise SystemExit(f"Chromosome mismatch: {metrics_path}")

    numeric = metrics[
        [
            "multiclass_ece",
            "multiclass_nll",
            "acceptor_ece",
            "donor_ece",
            "total_positions",
            "nonsplice_count",
            "acceptor_count",
            "donor_count",
        ]
    ].apply(pd.to_numeric, errors="coerce")

    if not np.isfinite(numeric.to_numpy()).all():
        raise SystemExit(f"Non-finite metrics: {metrics_path}")

    for column in [
        "total_positions",
        "nonsplice_count",
        "acceptor_count",
        "donor_count",
    ]:
        if metrics[column].nunique() != 1:
            raise SystemExit(
                f"{column} differs between methods: {metrics_path}"
            )

    first = metrics.iloc[0]
    class_total = (
        int(first["nonsplice_count"])
        + int(first["acceptor_count"])
        + int(first["donor_count"])
    )
    if class_total != int(first["total_positions"]):
        raise SystemExit(f"Class-count mismatch: {metrics_path}")

    if set(bins["method"]) != set(METHODS):
        raise SystemExit(f"Invalid bin method set: {bins_path}")

    bin_totals = bins.groupby("method")["count"].sum()
    if not (
        bin_totals == int(first["total_positions"])
    ).all():
        raise SystemExit(f"Bin-count mismatch: {bins_path}")

    bins.insert(1, "chrom", chrom)
    metric_frames.append(metrics)
    bin_frames.append(bins)

by_chrom = pd.concat(metric_frames, ignore_index=True)
all_bins = pd.concat(bin_frames, ignore_index=True)

# Verify that each reported chromosome ECE can be reconstructed.
check_rows = []

for (chrom, method), frame in all_bins.groupby(
    ["chrom", "method"]
):
    reconstructed = (
        frame["count"]
        * np.abs(
            frame["observed_accuracy"]
            - frame["mean_confidence"]
        )
    ).sum() / frame["count"].sum()

    reported = by_chrom.loc[
        (by_chrom["chrom"] == chrom)
        & (by_chrom["method"] == method),
        "multiclass_ece",
    ].iloc[0]

    check_rows.append(
        {
            "chrom": chrom,
            "method": method,
            "reported_multiclass_ece": reported,
            "reconstructed_multiclass_ece": reconstructed,
            "difference": reconstructed - reported,
            "absolute_difference": abs(reconstructed - reported),
        }
    )

checks = pd.DataFrame(check_rows)

if checks["absolute_difference"].max() > 1e-10:
    raise SystemExit(
        "ECE reconstruction failed:\n"
        + checks.sort_values(
            "absolute_difference",
            ascending=False,
        ).head().to_string(index=False)
    )

# Pool the sufficient statistics inside each reliability bin.
all_bins["confidence_sum"] = (
    all_bins["count"] * all_bins["mean_confidence"]
)
all_bins["correct_sum"] = (
    all_bins["count"] * all_bins["observed_accuracy"]
)

pooled_bins = (
    all_bins.groupby(
        ["method", "bin", "bin_low", "bin_high"],
        as_index=False,
    )
    .agg(
        count=("count", "sum"),
        confidence_sum=("confidence_sum", "sum"),
        correct_sum=("correct_sum", "sum"),
    )
)

pooled_bins["mean_confidence"] = (
    pooled_bins["confidence_sum"] / pooled_bins["count"]
)
pooled_bins["observed_accuracy"] = (
    pooled_bins["correct_sum"] / pooled_bins["count"]
)
pooled_bins["gap"] = np.abs(
    pooled_bins["observed_accuracy"]
    - pooled_bins["mean_confidence"]
)

summary_rows = []

for method in METHODS:
    metrics = by_chrom[by_chrom["method"] == method]
    bins = pooled_bins[pooled_bins["method"] == method]

    total = int(metrics["total_positions"].sum())
    if total != EXPECTED_POSITIONS:
        raise SystemExit(
            f"{method}: expected {EXPECTED_POSITIONS} positions, "
            f"found {total}"
        )

    pooled_ece = (
        bins["count"] * bins["gap"]
    ).sum() / bins["count"].sum()

    pooled_nll = (
        metrics["total_positions"]
        * metrics["multiclass_nll"]
    ).sum() / total

    weighted_chromosome_ece = (
        metrics["total_positions"]
        * metrics["multiclass_ece"]
    ).sum() / total

    summary_rows.append(
        {
            "method": method,
            "total_positions": total,
            "nonsplice_count": int(
                metrics["nonsplice_count"].sum()
            ),
            "acceptor_count": int(
                metrics["acceptor_count"].sum()
            ),
            "donor_count": int(
                metrics["donor_count"].sum()
            ),
            "pooled_multiclass_ece": pooled_ece,
            "pooled_multiclass_nll": pooled_nll,
            "position_weighted_mean_chromosome_ece": (
                weighted_chromosome_ece
            ),
            "min_chromosome_ece": (
                metrics["multiclass_ece"].min()
            ),
            "max_chromosome_ece": (
                metrics["multiclass_ece"].max()
            ),
            "min_chromosome_nll": (
                metrics["multiclass_nll"].min()
            ),
            "max_chromosome_nll": (
                metrics["multiclass_nll"].max()
            ),
        }
    )

summary = pd.DataFrame(summary_rows)

baseline = summary.loc[
    summary["method"] == "uncalibrated"
].iloc[0]

relative = summary[
    [
        "method",
        "total_positions",
        "pooled_multiclass_ece",
        "pooled_multiclass_nll",
    ]
].copy()

for metric in [
    "pooled_multiclass_ece",
    "pooled_multiclass_nll",
]:
    relative[f"{metric}_change_vs_uncalibrated"] = (
        relative[metric] - baseline[metric]
    )
    relative[f"{metric}_percent_reduction_vs_uncalibrated"] = (
        100
        * (baseline[metric] - relative[metric])
        / baseline[metric]
    )

by_chrom.to_csv(f"{prefix}_by_chromosome.csv", index=False)
checks.to_csv(
    f"{prefix}_ece_reconstruction_check.csv",
    index=False,
)
pooled_bins.to_csv(
    f"{prefix}_pooled_reliability_bins.csv",
    index=False,
)
summary.to_csv(
    f"{prefix}_summary_with_pooled_ece.csv",
    index=False,
)
relative.to_csv(
    f"{prefix}_relative_changes.csv",
    index=False,
)

weighted = summary.loc[
    summary["method"] == "genome_weighted_vectorT"
].iloc[0]
osai = summary.loc[
    summary["method"] == "openspliceai_style_vectorT"
].iloc[0]

print(
    "PASS: reconstructed all per-chromosome ECEs; "
    f"maximum error={checks['absolute_difference'].max():.3g}"
)
print(f"PASS: pooled {EXPECTED_POSITIONS:,} positions\n")

print(
    summary[
        [
            "method",
            "total_positions",
            "pooled_multiclass_ece",
            "pooled_multiclass_nll",
            "min_chromosome_ece",
            "max_chromosome_ece",
        ]
    ].to_string(index=False)
)

print("\nOSAI minus genome-weighted")
print(
    "ECE difference:",
    osai["pooled_multiclass_ece"]
    - weighted["pooled_multiclass_ece"],
)
print(
    "NLL difference:",
    osai["pooled_multiclass_nll"]
    - weighted["pooled_multiclass_nll"],
)

print("\nWrote prefix:", prefix)
