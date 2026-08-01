#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = [
    "uncalibrated",
    "fixed_global_T1p1",
    "unweighted_vectorT",
    "genome_weighted_vectorT",
    "openspliceai_style_vectorT",
]
CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
EXPECTED_POSITIONS = 416_140_000

RUNS = [
    {
        "model_role": "primary",
        "model": "seed11_epoch11",
        "seed": 11,
        "epoch": 11,
        "prefix": Path(
            "results/"
            "flank400_chromosome_transfer_seed11_epoch11_recomputed"
        ),
    },
    {
        "model_role": "replication",
        "model": "seed23_epoch14",
        "seed": 23,
        "epoch": 14,
        "prefix": Path(
            "results/"
            "flank400_chromosome_transfer_seed23_epoch14"
        ),
    },
]

robustness_frames = []
by_chromosome = {}

for run in RUNS:
    summary_path = Path(
        f"{run['prefix']}_summary_with_pooled_ece.csv"
    )
    chromosome_path = Path(
        f"{run['prefix']}_by_chromosome.csv"
    )

    for path in [summary_path, chromosome_path]:
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")

    summary = pd.read_csv(summary_path)
    chrom = pd.read_csv(chromosome_path)

    required_summary = {
        "method",
        "total_positions",
        "pooled_multiclass_ece",
        "pooled_multiclass_nll",
    }
    required_chrom = {
        "method",
        "chrom",
        "total_positions",
        "multiclass_ece",
        "multiclass_nll",
    }

    if not required_summary.issubset(summary.columns):
        raise SystemExit(
            f"Invalid summary schema: {summary_path}"
        )
    if not required_chrom.issubset(chrom.columns):
        raise SystemExit(
            f"Invalid chromosome schema: {chromosome_path}"
        )

    if set(summary["method"]) != set(METHODS):
        raise SystemExit(
            f"Unexpected summary methods: {summary_path}"
        )
    if set(chrom["method"]) != set(METHODS):
        raise SystemExit(
            f"Unexpected chromosome methods: {chromosome_path}"
        )
    if set(chrom["chrom"].astype(str)) != set(CHROMS):
        raise SystemExit(
            f"Unexpected chromosomes: {chromosome_path}"
        )
    if len(chrom) != len(METHODS) * len(CHROMS):
        raise SystemExit(
            f"Expected 25 chromosome rows: {chromosome_path}"
        )

    numeric_summary = summary[
        [
            "total_positions",
            "pooled_multiclass_ece",
            "pooled_multiclass_nll",
        ]
    ].apply(pd.to_numeric, errors="coerce")

    numeric_chrom = chrom[
        [
            "total_positions",
            "multiclass_ece",
            "multiclass_nll",
        ]
    ].apply(pd.to_numeric, errors="coerce")

    if not np.isfinite(numeric_summary.to_numpy()).all():
        raise SystemExit(f"Non-finite summary: {summary_path}")
    if not np.isfinite(numeric_chrom.to_numpy()).all():
        raise SystemExit(
            f"Non-finite chromosome metrics: {chromosome_path}"
        )

    if not (
        summary["total_positions"].astype(int)
        == EXPECTED_POSITIONS
    ).all():
        raise SystemExit(
            f"Unexpected pooled position count: {summary_path}"
        )

    # Independently verify pooled NLL from chromosome results.
    for method in METHODS:
        rows = chrom[chrom["method"] == method]
        total = int(rows["total_positions"].sum())

        if total != EXPECTED_POSITIONS:
            raise SystemExit(
                f"{run['model']}/{method}: "
                f"unexpected total {total}"
            )

        reconstructed_nll = (
            rows["total_positions"]
            * rows["multiclass_nll"]
        ).sum() / total

        reported_nll = summary.loc[
            summary["method"] == method,
            "pooled_multiclass_nll",
        ].iloc[0]

        if not np.isclose(
            reconstructed_nll,
            reported_nll,
            rtol=0,
            atol=1e-12,
        ):
            raise SystemExit(
                f"{run['model']}/{method}: "
                "pooled NLL reconstruction failed"
            )

    frame = summary[
        [
            "method",
            "total_positions",
            "pooled_multiclass_ece",
            "pooled_multiclass_nll",
        ]
    ].copy()

    frame.insert(0, "epoch", run["epoch"])
    frame.insert(0, "seed", run["seed"])
    frame.insert(0, "model", run["model"])
    frame.insert(0, "model_role", run["model_role"])

    baseline = frame.loc[
        frame["method"] == "uncalibrated"
    ].iloc[0]

    frame["ece_reduction_percent_vs_uncalibrated"] = (
        100
        * (
            baseline["pooled_multiclass_ece"]
            - frame["pooled_multiclass_ece"]
        )
        / baseline["pooled_multiclass_ece"]
    )
    frame["nll_reduction_percent_vs_uncalibrated"] = (
        100
        * (
            baseline["pooled_multiclass_nll"]
            - frame["pooled_multiclass_nll"]
        )
        / baseline["pooled_multiclass_nll"]
    )
    frame["ece_rank_within_model"] = (
        frame["pooled_multiclass_ece"]
        .rank(method="min")
        .astype(int)
    )
    frame["nll_rank_within_model"] = (
        frame["pooled_multiclass_nll"]
        .rank(method="min")
        .astype(int)
    )

    robustness_frames.append(frame)
    by_chromosome[run["model_role"]] = chrom.copy()

robustness = pd.concat(robustness_frames, ignore_index=True)

primary = robustness[
    robustness["model_role"] == "primary"
].copy()
replication = robustness[
    robustness["model_role"] == "replication"
].copy()

pooled_comparison = primary[
    [
        "method",
        "pooled_multiclass_ece",
        "pooled_multiclass_nll",
    ]
].merge(
    replication[
        [
            "method",
            "pooled_multiclass_ece",
            "pooled_multiclass_nll",
        ]
    ],
    on="method",
    suffixes=("_primary", "_replication"),
    validate="one_to_one",
)

for metric in [
    "pooled_multiclass_ece",
    "pooled_multiclass_nll",
]:
    pooled_comparison[
        f"{metric}_replication_minus_primary"
    ] = (
        pooled_comparison[f"{metric}_replication"]
        - pooled_comparison[f"{metric}_primary"]
    )
    pooled_comparison[
        f"{metric}_percent_change_replication_vs_primary"
    ] = (
        100
        * pooled_comparison[
            f"{metric}_replication_minus_primary"
        ]
        / pooled_comparison[f"{metric}_primary"]
    )

# Pair the same chromosome across the two seeds.
paired = by_chromosome["primary"][
    [
        "method",
        "chrom",
        "total_positions",
        "multiclass_ece",
        "multiclass_nll",
    ]
].merge(
    by_chromosome["replication"][
        [
            "method",
            "chrom",
            "total_positions",
            "multiclass_ece",
            "multiclass_nll",
        ]
    ],
    on=["method", "chrom"],
    suffixes=("_primary", "_replication"),
    validate="one_to_one",
)

if not (
    paired["total_positions_primary"]
    == paired["total_positions_replication"]
).all():
    raise SystemExit(
        "Primary and replication chromosome sizes differ"
    )

paired["ece_replication_minus_primary"] = (
    paired["multiclass_ece_replication"]
    - paired["multiclass_ece_primary"]
)
paired["nll_replication_minus_primary"] = (
    paired["multiclass_nll_replication"]
    - paired["multiclass_nll_primary"]
)

paired_summary = (
    paired.groupby("method", as_index=False)
    .agg(
        n_chromosomes=("chrom", "nunique"),
        mean_ece_difference=(
            "ece_replication_minus_primary",
            "mean",
        ),
        sd_ece_difference=(
            "ece_replication_minus_primary",
            "std",
        ),
        min_ece_difference=(
            "ece_replication_minus_primary",
            "min",
        ),
        max_ece_difference=(
            "ece_replication_minus_primary",
            "max",
        ),
        replication_lower_ece_count=(
            "ece_replication_minus_primary",
            lambda values: int((values < 0).sum()),
        ),
        mean_nll_difference=(
            "nll_replication_minus_primary",
            "mean",
        ),
        sd_nll_difference=(
            "nll_replication_minus_primary",
            "std",
        ),
        min_nll_difference=(
            "nll_replication_minus_primary",
            "min",
        ),
        max_nll_difference=(
            "nll_replication_minus_primary",
            "max",
        ),
        replication_lower_nll_count=(
            "nll_replication_minus_primary",
            lambda values: int((values < 0).sum()),
        ),
    )
)

# Key OSAI-versus-genome-weighted contrast for each seed.
contrast_rows = []

for run in RUNS:
    frame = robustness[
        robustness["model"] == run["model"]
    ]
    weighted = frame.loc[
        frame["method"] == "genome_weighted_vectorT"
    ].iloc[0]
    osai = frame.loc[
        frame["method"] == "openspliceai_style_vectorT"
    ].iloc[0]

    contrast_rows.append(
        {
            "model_role": run["model_role"],
            "model": run["model"],
            "seed": run["seed"],
            "epoch": run["epoch"],
            "ece_osai_minus_genome_weighted": (
                osai["pooled_multiclass_ece"]
                - weighted["pooled_multiclass_ece"]
            ),
            "nll_osai_minus_genome_weighted": (
                osai["pooled_multiclass_nll"]
                - weighted["pooled_multiclass_nll"]
            ),
            "ece_reduction_percent_osai_vs_genome_weighted": (
                100
                * (
                    weighted["pooled_multiclass_ece"]
                    - osai["pooled_multiclass_ece"]
                )
                / weighted["pooled_multiclass_ece"]
            ),
            "nll_reduction_percent_osai_vs_genome_weighted": (
                100
                * (
                    weighted["pooled_multiclass_nll"]
                    - osai["pooled_multiclass_nll"]
                )
                / weighted["pooled_multiclass_nll"]
            ),
        }
    )

key_contrasts = pd.DataFrame(contrast_rows)

# Expected qualitative pattern must hold independently in both seeds.
for model, frame in robustness.groupby("model"):
    values = frame.set_index("method")

    checks = [
        values.loc[
            "openspliceai_style_vectorT",
            "pooled_multiclass_ece",
        ]
        < values.loc[
            "genome_weighted_vectorT",
            "pooled_multiclass_ece",
        ],
        values.loc[
            "openspliceai_style_vectorT",
            "pooled_multiclass_nll",
        ]
        < values.loc[
            "genome_weighted_vectorT",
            "pooled_multiclass_nll",
        ],
        values.loc[
            "genome_weighted_vectorT",
            "pooled_multiclass_ece",
        ]
        < values.loc[
            "uncalibrated",
            "pooled_multiclass_ece",
        ],
        values.loc[
            "genome_weighted_vectorT",
            "pooled_multiclass_nll",
        ]
        < values.loc[
            "uncalibrated",
            "pooled_multiclass_nll",
        ],
        values.loc[
            "fixed_global_T1p1",
            "pooled_multiclass_ece",
        ]
        > values.loc[
            "uncalibrated",
            "pooled_multiclass_ece",
        ],
        values.loc[
            "fixed_global_T1p1",
            "pooled_multiclass_nll",
        ]
        > values.loc[
            "uncalibrated",
            "pooled_multiclass_nll",
        ],
        values.loc[
            "unweighted_vectorT",
            "pooled_multiclass_ece",
        ]
        < values.loc[
            "uncalibrated",
            "pooled_multiclass_ece",
        ],
        values.loc[
            "unweighted_vectorT",
            "pooled_multiclass_nll",
        ]
        > values.loc[
            "uncalibrated",
            "pooled_multiclass_nll",
        ],
    ]

    if not all(checks):
        raise SystemExit(
            f"{model}: expected qualitative pattern failed"
        )

output_prefix = Path(
    "results/flank400_chromosome_transfer_cross_seed"
)

robustness.to_csv(
    f"{output_prefix}_method_robustness.csv",
    index=False,
)
pooled_comparison.to_csv(
    f"{output_prefix}_pooled_comparison.csv",
    index=False,
)
paired.to_csv(
    f"{output_prefix}_paired_by_chromosome.csv",
    index=False,
)
paired_summary.to_csv(
    f"{output_prefix}_paired_summary.csv",
    index=False,
)
key_contrasts.to_csv(
    f"{output_prefix}_key_contrasts.csv",
    index=False,
)

print(
    "PASS: matched seed-11 and seed-23 chromosome-transfer "
    "comparison validated"
)
print(
    f"PASS: each model covers {EXPECTED_POSITIONS:,} positions"
)

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    220,
    "display.precision",
    12,
):
    print("\nPOOLED CROSS-SEED ROBUSTNESS")
    print(
        robustness[
            [
                "model_role",
                "model",
                "method",
                "pooled_multiclass_ece",
                "pooled_multiclass_nll",
                "ece_reduction_percent_vs_uncalibrated",
                "nll_reduction_percent_vs_uncalibrated",
            ]
        ].to_string(index=False)
    )

    print("\nSEED-23 MINUS SEED-11")
    print(pooled_comparison.to_string(index=False))

    print("\nOSAI VERSUS GENOME-WEIGHTED")
    print(key_contrasts.to_string(index=False))

    print("\nPAIRED CHROMOSOME SUMMARY")
    print(paired_summary.to_string(index=False))

print("\nWrote:")
for suffix in [
    "method_robustness",
    "pooled_comparison",
    "paired_by_chromosome",
    "paired_summary",
    "key_contrasts",
]:
    print(f"  {output_prefix}_{suffix}.csv")
