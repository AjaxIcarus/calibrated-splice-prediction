#!/usr/bin/env python3
"""Build matched schema-v2 cross-seed robustness tables for flank-400.

This script is intentionally read-only with respect to its two input directories.
It refuses to reuse an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = [
    "uncalibrated",
    "global_T_1.1",
    "true_logit_unweighted_vector_T",
    "true_logit_target_weighted_vector_T",
    "openspliceai_style_full_validation_vector_T_v2",
]

METHOD_LABELS = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Global T=1.1",
    "true_logit_unweighted_vector_T": "Unweighted true-logit vector-T",
    "true_logit_target_weighted_vector_T": "Target-weighted true-logit vector-T",
    "openspliceai_style_full_validation_vector_T_v2": (
        "OpenSpliceAI-style full-validation vector-T"
    ),
}

CORE_METRICS = [
    "weighted_multiclass_ece",
    "weighted_multiclass_nll",
    "weighted_multiclass_brier",
    "target_weighted_acceptor_auprc",
    "target_weighted_donor_auprc",
]

METRIC_LABELS = {
    "weighted_multiclass_ece": "Weighted multiclass ECE",
    "weighted_multiclass_nll": "Weighted multiclass NLL",
    "weighted_multiclass_brier": "Weighted multiclass Brier",
    "target_weighted_acceptor_auprc": "Target-weighted acceptor AUPRC",
    "target_weighted_donor_auprc": "Target-weighted donor AUPRC",
}

DIRECTIONS = {
    "weighted_multiclass_ece": "lower_is_better",
    "weighted_multiclass_nll": "lower_is_better",
    "weighted_multiclass_brier": "lower_is_better",
    "target_weighted_acceptor_auprc": "higher_is_better",
    "target_weighted_donor_auprc": "higher_is_better",
}

EXPECTED_CENSUS = {
    "cache_schema_version": 2,
    "all_sequences_included": True,
    "padding_excluded_from_sampling": True,
    "total_sequences_seen": 84258,
    "total_positions_seen": 421290000,
    "valid_positions_seen": 407278622,
    "padding_positions_seen": 14011378,
    "positive_positions_seen": 107988,
    "valid_nonsplice_positions_seen": 407170634,
    "sampled_negatives": 500000,
    "cache_random_seed": 11,
}

EXPECTED_ANALYSIS = {
    "n_bootstrap": 200,
    "bootstrap_seed": 17,
    "bootstrap_unit": "position",
    "resampling": "stratified_by_true_class",
    "paired_across_methods": True,
    "n_bins": 15,
    "global_temperature": 1.1,
    "method_order": METHODS,
}

REQUIRED_FILES = [
    "point_estimates.csv",
    "bootstrap_summary.csv",
    "paired_difference_summary.csv",
    "bootstrap_metadata.json",
]

TARGET_WEIGHTED = "true_logit_target_weighted_vector_T"
OPEN_SPLICE_STYLE = "openspliceai_style_full_validation_vector_T_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--replication-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-primary-point-sha256",
        default="a35b673ba71b60e550ca288ca915740b1591854553b777006902c76f54649882",
    )
    parser.add_argument(
        "--expected-replication-point-sha256",
        default="f2cb2e59ae9a514e9ebec8c4aa2f6f975bc4e5679d65f7464e12e33e6f7043cc",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"ERROR: missing columns in {path}: {missing}")


def validate_run(
    *,
    role: str,
    model: str,
    seed: int,
    epoch: int,
    directory: Path,
    expected_point_sha256: str,
    reference_point_columns: list[str] | None,
) -> tuple[dict[str, object], list[str]]:
    for filename in REQUIRED_FILES:
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"ERROR: missing or empty input: {path}")

    paths = {filename: directory / filename for filename in REQUIRED_FILES}
    point = pd.read_csv(paths["point_estimates.csv"])
    summary = pd.read_csv(paths["bootstrap_summary.csv"])
    paired = pd.read_csv(paths["paired_difference_summary.csv"])
    metadata = json.loads(paths["bootstrap_metadata.json"].read_text(encoding="utf-8"))

    point_digest = sha256(paths["point_estimates.csv"])
    if point_digest != expected_point_sha256:
        raise SystemExit(
            f"ERROR: point-estimate SHA-256 mismatch for {model}\n"
            f"expected: {expected_point_sha256}\nactual:   {point_digest}"
        )

    require_columns(point, ["method", *CORE_METRICS], paths["point_estimates.csv"])
    if point["method"].tolist() != METHODS:
        raise SystemExit(f"ERROR: method order mismatch in {paths['point_estimates.csv']}")

    point_columns = point.columns.tolist()
    if reference_point_columns is not None and point_columns != reference_point_columns:
        raise SystemExit("ERROR: seed-11 and seed-23 point-estimate schemas differ")

    metric_columns = [column for column in point_columns if column != "method"]
    if len(metric_columns) != 13:
        raise SystemExit(
            f"ERROR: expected 13 point-estimate metrics for {model}; "
            f"found {len(metric_columns)}"
        )
    if not np.isfinite(point[metric_columns].to_numpy(dtype=float)).all():
        raise SystemExit(f"ERROR: non-finite point estimate for {model}")

    require_columns(summary, ["method", "metric", "n_bootstrap"], paths["bootstrap_summary.csv"])
    if len(summary) != len(METHODS) * len(metric_columns):
        raise SystemExit(f"ERROR: invalid bootstrap-summary row count for {model}")
    if set(summary["method"]) != set(METHODS):
        raise SystemExit(f"ERROR: invalid bootstrap-summary methods for {model}")
    if set(summary["metric"]) != set(metric_columns):
        raise SystemExit(f"ERROR: invalid bootstrap-summary metrics for {model}")
    if set(summary["n_bootstrap"]) != {200}:
        raise SystemExit(f"ERROR: bootstrap count is not 200 for {model}")

    paired_columns = [
        "reference_method",
        "comparison_method",
        "metric",
        "direction",
        "mean_method_minus_reference",
        "mean_improvement",
        "improvement_ci_lower_2.5",
        "improvement_ci_upper_97.5",
        "probability_comparison_better",
        "improvement_ci_excludes_zero",
        "n_bootstrap",
    ]
    require_columns(paired, paired_columns, paths["paired_difference_summary.csv"])
    if len(paired) != 10 * len(metric_columns):
        raise SystemExit(f"ERROR: invalid paired-summary row count for {model}")
    if set(paired["metric"]) != set(metric_columns):
        raise SystemExit(f"ERROR: invalid paired-summary metrics for {model}")
    if set(paired["n_bootstrap"]) != {200}:
        raise SystemExit(f"ERROR: paired bootstrap count is not 200 for {model}")
    pair_counts = paired.groupby(["reference_method", "comparison_method"]).size()
    if len(pair_counts) != 10 or not (pair_counts == len(metric_columns)).all():
        raise SystemExit(f"ERROR: expected all 10 method pairs for {model}")

    for key, expected in EXPECTED_CENSUS.items():
        if metadata.get(key) != expected:
            raise SystemExit(
                f"ERROR: {model} metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    analysis = metadata.get("analysis", {})
    for key, expected in EXPECTED_ANALYSIS.items():
        if analysis.get(key) != expected:
            raise SystemExit(
                f"ERROR: {model} analysis mismatch for {key}: "
                f"{analysis.get(key)!r} != {expected!r}"
            )

    for metric, expected_direction in DIRECTIONS.items():
        observed = set(paired.loc[paired["metric"] == metric, "direction"])
        if observed != {expected_direction}:
            raise SystemExit(
                f"ERROR: paired direction mismatch for {model}/{metric}: {observed}"
            )

    source_hashes = {filename: sha256(path) for filename, path in paths.items()}
    record = {
        "role": role,
        "model": model,
        "seed": seed,
        "epoch": epoch,
        "directory": directory,
        "point": point,
        "paired": paired,
        "metadata": metadata,
        "source_hashes": source_hashes,
    }
    return record, point_columns


def markdown_table(core: pd.DataFrame) -> str:
    by_model = core.set_index(["method", "model"])

    def cell(method: str, metric: str) -> str:
        primary = float(by_model.loc[(method, "seed11_epoch11"), metric])
        replication = float(by_model.loc[(method, "seed23_epoch14"), metric])
        if metric in CORE_METRICS[:3]:
            return f"{primary:.3e} / {replication:.3e}"
        return f"{primary:.6f} / {replication:.6f}"

    lines = [
        "# Flank-400 matched cross-seed robustness (schema v2)",
        "",
        "Values are seed-11 epoch-11 / seed-23 epoch-14.",
        "",
        "| Method | Weighted ECE | Weighted NLL | Weighted Brier | Acceptor AUPRC | Donor AUPRC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        lines.append(
            "| "
            + " | ".join(
                [METHOD_LABELS[method]]
                + [cell(method, metric) for metric in CORE_METRICS]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "ECE, NLL, and Brier are lower-is-better; AUPRC is higher-is-better.",
            "All values use the matched schema-v2 valid-position target population.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_outputs(runs: list[dict[str, object]], output_dir: Path) -> None:
    if output_dir.exists():
        raise SystemExit(f"ERROR: refusing to reuse output directory: {output_dir}")

    all_frames: list[pd.DataFrame] = []
    core_frames: list[pd.DataFrame] = []
    change_rows: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []

    for run in runs:
        point = run["point"].copy()
        for column, value in [
            ("epoch", run["epoch"]),
            ("seed", run["seed"]),
            ("model", run["model"]),
            ("model_role", run["role"]),
        ]:
            point.insert(0, column, value)
        all_frames.append(point)
        core_frames.append(point[["model_role", "model", "seed", "epoch", "method", *CORE_METRICS]])

        raw_point = run["point"].set_index("method")
        for method in METHODS[1:]:
            for metric in CORE_METRICS:
                value = float(raw_point.loc[method, metric])
                baseline = float(raw_point.loc["uncalibrated", metric])
                delta = value - baseline
                direction = DIRECTIONS[metric]
                improvement = -delta if direction == "lower_is_better" else delta
                relative_change = 100.0 * delta / baseline
                relative_improvement = (
                    -relative_change if direction == "lower_is_better" else relative_change
                )
                change_rows.append(
                    {
                        "model_role": run["role"],
                        "model": run["model"],
                        "seed": run["seed"],
                        "epoch": run["epoch"],
                        "method": method,
                        "metric": metric,
                        "direction": direction,
                        "uncalibrated_value": baseline,
                        "method_value": value,
                        "method_minus_uncalibrated": delta,
                        "improvement": improvement,
                        "relative_change_percent": relative_change,
                        "relative_improvement_percent": relative_improvement,
                    }
                )

        paired = run["paired"]
        selected = paired[
            (paired["reference_method"] == TARGET_WEIGHTED)
            & (paired["comparison_method"] == OPEN_SPLICE_STYLE)
            & (paired["metric"].isin(CORE_METRICS))
        ].copy()
        if selected["metric"].tolist() != CORE_METRICS:
            selected = selected.set_index("metric").reindex(CORE_METRICS).reset_index()
        if selected["metric"].tolist() != CORE_METRICS or selected.isna().any().any():
            raise SystemExit(
                f"ERROR: missing target-weighted/OpenSpliceAI-style comparison for {run['model']}"
            )
        selected.insert(0, "epoch", run["epoch"])
        selected.insert(0, "seed", run["seed"])
        selected.insert(0, "model", run["model"])
        selected.insert(0, "model_role", run["role"])
        paired_frames.append(selected)

    all_points = pd.concat(all_frames, ignore_index=True)
    core = pd.concat(core_frames, ignore_index=True)
    changes = pd.DataFrame(change_rows)
    prior_paired = pd.concat(paired_frames, ignore_index=True)

    descriptive_rows: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = core[core["method"] == method]
        for metric in CORE_METRICS:
            values = method_rows[metric].to_numpy(dtype=float)
            descriptive_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "direction": DIRECTIONS[metric],
                    "n_models": len(values),
                    "descriptive_mean": values.mean(),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "range": values.max() - values.min(),
                }
            )
    descriptive = pd.DataFrame(descriptive_rows)

    output_dir.mkdir(parents=True)
    outputs = {
        "cross_seed_point_estimates_all_metrics.csv": all_points,
        "cross_seed_core_point_estimates.csv": core,
        "cross_seed_change_vs_uncalibrated.csv": changes,
        "cross_seed_descriptive_summary.csv": descriptive,
        "cross_seed_prior_aware_paired_comparison.csv": prior_paired,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)

    table_text = markdown_table(core)
    (output_dir / "cross_seed_compact_table.md").write_text(table_text, encoding="utf-8")

    proper = changes[changes["metric"].isin(CORE_METRICS[:3])]
    summary_lines = [
        "# Flank-400 cross-seed robustness interpretation",
        "",
        "## Proper scoring rules versus uncalibrated",
        "",
    ]
    for method in METHODS[1:]:
        summary_lines.append(f"### {METHOD_LABELS[method]}")
        subset = proper[proper["method"] == method]
        for metric in CORE_METRICS[:3]:
            metric_rows = subset[subset["metric"] == metric].set_index("model")
            first = metric_rows.loc["seed11_epoch11", "relative_improvement_percent"]
            second = metric_rows.loc["seed23_epoch14", "relative_improvement_percent"]
            summary_lines.append(
                f"- {METRIC_LABELS[metric]}: {first:+.2f}% / {second:+.2f}% improvement."
            )
        summary_lines.append("")

    excludes = prior_paired["improvement_ci_excludes_zero"].map(as_bool)
    summary_lines.extend(
        [
            "## Prior-aware paired comparison",
            "",
            (
                f"- {int(excludes.sum())} of {len(excludes)} seed-by-metric paired 95% "
                "bootstrap intervals exclude zero."
            ),
            "- Treat seed-level consistency as descriptive robustness: two trained seeds do not support a population-level variance estimate.",
            "- The bootstrap unit is position and intervals quantify evaluated-position uncertainty, not training-seed, gene, or chromosome uncertainty.",
            "- Target-weighted AUPRC values must not be mixed with legacy sample-based AUPRC values.",
            "",
        ]
    )
    (output_dir / "cross_seed_robustness_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    non_metadata_outputs = sorted(path.name for path in output_dir.iterdir())
    output_hashes = {
        filename: sha256(output_dir / filename) for filename in non_metadata_outputs
    }
    provenance = {
        "artifact_schema": "flank400_cross_seed_robustness_schema_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": [
            {
                "model_role": run["role"],
                "model": run["model"],
                "seed": run["seed"],
                "epoch": run["epoch"],
                "source_directory": str(run["directory"]),
                "source_sha256": run["source_hashes"],
                "cache_sha256": run["metadata"].get("cache_sha256"),
            }
            for run in runs
        ],
        "population": EXPECTED_CENSUS,
        "analysis": EXPECTED_ANALYSIS,
        "core_metrics": CORE_METRICS,
        "metric_directions": DIRECTIONS,
        "outputs_sha256": output_hashes,
        "interpretation_limits": [
            "Two seeds provide descriptive model-level robustness only.",
            "Position-level bootstrap intervals do not estimate training-seed variability.",
            "Target-weighted schema-v2 AUPRC is not interchangeable with legacy sample-based AUPRC.",
        ],
    }
    (output_dir / "cross_seed_metadata.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    expected_files = set(non_metadata_outputs) | {"cross_seed_metadata.json"}
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise SystemExit(
            f"ERROR: output inventory mismatch: expected {expected_files}, got {actual_files}"
        )
    if len(all_points) != 10 or len(core) != 10:
        raise SystemExit("ERROR: expected 10 cross-seed point-estimate rows")
    if len(changes) != 40:
        raise SystemExit("ERROR: expected 40 method-versus-uncalibrated rows")
    if len(descriptive) != 25:
        raise SystemExit("ERROR: expected 25 descriptive-summary rows")
    if len(prior_paired) != 10:
        raise SystemExit("ERROR: expected 10 prior-aware paired-comparison rows")

    print("PASS: exact authoritative point-estimate hashes validated")
    print("PASS: matched schema-v2 populations and bootstrap designs validated")
    print("PASS: 10 point-estimate rows and 40 baseline-comparison rows written")
    print("PASS: 10 prior-aware paired comparisons preserved with intervals")
    print("PASS: legacy files were not modified")
    print()
    print("==== COMPACT CROSS-SEED TABLE ====")
    print(table_text.rstrip())
    print()
    print("==== INTERPRETATION SUMMARY ====")
    print((output_dir / "cross_seed_robustness_summary.md").read_text(encoding="utf-8").rstrip())
    print()
    print("==== OUTPUTS ====")
    for path in sorted(output_dir.iterdir()):
        print(path)


def main() -> None:
    args = parse_args()
    run_specs = [
        {
            "role": "primary",
            "model": "seed11_epoch11",
            "seed": 11,
            "epoch": 11,
            "directory": args.primary_dir,
            "expected_point_sha256": args.expected_primary_point_sha256,
        },
        {
            "role": "replication",
            "model": "seed23_epoch14",
            "seed": 23,
            "epoch": 14,
            "directory": args.replication_dir,
            "expected_point_sha256": args.expected_replication_point_sha256,
        },
    ]

    runs: list[dict[str, object]] = []
    reference_columns: list[str] | None = None
    for spec in run_specs:
        run, reference_columns = validate_run(
            **spec,
            reference_point_columns=reference_columns,
        )
        runs.append(run)

    build_outputs(runs, args.output_dir)


if __name__ == "__main__":
    main()
