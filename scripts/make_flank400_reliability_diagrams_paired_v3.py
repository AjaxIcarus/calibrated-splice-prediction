#!/usr/bin/env python3
"""Generate provenance-checked schema-v2 reliability diagrams.

This plotter is designed for the corrected flank-400 calibration comparison.
It reconstructs the valid-position target population from a cache containing
all positive positions and a sample of valid non-splice positions. Padding is
never assigned weight because schema-v2 caches exclude it from sampling.

All five calibration methods use the same float64 softmax and the same 15-bin
convention as ``bootstrap_logit_metrics_paired_v2.py``. Before writing output,
the script checks the computed multiclass, acceptor, and donor ECE values
against the authoritative paired-bootstrap point-estimate table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "calibrated-splice-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "uncalibrated",
    "global_T_1.1",
    "true_logit_unweighted_vector_T",
    "true_logit_target_weighted_vector_T",
    "openspliceai_style_full_validation_vector_T_v2",
]

METHOD_LABELS = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Global T=1.1",
    "true_logit_unweighted_vector_T": "Unweighted vector-T",
    "true_logit_target_weighted_vector_T": "Target-weighted vector-T",
    "openspliceai_style_full_validation_vector_T_v2": (
        "OpenSpliceAI-style vector-T"
    ),
}

METHOD_COLORS = {
    "uncalibrated": "#4C78A8",
    "global_T_1.1": "#F58518",
    "true_logit_unweighted_vector_T": "#E45756",
    "true_logit_target_weighted_vector_T": "#54A24B",
    "openspliceai_style_full_validation_vector_T_v2": "#B279A2",
}

RELIABILITY_SPECS = [
    ("multiclass", "argmax", "Multiclass confidence", "Accuracy"),
    ("binary_class", "acceptor", "Acceptor probability", "Observed frequency"),
    ("binary_class", "donor", "Donor probability", "Observed frequency"),
]

REQUIRED_CACHE_FIELDS = [
    "logits_sample",
    "labels_sample",
    "cache_schema_version",
    "all_sequences_included",
    "padding_excluded_from_sampling",
    "total_sequences_seen",
    "total_positions_seen",
    "valid_positions_seen",
    "padding_positions_seen",
    "positive_positions_seen",
    "valid_nonsplice_positions_seen",
    "negatives_seen_total",
    "sampled_negatives",
    "valid_nonsplice_weight",
    "random_seed",
]

EXPECTED_ECE_COLUMNS = [
    "weighted_multiclass_ece",
    "acceptor_ece",
    "donor_ece",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create target-prior-weighted reliability diagrams for the five "
            "authoritative schema-v2 calibration methods."
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--unweighted-temperature-txt", required=True)
    parser.add_argument("--target-weighted-temperature-txt", required=True)
    parser.add_argument("--openspliceai-temperature-txt", required=True)
    parser.add_argument("--point-estimates-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-label",
        required=True,
        help="Human-readable model label used in figure titles.",
    )
    parser.add_argument(
        "--artifact-tag",
        required=True,
        help=(
            "Filesystem-safe tag used in figure filenames, for example "
            "flank400_seed23_epoch14_v2."
        ),
    )
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--global-temperature", type=float, default=1.1)
    parser.add_argument("--expected-cache-sha256")
    parser.add_argument("--expected-cache-random-seed", type=int)
    parser.add_argument("--expected-total-sequences", type=int)
    parser.add_argument("--expected-total-positions", type=int)
    parser.add_argument("--expected-valid-positions", type=int)
    parser.add_argument("--expected-padding-positions", type=int)
    parser.add_argument("--expected-positive-positions", type=int)
    parser.add_argument("--expected-valid-nonsplice-positions", type=int)
    parser.add_argument("--expected-sampled-negatives", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(cache: Mapping[str, np.ndarray], name: str):
    value = np.asarray(cache[name]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{name} must contain exactly one value; got {value.shape}")
    return value[0].item()


def require_equal(name: str, actual, expected) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {actual!r}")


def load_temperature(path: Path) -> np.ndarray:
    raw = path.read_text(encoding="utf-8").strip()
    cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    temperature = np.fromstring(cleaned, sep=" ", dtype=np.float64)
    if temperature.shape != (3,):
        raise ValueError(
            f"{path} must contain exactly three temperatures; got {temperature}"
        )
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError(f"{path} contains invalid temperatures: {temperature}")
    return temperature


def labels_to_indices(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    if labels.ndim == 2:
        if labels.shape != logits.shape:
            raise ValueError(
                f"one-hot labels shape {labels.shape} != logits shape {logits.shape}"
            )
        if not np.all(np.isfinite(labels)):
            raise ValueError("labels_sample contains NaN or infinite values")
        if not np.allclose(labels.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("labels_sample rows are not valid one-hot rows")
        y = labels.argmax(axis=1).astype(np.int64)
        if not np.all(labels[np.arange(len(y)), y] == 1):
            raise ValueError("labels_sample is not strictly one-hot encoded")
    elif labels.ndim == 1:
        y = labels.astype(np.int64)
    else:
        raise ValueError(f"labels_sample has unsupported shape {labels.shape}")

    if len(y) != len(logits):
        raise ValueError("logits_sample and labels_sample lengths differ")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("labels_sample contains a class outside {0, 1, 2}")
    return y


def validate_and_load_cache(
    cache_path: Path,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    cache_sha256 = sha256_file(cache_path)
    if args.expected_cache_sha256:
        require_equal(
            "cache_sha256",
            cache_sha256.lower(),
            args.expected_cache_sha256.lower(),
        )

    with np.load(cache_path, allow_pickle=False) as cache:
        missing = sorted(set(REQUIRED_CACHE_FIELDS) - set(cache.files))
        if missing:
            raise ValueError(f"cache is missing required fields: {missing}")

        cache_schema_version = int(scalar(cache, "cache_schema_version"))
        all_sequences_included = int(scalar(cache, "all_sequences_included"))
        padding_excluded = int(
            scalar(cache, "padding_excluded_from_sampling")
        )
        total_sequences = int(scalar(cache, "total_sequences_seen"))
        total_positions = int(scalar(cache, "total_positions_seen"))
        valid_positions = int(scalar(cache, "valid_positions_seen"))
        padding_positions = int(scalar(cache, "padding_positions_seen"))
        positive_positions = int(scalar(cache, "positive_positions_seen"))
        valid_nonsplice = int(
            scalar(cache, "valid_nonsplice_positions_seen")
        )
        negatives_seen_total = int(scalar(cache, "negatives_seen_total"))
        sampled_negatives = int(scalar(cache, "sampled_negatives"))
        stored_negative_weight = float(
            scalar(cache, "valid_nonsplice_weight")
        )
        cache_random_seed = int(scalar(cache, "random_seed"))

        logits = np.asarray(cache["logits_sample"], dtype=np.float64)
        labels = np.asarray(cache["labels_sample"])

    require_equal("cache_schema_version", cache_schema_version, 2)
    require_equal("all_sequences_included", all_sequences_included, 1)
    require_equal("padding_excluded_from_sampling", padding_excluded, 1)
    require_equal(
        "cache_random_seed",
        cache_random_seed,
        args.expected_cache_random_seed,
    )
    require_equal(
        "total_sequences_seen",
        total_sequences,
        args.expected_total_sequences,
    )
    require_equal(
        "total_positions_seen",
        total_positions,
        args.expected_total_positions,
    )
    require_equal(
        "valid_positions_seen",
        valid_positions,
        args.expected_valid_positions,
    )
    require_equal(
        "padding_positions_seen",
        padding_positions,
        args.expected_padding_positions,
    )
    require_equal(
        "positive_positions_seen",
        positive_positions,
        args.expected_positive_positions,
    )
    require_equal(
        "valid_nonsplice_positions_seen",
        valid_nonsplice,
        args.expected_valid_nonsplice_positions,
    )
    require_equal(
        "sampled_negatives",
        sampled_negatives,
        args.expected_sampled_negatives,
    )

    if total_positions != valid_positions + padding_positions:
        raise ValueError(
            "census failure: total_positions_seen != "
            "valid_positions_seen + padding_positions_seen"
        )
    if valid_positions != positive_positions + valid_nonsplice:
        raise ValueError(
            "census failure: valid_positions_seen != "
            "positive_positions_seen + valid_nonsplice_positions_seen"
        )
    if negatives_seen_total != valid_nonsplice:
        raise ValueError(
            "legacy/new metadata disagreement: negatives_seen_total != "
            "valid_nonsplice_positions_seen"
        )
    if sampled_negatives <= 0 or sampled_negatives > valid_nonsplice:
        raise ValueError("sampled_negatives is outside the valid range")

    reconstructed_negative_weight = valid_nonsplice / sampled_negatives
    if not np.isclose(
        stored_negative_weight,
        reconstructed_negative_weight,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "valid_nonsplice_weight does not match "
            "valid_nonsplice_positions_seen / sampled_negatives"
        )

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"logits_sample must have shape (N, 3); got {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits_sample contains NaN or infinite values")

    y = labels_to_indices(labels, logits)
    class_counts = np.bincount(y, minlength=3)
    if int(class_counts[0]) != sampled_negatives:
        raise ValueError(
            f"class-0 count {class_counts[0]} != sampled_negatives "
            f"{sampled_negatives}"
        )
    if int(class_counts[1] + class_counts[2]) != positive_positions:
        raise ValueError(
            "sampled positive count does not equal positive_positions_seen"
        )

    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = reconstructed_negative_weight

    metadata: Dict[str, object] = {
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "cache_schema_version": cache_schema_version,
        "all_sequences_included": bool(all_sequences_included),
        "padding_excluded_from_sampling": bool(padding_excluded),
        "total_sequences_seen": total_sequences,
        "total_positions_seen": total_positions,
        "valid_positions_seen": valid_positions,
        "padding_positions_seen": padding_positions,
        "positive_positions_seen": positive_positions,
        "valid_nonsplice_positions_seen": valid_nonsplice,
        "negatives_seen_total": negatives_seen_total,
        "sampled_negatives": sampled_negatives,
        "valid_nonsplice_weight": reconstructed_negative_weight,
        "cache_random_seed": cache_random_seed,
        "sample_class_counts": class_counts.astype(int).tolist(),
        "sample_size": int(len(y)),
        "reconstructed_target_weight_sum": float(weights.sum()),
    }
    return logits, y, weights, metadata


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def reliability_binary(
    probability: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    """Return equal-width target-weighted reliability-bin statistics."""

    if probability.ndim != 1 or observed.ndim != 1 or weights.ndim != 1:
        raise ValueError("reliability inputs must be one-dimensional")
    if not (len(probability) == len(observed) == len(weights)):
        raise ValueError("reliability input lengths differ")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probability is outside [0, 1]")

    bin_index = np.minimum(
        (probability * n_bins).astype(np.int64),
        n_bins - 1,
    )
    raw_count = np.bincount(bin_index, minlength=n_bins)
    target_weight = np.bincount(
        bin_index,
        weights=weights,
        minlength=n_bins,
    )
    weighted_probability = np.bincount(
        bin_index,
        weights=weights * probability,
        minlength=n_bins,
    )
    weighted_observed = np.bincount(
        bin_index,
        weights=weights * observed,
        minlength=n_bins,
    )

    nonempty = target_weight > 0.0
    mean_probability = np.full(n_bins, np.nan, dtype=np.float64)
    observed_frequency = np.full(n_bins, np.nan, dtype=np.float64)
    mean_probability[nonempty] = (
        weighted_probability[nonempty] / target_weight[nonempty]
    )
    observed_frequency[nonempty] = (
        weighted_observed[nonempty] / target_weight[nonempty]
    )
    absolute_gap = np.abs(mean_probability - observed_frequency)
    target_weight_fraction = target_weight / weights.sum()
    ece_contribution = np.zeros(n_bins, dtype=np.float64)
    ece_contribution[nonempty] = (
        target_weight_fraction[nonempty] * absolute_gap[nonempty]
    )

    bin_lower = np.arange(n_bins, dtype=np.float64) / n_bins
    bin_upper = np.arange(1, n_bins + 1, dtype=np.float64) / n_bins

    return pd.DataFrame(
        {
            "bin": np.arange(n_bins, dtype=np.int64),
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "raw_count": raw_count.astype(np.int64),
            "target_weight": target_weight,
            "target_weight_fraction": target_weight_fraction,
            "mean_predicted_probability": mean_probability,
            "observed_frequency": observed_frequency,
            "absolute_gap": absolute_gap,
            "ece_contribution": ece_contribution,
        }
    )


def compute_reliability_tables(
    probabilities: Mapping[str, np.ndarray],
    y: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_bins = []
    summary_rows = []

    for method in METHOD_ORDER:
        probs = probabilities[method]

        confidence = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y).astype(np.float64)
        multiclass = reliability_binary(
            confidence,
            correct,
            weights,
            n_bins,
        )
        multiclass.insert(0, "class", "argmax")
        multiclass.insert(0, "reliability_type", "multiclass")
        multiclass.insert(0, "method", method)
        all_bins.append(multiclass)
        summary_rows.append(
            {
                "method": method,
                "weighted_multiclass_ece": float(
                    multiclass["ece_contribution"].sum()
                ),
            }
        )

        for class_index, class_name in [(1, "acceptor"), (2, "donor")]:
            binary = reliability_binary(
                probs[:, class_index],
                (y == class_index).astype(np.float64),
                weights,
                n_bins,
            )
            binary.insert(0, "class", class_name)
            binary.insert(0, "reliability_type", "binary_class")
            binary.insert(0, "method", method)
            all_bins.append(binary)
            summary_rows[-1][f"{class_name}_ece"] = float(
                binary["ece_contribution"].sum()
            )

    bins = pd.concat(all_bins, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return bins, summary


def load_and_validate_point_estimates(
    point_estimates_path: Path,
    computed_summary: pd.DataFrame,
) -> pd.DataFrame:
    expected = pd.read_csv(point_estimates_path)
    required_columns = {"method", *EXPECTED_ECE_COLUMNS}
    missing_columns = sorted(required_columns - set(expected.columns))
    if missing_columns:
        raise ValueError(
            f"{point_estimates_path} is missing columns: {missing_columns}"
        )
    if expected["method"].duplicated().any():
        duplicates = expected.loc[
            expected["method"].duplicated(keep=False),
            "method",
        ].tolist()
        raise ValueError(f"point-estimate table has duplicate methods: {duplicates}")

    expected = expected.set_index("method")
    missing_methods = sorted(set(METHOD_ORDER) - set(expected.index))
    if missing_methods:
        raise ValueError(
            f"point-estimate table is missing methods: {missing_methods}"
        )

    actual = computed_summary.set_index("method")
    comparison_rows = []
    for method in METHOD_ORDER:
        row = {"method": method}
        for metric in EXPECTED_ECE_COLUMNS:
            expected_value = float(expected.loc[method, metric])
            actual_value = float(actual.loc[method, metric])
            difference = actual_value - expected_value
            if not np.isclose(
                actual_value,
                expected_value,
                rtol=1e-7,
                atol=5e-12,
            ):
                raise ValueError(
                    f"point-estimate mismatch for {method}/{metric}: "
                    f"expected={expected_value:.17g}, "
                    f"actual={actual_value:.17g}, "
                    f"difference={difference:+.17g}"
                )
            row[f"expected_{metric}"] = expected_value
            row[metric] = actual_value
            row[f"difference_{metric}"] = difference
        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def plot_panel(
    ax: plt.Axes,
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    reliability_type: str,
    class_name: str,
    title: str,
    x_label: str,
    show_legend: bool,
) -> None:
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label="Perfect calibration",
        zorder=1,
    )

    summary_by_method = summary.set_index("method")
    metric = (
        "weighted_multiclass_ece"
        if reliability_type == "multiclass"
        else f"{class_name}_ece"
    )

    for method in METHOD_ORDER:
        subset = bins[
            (bins["method"] == method)
            & (bins["reliability_type"] == reliability_type)
            & (bins["class"] == class_name)
            & bins["mean_predicted_probability"].notna()
        ].sort_values("bin")
        ece = float(summary_by_method.loc[method, metric])
        ax.plot(
            subset["mean_predicted_probability"],
            subset["observed_frequency"],
            color=METHOD_COLORS[method],
            marker="o",
            markersize=3.8,
            linewidth=1.35,
            label=f"{METHOD_LABELS[method]} (ECE={ece:.2e})",
            zorder=2,
        )

    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Observed frequency" if class_name != "argmax" else "Accuracy")
    ax.grid(True, color="#E6E6E6", linewidth=0.7)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=7.6,
            frameon=True,
            framealpha=0.95,
        )


def save_combined_figure(
    output_dir: Path,
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    model_label: str,
    artifact_tag: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.8))
    for ax, (reliability_type, class_name, title, _y_label) in zip(
        axes,
        RELIABILITY_SPECS,
    ):
        x_label = (
            "Mean predicted confidence"
            if reliability_type == "multiclass"
            else "Mean predicted probability"
        )
        plot_panel(
            ax,
            bins,
            summary,
            reliability_type,
            class_name,
            title,
            x_label,
            show_legend=False,
        )

    figure.suptitle(
        f"Flank-400 {model_label} reliability on the reconstructed "
        "valid-position test population",
        fontsize=13,
        y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8.2,
        title="Legend labels report multiclass ECE",
        title_fontsize=8.2,
        frameon=True,
        framealpha=0.95,
    )
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.94))
    figure.savefig(
        output_dir / f"reliability_diagrams_{artifact_tag}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / f"reliability_diagrams_{artifact_tag}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_individual_figures(
    output_dir: Path,
    bins: pd.DataFrame,
    summary: pd.DataFrame,
    artifact_tag: str,
) -> None:
    for reliability_type, class_name, title, y_label in RELIABILITY_SPECS:
        figure, ax = plt.subplots(figsize=(6.5, 5.5))
        x_label = (
            "Mean predicted confidence"
            if reliability_type == "multiclass"
            else "Mean predicted probability"
        )
        plot_panel(
            ax,
            bins,
            summary,
            reliability_type,
            class_name,
            title,
            x_label,
            show_legend=True,
        )
        ax.set_ylabel(y_label)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"reliability_{class_name}_{artifact_tag}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least 2")
    if not np.isclose(
        args.global_temperature,
        1.1,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "this authoritative comparison requires --global-temperature 1.1"
        )
    model_label = args.model_label.strip()
    if not model_label or "\n" in model_label or "\r" in model_label:
        raise ValueError("--model-label must be non-empty and single-line")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.artifact_tag) is None:
        raise ValueError(
            "--artifact-tag must contain only letters, digits, '.', '_', or "
            "'-', and must begin with a letter or digit"
        )

    cache_path = Path(args.cache)
    unweighted_path = Path(args.unweighted_temperature_txt)
    target_weighted_path = Path(args.target_weighted_temperature_txt)
    openspliceai_path = Path(args.openspliceai_temperature_txt)
    point_estimates_path = Path(args.point_estimates_csv)
    output_dir = Path(args.output_dir)

    input_paths: Sequence[Path] = [
        cache_path,
        unweighted_path,
        target_weighted_path,
        openspliceai_path,
        point_estimates_path,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing output directory: {output_dir}"
        )

    logits, y, weights, cache_metadata = validate_and_load_cache(
        cache_path,
        args,
    )
    unweighted_temperature = load_temperature(unweighted_path)
    target_weighted_temperature = load_temperature(target_weighted_path)
    openspliceai_temperature = load_temperature(openspliceai_path)

    probabilities = {
        "uncalibrated": softmax(logits),
        "global_T_1.1": softmax(logits / args.global_temperature),
        "true_logit_unweighted_vector_T": softmax(
            logits / unweighted_temperature.reshape(1, 3)
        ),
        "true_logit_target_weighted_vector_T": softmax(
            logits / target_weighted_temperature.reshape(1, 3)
        ),
        "openspliceai_style_full_validation_vector_T_v2": softmax(
            logits / openspliceai_temperature.reshape(1, 3)
        ),
    }
    if list(probabilities) != METHOD_ORDER:
        raise RuntimeError("internal method-order mismatch")

    bins, computed_summary = compute_reliability_tables(
        probabilities,
        y,
        weights,
        args.n_bins,
    )
    summary = load_and_validate_point_estimates(
        point_estimates_path,
        computed_summary,
    )

    print(f"Cache: {cache_path}")
    print(f"Cache SHA-256: {cache_metadata['cache_sha256']}")
    print(f"Cache class counts: {cache_metadata['sample_class_counts']}")
    print(
        "Target non-splice weight: "
        f"{cache_metadata['valid_nonsplice_weight']:.12f}"
    )
    print(f"Unweighted temperature: {unweighted_temperature}")
    print(f"Target-weighted temperature: {target_weighted_temperature}")
    print(f"OpenSpliceAI-style temperature: {openspliceai_temperature}")
    print(f"Reliability bins: {args.n_bins}")
    print("PASS: plotted ECE values reproduce authoritative point estimates")

    output_dir.mkdir(parents=True, exist_ok=False)
    bins.to_csv(output_dir / "reliability_bins.csv", index=False)
    summary.to_csv(output_dir / "reliability_summary.csv", index=False)

    save_combined_figure(
        output_dir,
        bins,
        computed_summary,
        model_label,
        args.artifact_tag,
    )
    save_individual_figures(
        output_dir,
        bins,
        computed_summary,
        args.artifact_tag,
    )

    script_path = Path(__file__).resolve()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "model_label": model_label,
        "artifact_tag": args.artifact_tag,
        "n_bins": args.n_bins,
        "binning": {
            "type": "equal_width",
            "interval_convention": (
                "floor(probability * n_bins), clipped to n_bins - 1"
            ),
            "ece_weighting": "reconstructed valid-position target population",
        },
        "numeric_path": "float64 NumPy softmax for all methods",
        "cache": cache_metadata,
        "input_files": {
            str(path): sha256_file(path) for path in input_paths[1:]
        },
        "temperatures": {
            "uncalibrated": [1.0, 1.0, 1.0],
            "global_T_1.1": [1.1, 1.1, 1.1],
            "true_logit_unweighted_vector_T": (
                unweighted_temperature.tolist()
            ),
            "true_logit_target_weighted_vector_T": (
                target_weighted_temperature.tolist()
            ),
            "openspliceai_style_full_validation_vector_T_v2": (
                openspliceai_temperature.tolist()
            ),
        },
        "method_order": METHOD_ORDER,
        "point_estimate_validation": {
            "source": str(point_estimates_path),
            "metrics": EXPECTED_ECE_COLUMNS,
            "rtol": 1e-7,
            "atol": 5e-12,
            "status": "passed",
        },
        "outputs": [
            "reliability_bins.csv",
            "reliability_summary.csv",
            f"reliability_diagrams_{args.artifact_tag}.png",
            f"reliability_diagrams_{args.artifact_tag}.pdf",
            f"reliability_argmax_{args.artifact_tag}.png",
            f"reliability_acceptor_{args.artifact_tag}.png",
            f"reliability_donor_{args.artifact_tag}.png",
            "reliability_metadata.json",
        ],
    }
    (output_dir / "reliability_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nReliability summary:")
    print(
        computed_summary[
            ["method", *EXPECTED_ECE_COLUMNS]
        ].to_string(index=False)
    )
    print("\nWrote:")
    for output_name in metadata["outputs"]:
        print(output_dir / output_name)
    print("\nPASS: schema-v2 reliability diagrams completed")


if __name__ == "__main__":
    main()
