#!/usr/bin/env python3
"""Schema-v2 paired bootstrap for all flank-400 calibration methods.

The cache is a stratified sample: every positive position is retained and a
fixed number of valid non-splice positions is sampled. Bootstrap resampling is
therefore performed independently within the three true classes, preserving
the observed stratum sizes and the reconstructed target-position prior.
Exactly the same resampled indices are used for every calibration method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "uncalibrated",
    "global_T_1.1",
    "true_logit_unweighted_vector_T",
    "true_logit_target_weighted_vector_T",
    "openspliceai_style_full_validation_vector_T_v2",
]

LOWER_IS_BETTER = {
    "weighted_multiclass_ece": True,
    "weighted_multiclass_nll": True,
    "weighted_multiclass_brier": True,
    "acceptor_ece": True,
    "donor_ece": True,
    "acceptor_nll": True,
    "donor_nll": True,
    "acceptor_brier": True,
    "donor_brier": True,
    "sampled_acceptor_auprc": False,
    "sampled_donor_auprc": False,
    "target_weighted_acceptor_auprc": False,
    "target_weighted_donor_auprc": False,
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a schema-v2, target-prior-aware, paired position-level "
            "bootstrap for five calibration methods."
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--unweighted-temperature-txt", required=True)
    parser.add_argument("--target-weighted-temperature-txt", required=True)
    parser.add_argument("--openspliceai-temperature-txt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
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
    }
    return logits, y, weights, metadata


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_multiclass_ece(
    probs: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> float:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    bin_index = np.minimum((confidence * n_bins).astype(np.int64), n_bins - 1)

    bin_weight = np.bincount(bin_index, weights=weights, minlength=n_bins)
    bin_confidence = np.bincount(
        bin_index,
        weights=weights * confidence,
        minlength=n_bins,
    )
    bin_accuracy = np.bincount(
        bin_index,
        weights=weights * correct,
        minlength=n_bins,
    )

    nonempty = bin_weight > 0
    total_weight = weights.sum()
    return float(
        np.sum(
            np.abs(
                bin_accuracy[nonempty] / bin_weight[nonempty]
                - bin_confidence[nonempty] / bin_weight[nonempty]
            )
            * bin_weight[nonempty]
            / total_weight
        )
    )


def weighted_binary_ece(
    probability: np.ndarray,
    y_binary: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> float:
    bin_index = np.minimum(
        (probability * n_bins).astype(np.int64),
        n_bins - 1,
    )
    bin_weight = np.bincount(bin_index, weights=weights, minlength=n_bins)
    bin_probability = np.bincount(
        bin_index,
        weights=weights * probability,
        minlength=n_bins,
    )
    bin_observed = np.bincount(
        bin_index,
        weights=weights * y_binary,
        minlength=n_bins,
    )

    nonempty = bin_weight > 0
    total_weight = weights.sum()
    return float(
        np.sum(
            np.abs(
                bin_observed[nonempty] / bin_weight[nonempty]
                - bin_probability[nonempty] / bin_weight[nonempty]
            )
            * bin_weight[nonempty]
            / total_weight
        )
    )


def average_precision_pair(
    y_binary: np.ndarray,
    scores: np.ndarray,
    target_weights: np.ndarray,
) -> Tuple[float, float]:
    """Return sampled and target-weighted non-interpolated average precision.

    Tied scores are aggregated before precision is evaluated, matching the
    threshold behavior of standard average-precision implementations.
    """

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y_binary[order].astype(np.float64, copy=False)
    sorted_target_weights = target_weights[order]

    group_start = np.r_[
        0,
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1,
    ]

    sampled_total_by_group = np.diff(np.r_[group_start, len(sorted_scores)])
    sampled_positive_by_group = np.add.reduceat(sorted_y, group_start)
    target_total_by_group = np.add.reduceat(
        sorted_target_weights,
        group_start,
    )
    target_positive_by_group = np.add.reduceat(
        sorted_target_weights * sorted_y,
        group_start,
    )

    sampled_positive_total = sampled_positive_by_group.sum()
    target_positive_total = target_positive_by_group.sum()
    if sampled_positive_total <= 0 or target_positive_total <= 0:
        raise ValueError("average precision is undefined without positive cases")

    sampled_cumulative_total = np.cumsum(sampled_total_by_group)
    sampled_cumulative_positive = np.cumsum(sampled_positive_by_group)
    sampled_precision = (
        sampled_cumulative_positive / sampled_cumulative_total
    )
    sampled_ap = np.sum(
        sampled_precision
        * sampled_positive_by_group
        / sampled_positive_total
    )

    target_cumulative_total = np.cumsum(target_total_by_group)
    target_cumulative_positive = np.cumsum(target_positive_by_group)
    target_precision = target_cumulative_positive / target_cumulative_total
    target_ap = np.sum(
        target_precision
        * target_positive_by_group
        / target_positive_total
    )

    return float(sampled_ap), float(target_ap)


def compute_metrics(
    probs: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> Dict[str, float]:
    row = np.arange(len(y))
    p_true = np.clip(probs[row, y], 1e-12, 1.0)
    one_hot = np.eye(3, dtype=np.float64)[y]

    metrics: Dict[str, float] = {
        "weighted_multiclass_ece": weighted_multiclass_ece(
            probs,
            y,
            weights,
            n_bins,
        ),
        "weighted_multiclass_nll": weighted_mean(-np.log(p_true), weights),
        "weighted_multiclass_brier": weighted_mean(
            np.sum((probs - one_hot) ** 2, axis=1),
            weights,
        ),
    }

    for class_index, class_name in [(1, "acceptor"), (2, "donor")]:
        y_binary = (y == class_index).astype(np.float64)
        probability = probs[:, class_index]
        probability_clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)

        metrics[f"{class_name}_ece"] = weighted_binary_ece(
            probability,
            y_binary,
            weights,
            n_bins,
        )
        metrics[f"{class_name}_nll"] = weighted_mean(
            -(
                y_binary * np.log(probability_clipped)
                + (1.0 - y_binary) * np.log1p(-probability_clipped)
            ),
            weights,
        )
        metrics[f"{class_name}_brier"] = weighted_mean(
            (probability - y_binary) ** 2,
            weights,
        )

        sampled_ap, target_ap = average_precision_pair(
            y_binary,
            probability,
            weights,
        )
        metrics[f"sampled_{class_name}_auprc"] = sampled_ap
        metrics[f"target_weighted_{class_name}_auprc"] = target_ap

    return metrics


def summarize_replicates(
    bootstrap: pd.DataFrame,
    metric_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for method_name in METHOD_ORDER:
        method_frame = bootstrap[bootstrap["method"] == method_name]
        for metric_name in metric_names:
            values = method_frame[metric_name].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "method": method_name,
                    "metric": metric_name,
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_std": float(np.std(values, ddof=1)),
                    "ci_lower_2.5": float(np.percentile(values, 2.5)),
                    "ci_upper_97.5": float(np.percentile(values, 97.5)),
                    "n_bootstrap": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def make_paired_differences(
    bootstrap: pd.DataFrame,
    metric_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    indexed = {
        method_name: bootstrap[bootstrap["method"] == method_name]
        .sort_values("bootstrap")
        .set_index("bootstrap")
        for method_name in METHOD_ORDER
    }

    replicate_rows = []
    summary_rows = []

    for reference_method, comparison_method in combinations(METHOD_ORDER, 2):
        reference = indexed[reference_method]
        comparison = indexed[comparison_method]
        if not reference.index.equals(comparison.index):
            raise RuntimeError("paired bootstrap indices are not aligned")

        for metric_name in metric_names:
            method_minus_reference = (
                comparison[metric_name] - reference[metric_name]
            ).to_numpy(dtype=np.float64)
            if LOWER_IS_BETTER[metric_name]:
                improvement = -method_minus_reference
                direction = "lower_is_better"
            else:
                improvement = method_minus_reference
                direction = "higher_is_better"

            for bootstrap_id, difference, gain in zip(
                reference.index.to_numpy(),
                method_minus_reference,
                improvement,
            ):
                replicate_rows.append(
                    {
                        "bootstrap": int(bootstrap_id),
                        "reference_method": reference_method,
                        "comparison_method": comparison_method,
                        "metric": metric_name,
                        "direction": direction,
                        "method_minus_reference": float(difference),
                        "improvement": float(gain),
                    }
                )

            ci_lower = float(np.percentile(improvement, 2.5))
            ci_upper = float(np.percentile(improvement, 97.5))
            probability_better = float(
                np.mean(improvement > 0.0)
                + 0.5 * np.mean(improvement == 0.0)
            )
            summary_rows.append(
                {
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "metric": metric_name,
                    "direction": direction,
                    "mean_method_minus_reference": float(
                        np.mean(method_minus_reference)
                    ),
                    "mean_improvement": float(np.mean(improvement)),
                    "improvement_ci_lower_2.5": ci_lower,
                    "improvement_ci_upper_97.5": ci_upper,
                    "probability_comparison_better": probability_better,
                    "improvement_ci_excludes_zero": bool(
                        ci_lower > 0.0 or ci_upper < 0.0
                    ),
                    "n_bootstrap": int(len(improvement)),
                }
            )

    return pd.DataFrame(replicate_rows), pd.DataFrame(summary_rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate."""

    def format_value(value) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            return f"{float(value):.10g}"
        if isinstance(value, (bool, np.bool_)):
            return "True" if bool(value) else "False"
        return str(value).replace("|", r"\|").replace("\n", " ")

    headers = [str(column).replace("|", r"\|") for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_markdown_summary(
    path: Path,
    point_estimates: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> None:
    core_metrics = [
        "weighted_multiclass_ece",
        "weighted_multiclass_nll",
        "weighted_multiclass_brier",
        "target_weighted_acceptor_auprc",
        "target_weighted_donor_auprc",
    ]
    core_points = point_estimates[
        ["method", *core_metrics]
    ].copy()

    primary_pair = paired_summary[
        (
            paired_summary["reference_method"]
            == "true_logit_target_weighted_vector_T"
        )
        & (
            paired_summary["comparison_method"]
            == "openspliceai_style_full_validation_vector_T_v2"
        )
        & paired_summary["metric"].isin(core_metrics)
    ].copy()

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Schema-v2 paired bootstrap summary\n\n")
        handle.write(
            f"- Bootstrap unit: position, stratified by true class\n"
            f"- Replicates: {n_bootstrap}\n"
            f"- Bootstrap seed: {bootstrap_seed}\n"
            f"- All five methods use identical resampled indices\n\n"
        )
        handle.write("## Point estimates\n\n")
        handle.write(dataframe_to_markdown(core_points))
        handle.write("\n\n## Bootstrap intervals\n\n")
        handle.write(
            dataframe_to_markdown(
                bootstrap_summary[
                    bootstrap_summary["metric"].isin(core_metrics)
                ]
            )
        )
        handle.write(
            "\n\n## OpenSpliceAI-style versus target-weighted vector-T\n\n"
        )
        handle.write(dataframe_to_markdown(primary_pair))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 2:
        raise ValueError("--n-bootstrap must be at least 2")
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

    cache_path = Path(args.cache)
    unweighted_path = Path(args.unweighted_temperature_txt)
    target_weighted_path = Path(args.target_weighted_temperature_txt)
    openspliceai_path = Path(args.openspliceai_temperature_txt)
    output_dir = Path(args.output_dir)

    for path in [
        cache_path,
        unweighted_path,
        target_weighted_path,
        openspliceai_path,
    ]:
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

    methods = {
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
    if list(methods) != METHOD_ORDER:
        raise RuntimeError("internal method-order mismatch")

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
    print(f"Bootstrap replicates: {args.n_bootstrap}")
    print(f"Bootstrap seed: {args.bootstrap_seed}")
    print("Bootstrap design: paired, stratified by true class")

    output_dir.mkdir(parents=True, exist_ok=False)

    point_rows = []
    for method_name in METHOD_ORDER:
        metrics = compute_metrics(
            methods[method_name],
            y,
            weights,
            args.n_bins,
        )
        point_rows.append({"method": method_name, **metrics})
        print(
            f"Point estimate {method_name}: "
            f"ECE={metrics['weighted_multiclass_ece']:.12g} "
            f"NLL={metrics['weighted_multiclass_nll']:.12g} "
            f"Brier={metrics['weighted_multiclass_brier']:.12g}"
        )
    point_estimates = pd.DataFrame(point_rows)
    metric_names = [
        column for column in point_estimates.columns if column != "method"
    ]
    if set(metric_names) != set(LOWER_IS_BETTER):
        raise RuntimeError("internal metric registry mismatch")

    class_indices = [np.flatnonzero(y == class_index) for class_index in range(3)]
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_rows = []

    for bootstrap_id in range(args.n_bootstrap):
        sampled_parts = [
            indices[rng.integers(0, len(indices), size=len(indices))]
            for indices in class_indices
        ]
        sampled_index = np.concatenate(sampled_parts)
        sampled_y = y[sampled_index]
        sampled_weights = weights[sampled_index]

        for method_name in METHOD_ORDER:
            metrics = compute_metrics(
                methods[method_name][sampled_index],
                sampled_y,
                sampled_weights,
                args.n_bins,
            )
            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_id,
                    "method": method_name,
                    **metrics,
                }
            )

        if (bootstrap_id + 1) % 10 == 0 or (
            bootstrap_id + 1 == args.n_bootstrap
        ):
            print(
                f"Completed bootstrap {bootstrap_id + 1}/"
                f"{args.n_bootstrap}",
                flush=True,
            )

    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap_summary = summarize_replicates(bootstrap, metric_names)
    paired_replicates, paired_summary = make_paired_differences(
        bootstrap,
        metric_names,
    )

    point_path = output_dir / "point_estimates.csv"
    bootstrap_path = output_dir / "bootstrap_replicates.csv"
    bootstrap_summary_path = output_dir / "bootstrap_summary.csv"
    paired_path = output_dir / "paired_difference_replicates.csv"
    paired_summary_path = output_dir / "paired_difference_summary.csv"
    metadata_path = output_dir / "bootstrap_metadata.json"
    markdown_path = output_dir / "bootstrap_summary.md"

    point_estimates.to_csv(point_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    bootstrap_summary.to_csv(bootstrap_summary_path, index=False)
    paired_replicates.to_csv(paired_path, index=False)
    paired_summary.to_csv(paired_summary_path, index=False)
    write_markdown_summary(
        markdown_path,
        point_estimates,
        bootstrap_summary,
        paired_summary,
        args.n_bootstrap,
        args.bootstrap_seed,
    )

    metadata = {
        **cache_metadata,
        "analysis": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "position",
            "resampling": "stratified_by_true_class",
            "paired_across_methods": True,
            "n_bins": args.n_bins,
            "ece_bins": "[lo, hi), final bin includes 1.0",
            "global_temperature": args.global_temperature,
            "method_order": METHOD_ORDER,
            "metric_direction": {
                name: (
                    "lower_is_better"
                    if is_lower
                    else "higher_is_better"
                )
                for name, is_lower in LOWER_IS_BETTER.items()
            },
            "limitation": (
                "Position-level bootstrap; the cache has no sequence/gene "
                "identifier, so within-sequence biological dependence is not "
                "represented."
            ),
        },
        "temperature_inputs": {
            "unweighted": {
                "path": str(unweighted_path),
                "sha256": sha256_file(unweighted_path),
                "value": unweighted_temperature.tolist(),
            },
            "target_weighted": {
                "path": str(target_weighted_path),
                "sha256": sha256_file(target_weighted_path),
                "value": target_weighted_temperature.tolist(),
            },
            "openspliceai_style": {
                "path": str(openspliceai_path),
                "sha256": sha256_file(openspliceai_path),
                "value": openspliceai_temperature.tolist(),
            },
        },
        "outputs": [
            point_path.name,
            bootstrap_path.name,
            bootstrap_summary_path.name,
            paired_path.name,
            paired_summary_path.name,
            metadata_path.name,
            markdown_path.name,
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nWrote:")
    for path in [
        point_path,
        bootstrap_path,
        bootstrap_summary_path,
        paired_path,
        paired_summary_path,
        metadata_path,
        markdown_path,
    ]:
        print(path)
    print("\nPASS: schema-v2 unified paired bootstrap completed")


if __name__ == "__main__":
    main()
