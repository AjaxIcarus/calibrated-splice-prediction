#!/usr/bin/env python3
"""Evaluate a three-class temperature on a corrected schema-v2 logit cache."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from calibration_v2_common import (
    apply_vector_temperature,
    argmax_summary,
    calibration_metrics,
    create_new_output_dir,
    load_and_validate_cache,
    load_temperature_text,
    make_target_weights,
    write_json,
)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def threshold_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    class_index: int,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_binary = (labels == class_index)
    class_probabilities = probabilities[:, class_index]

    for threshold in thresholds:
        predicted = class_probabilities >= threshold
        true_positive_mask = predicted & y_binary
        false_positive_mask = predicted & ~y_binary
        false_negative_mask = ~predicted & y_binary

        true_positives = int(np.count_nonzero(true_positive_mask))
        false_positives = int(np.count_nonzero(false_positive_mask))
        false_negatives = int(np.count_nonzero(false_negative_mask))
        sampled_precision_denominator = (
            true_positives + false_positives
        )
        sampled_recall_denominator = (
            true_positives + false_negatives
        )

        weighted_true_positives = float(
            np.sum(weights[true_positive_mask])
        )
        weighted_false_positives = float(
            np.sum(weights[false_positive_mask])
        )
        weighted_false_negatives = float(
            np.sum(weights[false_negative_mask])
        )
        weighted_precision_denominator = (
            weighted_true_positives + weighted_false_positives
        )
        weighted_recall_denominator = (
            weighted_true_positives + weighted_false_negatives
        )

        rows.append(
            {
                "class": class_index,
                "class_name": (
                    "acceptor" if class_index == 1 else "donor"
                ),
                "threshold": threshold,
                "sampled_predicted_positive": int(
                    np.count_nonzero(predicted)
                ),
                "sampled_tp": true_positives,
                "sampled_fp": false_positives,
                "sampled_fn": false_negatives,
                "sampled_precision": (
                    true_positives / sampled_precision_denominator
                    if sampled_precision_denominator
                    else np.nan
                ),
                "sampled_recall": (
                    true_positives / sampled_recall_denominator
                    if sampled_recall_denominator
                    else np.nan
                ),
                "target_weighted_predicted_positive": float(
                    np.sum(weights[predicted])
                ),
                "target_weighted_tp": weighted_true_positives,
                "target_weighted_fp": weighted_false_positives,
                "target_weighted_fn": weighted_false_negatives,
                "target_weighted_precision": (
                    weighted_true_positives
                    / weighted_precision_denominator
                    if weighted_precision_denominator
                    else np.nan
                ),
                "target_weighted_recall": (
                    weighted_true_positives
                    / weighted_recall_denominator
                    if weighted_recall_denominator
                    else np.nan
                ),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenSpliceAI-style class-wise temperature using "
            "true logits and target-position reconstruction weights from "
            "a corrected schema-v2 cache."
        )
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--temperature-txt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method-name",
        default="openspliceai_style_full_validation_vectorT_v2",
    )
    parser.add_argument("--n-bins", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")

    output_dir = create_new_output_dir(args.output_dir)
    cache = load_and_validate_cache(args.cache)
    temperatures = load_temperature_text(args.temperature_txt)
    probabilities = apply_vector_temperature(
        cache.logits,
        temperatures,
    )
    labels = cache.labels.argmax(axis=1)
    target_weights = make_target_weights(cache)

    sampled_metrics = calibration_metrics(
        probabilities,
        cache.labels,
        args.n_bins,
        weights=None,
    )
    target_metrics = calibration_metrics(
        probabilities,
        cache.labels,
        args.n_bins,
        weights=target_weights,
    )

    acceptor_binary = (labels == 1).astype(np.int8)
    donor_binary = (labels == 2).astype(np.int8)
    sampled_acceptor_auprc = float(
        average_precision_score(
            acceptor_binary,
            probabilities[:, 1],
        )
    )
    sampled_donor_auprc = float(
        average_precision_score(
            donor_binary,
            probabilities[:, 2],
        )
    )
    target_acceptor_auprc = float(
        average_precision_score(
            acceptor_binary,
            probabilities[:, 1],
            sample_weight=target_weights,
        )
    )
    target_donor_auprc = float(
        average_precision_score(
            donor_binary,
            probabilities[:, 2],
            sample_weight=target_weights,
        )
    )

    summary_row: dict[str, object] = {
        "method": args.method_name,
        "T_nonsplice": temperatures[0],
        "T_acceptor": temperatures[1],
        "T_donor": temperatures[2],
        "weighted_multiclass_ece": (
            target_metrics["multiclass_ece"]
        ),
        "weighted_multiclass_nll": (
            target_metrics["multiclass_nll"]
        ),
        "weighted_nll": target_metrics["multiclass_nll"],
        "weighted_multiclass_brier": (
            target_metrics["multiclass_brier"]
        ),
        "acceptor_ece": target_metrics["acceptor_ece"],
        "donor_ece": target_metrics["donor_ece"],
        "sampled_multiclass_ece": (
            sampled_metrics["multiclass_ece"]
        ),
        "sampled_multiclass_nll": (
            sampled_metrics["multiclass_nll"]
        ),
        "acceptor_auprc": sampled_acceptor_auprc,
        "donor_auprc": sampled_donor_auprc,
        "sampled_acceptor_auprc": sampled_acceptor_auprc,
        "sampled_donor_auprc": sampled_donor_auprc,
        "target_weighted_acceptor_auprc": target_acceptor_auprc,
        "target_weighted_donor_auprc": target_donor_auprc,
        "cache_schema_version": (
            cache.metadata["cache_schema_version"]
        ),
        "total_sequences_seen": (
            cache.metadata["total_sequences_seen"]
        ),
        "total_positions_seen": (
            cache.metadata["total_positions_seen"]
        ),
        "valid_positions_seen": (
            cache.metadata["valid_positions_seen"]
        ),
        "padding_positions_seen": (
            cache.metadata["padding_positions_seen"]
        ),
        "positives_seen": (
            cache.metadata["positive_positions_seen"]
        ),
        "valid_nonsplice_positions_seen": (
            cache.metadata["valid_nonsplice_positions_seen"]
        ),
        "sampled_negatives": cache.metadata["sampled_negatives"],
        "negative_weight": cache.negative_weight,
        "random_seed": cache.metadata["random_seed"],
        "cache_sha256": cache.sha256,
    }
    summary_path = (
        output_dir
        / "openspliceai_style_temperature_test_metrics.csv"
    )
    write_csv(summary_path, [summary_row])

    threshold_rows: list[dict[str, object]] = []
    for class_index in (1, 2):
        threshold_rows.extend(
            threshold_metrics(
                probabilities,
                labels,
                target_weights,
                class_index,
                (0.01, 0.05, 0.1, 0.5),
            )
        )
    threshold_path = (
        output_dir
        / "openspliceai_style_temperature_threshold_metrics.csv"
    )
    write_csv(threshold_path, threshold_rows)

    sampled_argmax = argmax_summary(probabilities, cache.labels)
    predicted_labels = probabilities.argmax(axis=1)
    argmax_rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(
        ("nonsplice", "acceptor", "donor")
    ):
        predicted_mask = predicted_labels == class_index
        true_mask = labels == class_index
        argmax_rows.append(
            {
                "predicted_class": class_index,
                "class_name": class_name,
                "sampled_true_count": (
                    sampled_argmax[class_name]["true"]
                ),
                "sampled_predicted_count": (
                    sampled_argmax[class_name]["predicted"]
                ),
                "sampled_true_positive_count": (
                    sampled_argmax[class_name]["tp"]
                ),
                "target_weighted_true_count": float(
                    np.sum(target_weights[true_mask])
                ),
                "target_weighted_predicted_count": float(
                    np.sum(target_weights[predicted_mask])
                ),
                "target_weighted_true_positive_count": float(
                    np.sum(
                        target_weights[predicted_mask & true_mask]
                    )
                ),
            }
        )
    argmax_path = (
        output_dir / "openspliceai_style_temperature_argmax.csv"
    )
    write_csv(argmax_path, argmax_rows)

    metadata_path = output_dir / "evaluation_metadata.json"
    write_json(
        metadata_path,
        {
            "pipeline_version": 2,
            "method_name": args.method_name,
            "cache_path": str(cache.path),
            "cache_sha256": cache.sha256,
            "cache_metadata": cache.metadata,
            "cache_class_counts": cache.class_counts.tolist(),
            "temperature_path": str(Path(args.temperature_txt)),
            "temperature": temperatures.tolist(),
            "n_bins": args.n_bins,
            "auprc_note": (
                "acceptor_auprc and donor_auprc retain the historical "
                "sampled-cache definition; target-weighted variants are "
                "reported separately"
            ),
        },
    )

    print("Cache:", cache.path)
    print("Cache class counts:", cache.class_counts.tolist())
    print(
        f"Target non-splice weight: {cache.negative_weight:.12f}"
    )
    print("Temperature:", temperatures)
    print("\nTarget-position metrics:")
    print(
        "weighted_multiclass_ece:",
        target_metrics["multiclass_ece"],
    )
    print(
        "weighted_multiclass_nll:",
        target_metrics["multiclass_nll"],
    )
    print("acceptor_auprc:", sampled_acceptor_auprc)
    print("donor_auprc:", sampled_donor_auprc)
    print("\nWrote:")
    for path in (
        summary_path,
        threshold_path,
        argmax_path,
        metadata_path,
    ):
        print(path)
    print(
        "\nPASS: OpenSpliceAI-style temperature evaluation v2 completed"
    )


if __name__ == "__main__":
    main()
