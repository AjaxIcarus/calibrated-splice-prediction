#!/usr/bin/env python3
"""Fit sampled-cache and target-weighted vector temperatures from true logits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from calibration_v2_common import (
    apply_global_temperature,
    apply_vector_temperature,
    argmax_summary,
    calibration_metrics,
    create_new_output_dir,
    load_and_validate_cache,
    make_target_weights,
    softmax_np,
    validate_cache_pair,
    write_json,
    write_temperature_text,
)


def fit_vector_temperature(
    logits_np: np.ndarray,
    labels_np: np.ndarray,
    *,
    weights_np: np.ndarray | None,
    learning_rate: float,
    steps: int,
    fit_name: str,
    print_every: int,
) -> tuple[np.ndarray, float]:
    logits = torch.as_tensor(logits_np, dtype=torch.float32)
    labels = torch.as_tensor(
        labels_np.argmax(axis=1),
        dtype=torch.long,
    )
    if weights_np is None:
        weights = torch.ones(len(labels), dtype=torch.float32)
    else:
        normalized_weights = weights_np / np.mean(weights_np)
        weights = torch.as_tensor(
            normalized_weights,
            dtype=torch.float32,
        )

    raw_temperature = torch.nn.Parameter(
        torch.zeros(3, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        [raw_temperature],
        lr=learning_rate,
    )

    best_loss = float("inf")
    best_temperature: np.ndarray | None = None

    for step in range(steps):
        optimizer.zero_grad()
        temperatures = (
            0.05 + 4.95 * torch.sigmoid(raw_temperature)
        )
        scaled_logits = logits / temperatures.reshape(1, 3)
        per_position_loss = torch.nn.functional.cross_entropy(
            scaled_logits,
            labels,
            reduction="none",
        )
        loss = torch.sum(weights * per_position_loss) / torch.sum(weights)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        evaluated_temperature = (
            temperatures.detach().cpu().numpy().copy()
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_temperature = evaluated_temperature

        if (
            step == 0
            or (step + 1) % print_every == 0
            or step == steps - 1
        ):
            print(
                f"{fit_name} step={step + 1}/{steps} "
                f"nll={loss_value:.10f} "
                f"T_nonsplice={evaluated_temperature[0]:.8f} "
                f"T_acceptor={evaluated_temperature[1]:.8f} "
                f"T_donor={evaluated_temperature[2]:.8f}",
                flush=True,
            )

    if best_temperature is None:
        raise RuntimeError(f"{fit_name}: no temperature was evaluated")
    return best_temperature, best_loss


def write_metrics_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fieldnames = [
        "split",
        "target_prior",
        "method",
        "multiclass_ece",
        "multiclass_nll",
        "multiclass_brier",
        "acceptor_ece",
        "acceptor_nll",
        "acceptor_brier",
        "donor_ece",
        "donor_nll",
        "donor_brier",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit both the splice-enriched unweighted vector temperature "
            "and the target-position-weighted vector temperature directly "
            "from cache logits_sample."
        )
    )
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=1.1,
    )
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.global_temperature <= 0:
        raise ValueError("--global-temperature must be positive")
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")

    output_dir = create_new_output_dir(args.out_dir)

    print("Loading and validating corrected caches...", flush=True)
    validation = load_and_validate_cache(args.validation_cache)
    test = load_and_validate_cache(args.test_cache)
    validate_cache_pair(validation, test)

    validation_weights = make_target_weights(validation)
    test_weights = make_target_weights(test)

    print(
        "Validation cache:",
        validation.path,
        validation.logits.shape,
    )
    print("Validation class counts:", validation.class_counts.tolist())
    print(
        f"Validation non-splice weight: "
        f"{validation.negative_weight:.12f}"
    )
    print("Test cache:", test.path, test.logits.shape)
    print("Test class counts:", test.class_counts.tolist())
    print(
        f"Test non-splice weight: {test.negative_weight:.12f}"
    )
    print(
        "PASS: schema-v2 metadata, complete-population census, "
        "one-hot labels, true logits, stored probabilities, and paired "
        "sampling settings are consistent"
    )

    print(
        "\nFitting splice-enriched unweighted true-logit vector "
        "temperature...",
        flush=True,
    )
    unweighted_temperature, unweighted_loss = fit_vector_temperature(
        validation.logits,
        validation.labels,
        weights_np=None,
        learning_rate=args.lr,
        steps=args.steps,
        fit_name="unweighted",
        print_every=args.print_every,
    )

    print(
        "\nFitting target-position-weighted true-logit vector "
        "temperature...",
        flush=True,
    )
    weighted_temperature, weighted_loss = fit_vector_temperature(
        validation.logits,
        validation.labels,
        weights_np=validation_weights,
        learning_rate=args.lr,
        steps=args.steps,
        fit_name="target_weighted",
        print_every=args.print_every,
    )

    methods = {
        "uncalibrated": lambda logits: softmax_np(logits),
        f"global_T_{args.global_temperature}": (
            lambda logits: apply_global_temperature(
                logits,
                args.global_temperature,
            )
        ),
        (
            "true_logit_unweighted_vector_T_"
            f"{unweighted_temperature[0]:.4f}_"
            f"{unweighted_temperature[1]:.4f}_"
            f"{unweighted_temperature[2]:.4f}"
        ): (
            lambda logits: apply_vector_temperature(
                logits,
                unweighted_temperature,
            )
        ),
        (
            "true_logit_target_weighted_vector_T_"
            f"{weighted_temperature[0]:.4f}_"
            f"{weighted_temperature[1]:.4f}_"
            f"{weighted_temperature[2]:.4f}"
        ): (
            lambda logits: apply_vector_temperature(
                logits,
                weighted_temperature,
            )
        ),
    }

    metric_rows: list[dict[str, object]] = []
    evaluation_populations = (
        ("validation", "sampled_splice_enriched", validation, None),
        ("test", "sampled_splice_enriched", test, None),
        (
            "validation",
            "reconstructed_target_position",
            validation,
            validation_weights,
        ),
        (
            "test",
            "reconstructed_target_position",
            test,
            test_weights,
        ),
    )
    for split, target_prior, cache, weights in evaluation_populations:
        for method_name, apply_method in methods.items():
            probabilities = apply_method(cache.logits)
            row: dict[str, object] = {
                "split": split,
                "target_prior": target_prior,
                "method": method_name,
            }
            row.update(
                calibration_metrics(
                    probabilities,
                    cache.labels,
                    args.n_bins,
                    weights,
                )
            )
            metric_rows.append(row)

    metrics_path = (
        output_dir / "logit_vector_temperature_summary.csv"
    )
    write_metrics_csv(metrics_path, metric_rows)

    unweighted_temperature_path = (
        output_dir / "temperature_unweighted_best.txt"
    )
    weighted_temperature_path = (
        output_dir / "temperature_target_weighted_best.txt"
    )
    write_temperature_text(
        unweighted_temperature_path,
        unweighted_temperature,
    )
    write_temperature_text(
        weighted_temperature_path,
        weighted_temperature,
    )
    np.save(
        output_dir / "temperature_unweighted_best.npy",
        unweighted_temperature,
    )
    np.save(
        output_dir / "temperature_target_weighted_best.npy",
        weighted_temperature,
    )

    diagnostic_path = output_dir / "logit_vector_argmax_summary.txt"
    test_method_probabilities = {
        method_name: apply_method(test.logits)
        for method_name, apply_method in methods.items()
    }
    with diagnostic_path.open("w", encoding="utf-8") as handle:
        handle.write("Calibration v2 true-logit argmax summary\n")
        handle.write("========================================\n")
        handle.write(
            f"validation_negative_weight: "
            f"{validation.negative_weight:.12f}\n"
        )
        handle.write(
            f"test_negative_weight: {test.negative_weight:.12f}\n"
        )
        handle.write(
            "unweighted_temperature: "
            f"{unweighted_temperature.tolist()}\n"
        )
        handle.write(
            "target_weighted_temperature: "
            f"{weighted_temperature.tolist()}\n"
        )
        for method_name, probabilities in (
            test_method_probabilities.items()
        ):
            handle.write(f"\n{method_name}\n")
            handle.write("-" * len(method_name) + "\n")
            for class_name, values in argmax_summary(
                probabilities,
                test.labels,
            ).items():
                handle.write(
                    f"{class_name}: true={values['true']}, "
                    f"predicted={values['predicted']}, "
                    f"tp={values['tp']}\n"
                )

    run_metadata = {
        "pipeline_version": 2,
        "objective_distinction": {
            "unweighted": (
                "Unweighted cross-entropy on the sampled, "
                "splice-enriched validation cache"
            ),
            "target_weighted": (
                "Cross-entropy with sampled non-splice positions "
                "weighted by valid_nonsplice_positions_seen / "
                "sampled_negatives"
            ),
        },
        "fit": {
            "learning_rate": args.lr,
            "steps": args.steps,
            "temperature_bounds": [0.05, 5.0],
            "best_unweighted_validation_nll": unweighted_loss,
            "best_target_weighted_validation_nll": weighted_loss,
            "unweighted_temperature": unweighted_temperature.tolist(),
            "target_weighted_temperature": (
                weighted_temperature.tolist()
            ),
        },
        "validation_cache": {
            "path": str(validation.path),
            "sha256": validation.sha256,
            "metadata": validation.metadata,
            "class_counts": validation.class_counts.tolist(),
        },
        "test_cache": {
            "path": str(test.path),
            "sha256": test.sha256,
            "metadata": test.metadata,
            "class_counts": test.class_counts.tolist(),
        },
    }
    metadata_path = output_dir / "calibration_v2_run_metadata.json"
    write_json(metadata_path, run_metadata)

    print("\nBest unweighted temperature:", unweighted_temperature)
    print("Best unweighted validation NLL:", unweighted_loss)
    print("Best target-weighted temperature:", weighted_temperature)
    print("Best target-weighted validation NLL:", weighted_loss)
    print("\nWrote:")
    for path in (
        metrics_path,
        diagnostic_path,
        unweighted_temperature_path,
        weighted_temperature_path,
        metadata_path,
    ):
        print(path)
    print("\nPASS: calibration v2 cache fitting completed")


if __name__ == "__main__":
    main()
