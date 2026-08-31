#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


REPO = Path.home() / "projects" / "calibrated-splice-prediction"
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from calibration_v2_common import (  # noqa: E402
    apply_global_temperature,
    calibration_metrics,
    create_new_output_dir,
    load_and_validate_cache,
    make_target_weights,
    multiclass_nll,
    sha256_file,
    softmax_np,
    validate_cache_pair,
)
from bootstrap_logit_metrics_paired_v2 import average_precision_pair  # noqa: E402


MODELS = [
    {
        "name": "seed11_epoch12",
        "validation": Path.home() / (
            "tcbb_v9_work_2026-08-17/"
            "logit_cache_flank400_seed11_epoch12_fullpopulation_validonly_v2/"
            "validation_sampled_logits.npz"
        ),
        "test": Path.home() / (
            "tcbb_v9_work_2026-08-17/"
            "logit_cache_flank400_seed11_epoch12_fullpopulation_validonly_v2/"
            "test_sampled_logits.npz"
        ),
        "validation_sha256":
            "07a27bad3ee844928c4e9de9f5a7311b4c6033fab2557a4dcf03292b6a960d23",
        "test_sha256":
            "b01ecf561719efe057ef5775e83223b3b1aca7d16d34a33edf3df0f1673bb728",
    },
    {
        "name": "seed23_epoch13",
        "validation": Path.home() / (
            "tcbb_v9_work_2026-08-17/"
            "logit_cache_flank400_seed23_epoch13_fullpopulation_validonly_v2/"
            "validation_sampled_logits.npz"
        ),
        "test": Path.home() / (
            "tcbb_v9_work_2026-08-17/"
            "logit_cache_flank400_seed23_epoch13_fullpopulation_validonly_v2/"
            "test_sampled_logits.npz"
        ),
        "validation_sha256":
            "2c8240ecff47a97234073c7c60c2b4f5c2d83f38a964aa0b7058fa90b0160d50",
        "test_sha256":
            "f224cf31a508ee80470a5b0de7e62ef43473948dfb0928ab6685938e779a4132",
    },
]


def target_auprc(probs, labels, weights):
    result = {}
    y = labels.argmax(axis=1)

    for class_index, class_name in ((1, "acceptor"), (2, "donor")):
        y_binary = (y == class_index).astype(np.float64)

        sampled_ap, target_ap = average_precision_pair(
            y_binary,
            probs[:, class_index],
            weights,
        )

        result[f"sampled_{class_name}_auprc"] = sampled_ap
        result[f"target_weighted_{class_name}_auprc"] = target_ap

    return result


def fit_scalar(validation):
    weights = make_target_weights(validation)

    def objective(log_temperature):
        temperature = math.exp(float(log_temperature))
        probs = apply_global_temperature(
            validation.logits,
            temperature,
        )
        return multiclass_nll(
            probs,
            validation.labels,
            weights,
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(5.0)),
        method="bounded",
        options={
            "xatol": 1e-12,
            "maxiter": 500,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"scalar optimization failed: {result.message}"
        )

    temperature = math.exp(float(result.x))
    return temperature, float(result.fun), result


def evaluate(cache, temperature):
    weights = make_target_weights(cache)

    probabilities = apply_global_temperature(
        cache.logits,
        temperature,
    )

    metrics = calibration_metrics(
        probabilities,
        cache.labels,
        n_bins=15,
        weights=weights,
    )
    metrics.update(
        target_auprc(
            probabilities,
            cache.labels,
            weights,
        )
    )

    return metrics


def evaluate_uncalibrated(cache):
    weights = make_target_weights(cache)
    probabilities = softmax_np(cache.logits)

    metrics = calibration_metrics(
        probabilities,
        cache.labels,
        n_bins=15,
        weights=weights,
    )
    metrics.update(
        target_auprc(
            probabilities,
            cache.labels,
            weights,
        )
    )

    return metrics


def main():
    output_dir = create_new_output_dir(
        Path.home()
        / "tcbb_v9_work_2026-08-17"
        / "target_aware_scalar_corrected_checkpoints_results"
    )

    rows = []
    metadata = {
        "analysis": (
            "Fitted target-aware single scalar temperature; "
            "objective is target-weighted validation multiclass NLL."
        ),
        "temperature_bounds": [0.05, 5.0],
        "optimization_parameterization": "log_temperature",
        "ece_bins": 15,
        "models": {},
    }

    for config in MODELS:
        name = config["name"]

        validation = load_and_validate_cache(config["validation"])
        test = load_and_validate_cache(config["test"])
        validate_cache_pair(validation, test)

        if validation.sha256 != config["validation_sha256"]:
            raise RuntimeError(
                f"{name}: validation cache SHA-256 mismatch"
            )

        if test.sha256 != config["test_sha256"]:
            raise RuntimeError(
                f"{name}: test cache SHA-256 mismatch"
            )

        temperature, validation_nll, optimization = fit_scalar(
            validation
        )

        uncalibrated = evaluate_uncalibrated(test)
        scalar = evaluate(test, temperature)

        (output_dir / f"{name}_temperature.txt").write_text(
            f"{temperature:.15g}\n",
            encoding="utf-8",
        )

        for method, metrics in (
            ("uncalibrated", uncalibrated),
            ("target_aware_fitted_scalar_T", scalar),
        ):
            rows.append(
                {
                    "model": name,
                    "method": method,
                    "temperature": (
                        1.0
                        if method == "uncalibrated"
                        else temperature
                    ),
                    **metrics,
                }
            )

        metadata["models"][name] = {
            "validation_cache": str(config["validation"]),
            "validation_cache_sha256": validation.sha256,
            "test_cache": str(config["test"]),
            "test_cache_sha256": test.sha256,
            "validation_class_counts":
                validation.class_counts.tolist(),
            "test_class_counts":
                test.class_counts.tolist(),
            "validation_nonsplice_weight":
                validation.negative_weight,
            "test_nonsplice_weight":
                test.negative_weight,
            "fitted_temperature": temperature,
            "best_target_weighted_validation_nll":
                validation_nll,
            "optimizer_success": bool(optimization.success),
            "optimizer_iterations": int(optimization.nit),
            "optimizer_function_evaluations": int(
                optimization.nfev
            ),
        }

    fieldnames = list(rows[0].keys())

    csv_path = output_dir / "scalar_baseline_metrics.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata["script_sha256"] = sha256_file(
        Path(__file__).resolve()
    )

    json_path = output_dir / "scalar_baseline_metadata.json"
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "scalar_baseline_summary.txt"

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "TCBB v9 fitted target-aware scalar baseline\n"
            "===========================================\n\n"
        )

        for config in MODELS:
            name = config["name"]
            info = metadata["models"][name]

            handle.write(f"{name}\n")
            handle.write("-" * len(name) + "\n")
            handle.write(
                "fitted_temperature="
                f"{info['fitted_temperature']:.12g}\n"
            )
            handle.write(
                "target_weighted_validation_nll="
                f"{info['best_target_weighted_validation_nll']:.12g}\n"
            )

            model_rows = [
                row for row in rows
                if row["model"] == name
            ]

            for row in model_rows:
                handle.write(
                    f"\nmethod={row['method']}\n"
                    f"  ECE={row['multiclass_ece']:.12g}\n"
                    f"  NLL={row['multiclass_nll']:.12g}\n"
                    f"  Brier={row['multiclass_brier']:.12g}\n"
                    f"  acceptor_AUPRC="
                    f"{row['target_weighted_acceptor_auprc']:.12g}\n"
                    f"  donor_AUPRC="
                    f"{row['target_weighted_donor_auprc']:.12g}\n"
                )

            handle.write("\n")

    print("PASS: fitted target-aware scalar analysis completed")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY={report_path}")
    print(f"METRICS={csv_path}")
    print(f"METADATA={json_path}")


if __name__ == "__main__":
    main()
