#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.home() / "projects" / "calibrated-splice-prediction"
WORK = Path.home() / "tcbb_v9_work_2026-08-17"

sys.path.insert(0, str(REPO / "scripts"))

from bootstrap_logit_metrics_paired_v2 import (  # noqa: E402
    compute_metrics,
    load_temperature,
    sha256_file,
    softmax,
    validate_and_load_cache,
)


METHOD_ORDER = [
    "target_aware_fitted_scalar_T",
    "true_logit_target_weighted_vector_T",
    "openspliceai_style_full_validation_vector_T_v2",
]

MODELS = {
    "seed11_epoch12": {
        "cache": WORK / (
            "logit_cache_flank400_seed11_epoch12_"
            "fullpopulation_validonly_v2/test_sampled_logits.npz"
        ),
        "cache_sha256":
            "b01ecf561719efe057ef5775e83223b3b1aca7d16d34a33edf3df0f1673bb728",

        "scalar": WORK / (
            "target_aware_scalar_corrected_checkpoints_results/"
            "seed11_epoch12_temperature.txt"
        ),
        "scalar_sha256":
            "b511229a378f3ac1da9a8c5ab7e5240a2dfb9664d3bc3c2819a48e5e78da9b53",

        "target_vector": WORK / (
            "logit_calibration_seed11_epoch12_corrected_v9_steps6000/"
            "temperature_target_weighted_best.txt"
        ),
        "target_vector_sha256":
            "c5d49d16580fdeb2cc1e2d18b684f4bb1c4505902a4b23a0502c3eea41c11c76",

        "openspliceai_vector": WORK / (
            "openspliceai_style_vectorT_seed11_epoch12_corrected_v9_"
            "fullval_validonly_v2/temperature_best.txt"
        ),
        "openspliceai_vector_sha256":
            "a20ad754ea5667111935e7873b16825098085a0ff56677015dcff424c522d1ec",
    },

    "seed23_epoch13": {
        "cache": WORK / (
            "logit_cache_flank400_seed23_epoch13_"
            "fullpopulation_validonly_v2/test_sampled_logits.npz"
        ),
        "cache_sha256":
            "f224cf31a508ee80470a5b0de7e62ef43473948dfb0928ab6685938e779a4132",

        "scalar": WORK / (
            "target_aware_scalar_corrected_checkpoints_results/"
            "seed23_epoch13_temperature.txt"
        ),
        "scalar_sha256":
            "542030f18b486cba339c6d392417d0b15c6168371363617b531f9bac1f5b8191",

        "target_vector": WORK / (
            "logit_calibration_seed23_epoch13_corrected_v9_steps6000/"
            "temperature_target_weighted_best.txt"
        ),
        "target_vector_sha256":
            "8baad0d64d364825a8b3176c891d3bf34a5f5ef6f26216a0a22375b9f4f7459f",

        "openspliceai_vector": WORK / (
            "openspliceai_style_vectorT_seed23_epoch13_corrected_v9_"
            "fullval_validonly_v2/temperature_best.txt"
        ),
        "openspliceai_vector_sha256":
            "f44e3e5813e45d37f0953c02ca1ce11ba694318f7b1a091149af8ccf95a93e90",
    },
}


def read_scalar(path: Path) -> float:
    value = float(path.read_text(encoding="utf-8").strip())
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid scalar temperature: {value}")
    return value


def check_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"SHA-256 mismatch for {path}\n"
            f"expected={expected}\nactual={actual}"
        )


def paired_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        c for c in frame.columns
        if c not in {"bootstrap", "method"}
    ]

    references = [
        "true_logit_target_weighted_vector_T",
        "openspliceai_style_full_validation_vector_T_v2",
    ]

    rows = []

    scalar = (
        frame[frame["method"] == "target_aware_fitted_scalar_T"]
        .sort_values("bootstrap")
        .set_index("bootstrap")
    )

    for reference_name in references:
        reference = (
            frame[frame["method"] == reference_name]
            .sort_values("bootstrap")
            .set_index("bootstrap")
        )

        if not scalar.index.equals(reference.index):
            raise RuntimeError("paired bootstrap IDs are not aligned")

        for metric in metrics:
            difference = (
                scalar[metric].to_numpy(dtype=np.float64)
                - reference[metric].to_numpy(dtype=np.float64)
            )

            lower_is_better = not metric.endswith("auprc")

            improvement = (
                -difference if lower_is_better else difference
            )

            rows.append({
                "reference_method": reference_name,
                "comparison_method":
                    "target_aware_fitted_scalar_T",
                "metric": metric,
                "direction":
                    "lower_is_better"
                    if lower_is_better
                    else "higher_is_better",
                "mean_scalar_minus_reference":
                    float(np.mean(difference)),
                "mean_improvement":
                    float(np.mean(improvement)),
                "improvement_ci_lower_2.5":
                    float(np.percentile(improvement, 2.5)),
                "improvement_ci_upper_97.5":
                    float(np.percentile(improvement, 97.5)),
                "probability_scalar_better":
                    float(
                        np.mean(improvement > 0)
                        + 0.5 * np.mean(improvement == 0)
                    ),
                "n_bootstrap": len(improvement),
            })

    return pd.DataFrame(rows)


def run_model(
    name: str,
    config: dict,
    output_dir: Path,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> None:

    for key, hash_key in [
        ("cache", "cache_sha256"),
        ("scalar", "scalar_sha256"),
        ("target_vector", "target_vector_sha256"),
        ("openspliceai_vector", "openspliceai_vector_sha256"),
    ]:
        path = config[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        check_hash(path, config[hash_key])

    class Args:
        expected_cache_sha256 = config["cache_sha256"]
        expected_cache_random_seed = None
        expected_total_sequences = None
        expected_total_positions = None
        expected_valid_positions = None
        expected_padding_positions = None
        expected_positive_positions = None
        expected_valid_nonsplice_positions = None
        expected_sampled_negatives = None

    logits, y, weights, metadata = validate_and_load_cache(
        config["cache"],
        Args(),
    )

    scalar_t = read_scalar(config["scalar"])
    target_vector = load_temperature(config["target_vector"])
    openspliceai_vector = load_temperature(
        config["openspliceai_vector"]
    )

    methods = {
        "target_aware_fitted_scalar_T":
            softmax(logits / scalar_t),

        "true_logit_target_weighted_vector_T":
            softmax(logits / target_vector.reshape(1, 3)),

        "openspliceai_style_full_validation_vector_T_v2":
            softmax(logits / openspliceai_vector.reshape(1, 3)),
    }

    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=False)

    point_rows = []

    for method in METHOD_ORDER:
        metrics = compute_metrics(
            methods[method],
            y,
            weights,
            15,
        )
        point_rows.append({
            "method": method,
            **metrics,
        })

    pd.DataFrame(point_rows).to_csv(
        model_dir / "point_estimates.csv",
        index=False,
    )

    class_indices = [
        np.flatnonzero(y == class_index)
        for class_index in range(3)
    ]

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_rows = []

    start = time.perf_counter()

    for bootstrap_id in range(n_bootstrap):
        sampled_parts = [
            indices[
                rng.integers(
                    0,
                    len(indices),
                    size=len(indices),
                )
            ]
            for indices in class_indices
        ]

        sampled_index = np.concatenate(sampled_parts)
        sampled_y = y[sampled_index]
        sampled_weights = weights[sampled_index]

        for method in METHOD_ORDER:
            metrics = compute_metrics(
                methods[method][sampled_index],
                sampled_y,
                sampled_weights,
                15,
            )

            bootstrap_rows.append({
                "bootstrap": bootstrap_id,
                "method": method,
                **metrics,
            })

    elapsed = time.perf_counter() - start

    bootstrap = pd.DataFrame(bootstrap_rows)

    bootstrap.to_csv(
        model_dir / "bootstrap_replicates.csv",
        index=False,
    )

    summary = paired_summary(bootstrap)

    summary.to_csv(
        model_dir / "scalar_paired_comparisons.csv",
        index=False,
    )

    runtime = {
        "model": name,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "elapsed_seconds": elapsed,
        "seconds_per_replicate": elapsed / n_bootstrap,
        "projected_seconds_for_2000":
            elapsed / n_bootstrap * 2000,
        "cache_metadata": metadata,
        "scalar_temperature": scalar_t,
        "target_vector_temperature":
            target_vector.tolist(),
        "openspliceai_vector_temperature":
            openspliceai_vector.tolist(),
    }

    (model_dir / "run_metadata.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"{name}: {n_bootstrap} replicates in "
        f"{elapsed:.2f}s; "
        f"{elapsed / n_bootstrap:.2f}s/replicate; "
        f"projected 2000="
        f"{elapsed / n_bootstrap * 2000 / 3600:.2f}h"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=170817)
    args = parser.parse_args()

    if args.n_bootstrap < 2:
        raise ValueError("n-bootstrap must be >=2")

    output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse output directory: {output_dir}"
        )

    output_dir.mkdir(parents=True)

    for name, config in MODELS.items():
        run_model(
            name,
            config,
            output_dir,
            args.n_bootstrap,
            args.bootstrap_seed,
        )

    print("PASS: scalar-vs-vector paired bootstrap completed")


if __name__ == "__main__":
    main()
