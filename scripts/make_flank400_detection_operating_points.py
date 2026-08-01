#!/usr/bin/env python3
"""Cross-seed AUPRC collation and validation-selected test operating points."""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


TARGET_RECALL = 0.95
RESULTS = Path("results")
OUT = RESULTS / "flank400_detection_cross_seed_publication"
FROZEN_MARKER = (
    RESULTS / "FLANK400_CHROMOSOME_TRANSFER_FROZEN_2026-07-24.txt"
)

MODELS = {
    "seed11_epoch11": {
        "role": "primary",
        "cache": RESULTS / "logit_cache_flank400_seed11_epoch11",
        "detection": RESULTS / "detection_metrics_flank400_seed11_epoch11"
        / "logit_detection_auprc_summary.csv",
        "style": RESULTS
        / "openspliceai_style_evaluation_flank400_seed11_epoch11"
        / "openspliceai_style_temperature_test_metrics.csv",
    },
    "seed23_epoch14": {
        "role": "replication",
        "cache": RESULTS / "logit_cache_flank400_seed23_epoch14",
        "detection": RESULTS / "detection_metrics_flank400_seed23_epoch14"
        / "logit_detection_auprc_summary.csv",
        "style": RESULTS
        / "openspliceai_style_evaluation_flank400_seed23_epoch14"
        / "openspliceai_style_temperature_test_metrics.csv",
    },
}

CLASSES = {1: "acceptor", 2: "donor"}
SAVED_METHODS = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Fixed global T=1.1",
    "logit_unweighted_vector": "Unweighted vector-T",
    "logit_weighted_vector": "Genome-weighted vector-T",
}


def load_cache(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        missing = {
            "logits_sample",
            "labels_sample",
        } - set(data.files)

        if missing:
            raise KeyError(
                f"{path}: missing {sorted(missing)}"
            )

        logits = np.asarray(data["logits_sample"])
        labels = np.asarray(data["labels_sample"])

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(
            f"{path}: unexpected logits shape {logits.shape}"
        )

    if labels.ndim == 2:
        if labels.shape != logits.shape:
            raise ValueError(
                f"{path}: labels/logits shape mismatch"
            )

        if not np.all(np.isfinite(labels)):
            raise ValueError(
                f"{path}: labels contain NaN or Inf"
            )

        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError(
                f"{path}: labels are not binary encoded"
            )

        row_sums = labels.sum(axis=1)

        if np.any(~np.isin(row_sums, [0, 1])):
            raise ValueError(
                f"{path}: labels contain multi-hot rows"
            )

        all_zero_count = int(np.sum(row_sums == 0))

        if all_zero_count:
            print(
                f"NOTE: {path}: treating "
                f"{all_zero_count:,} all-zero label rows as "
                "non-splice, matching the frozen cache pipeline",
                flush=True,
            )

        labels = labels.argmax(axis=1)

    elif labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError(
            f"{path}: unexpected labels shape {labels.shape}"
        )

    labels = labels.astype(np.int8)

    if not set(np.unique(labels)).issubset({0, 1, 2}):
        raise ValueError(
            f"{path}: labels must use class IDs 0, 1, 2"
        )

    return logits, labels


def read_frozen_total_positions():
    if not FROZEN_MARKER.is_file():
        raise FileNotFoundError(FROZEN_MARKER)

    text = FROZEN_MARKER.read_text()
    match = re.search(
        r"^positions_per_model=(\d+)$",
        text,
        re.MULTILINE,
    )

    if not match:
        raise ValueError(
            f"{FROZEN_MARKER}: positions_per_model was not found"
        )

    return int(match.group(1))


def softmax(logits, temperature):
    temperature = np.asarray(
        temperature,
        dtype=np.float64,
    )

    if temperature.ndim:
        temperature = temperature.reshape(1, 3)

    if (
        np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0)
    ):
        raise ValueError(
            f"Invalid temperature: {temperature}"
        )

    values = (
        logits.astype(np.float64, copy=False)
        / temperature
    )
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)

    return values / values.sum(
        axis=1,
        keepdims=True,
    )


def select_threshold(scores, positive):
    positive_scores = np.asarray(
        scores[positive],
        dtype=np.float64,
    )

    needed = math.ceil(
        TARGET_RECALL * len(positive_scores)
    )
    index = len(positive_scores) - needed

    return float(
        np.partition(
            positive_scores,
            index,
        )[index]
    )


def evaluate(
    scores,
    labels,
    positive_class,
    threshold,
    nonsplice_weight,
):
    positive = labels == positive_class
    nonsplice = labels == 0
    other_splice = (
        (labels != 0)
        & ~positive
    )
    predicted = scores >= threshold

    tp = int(
        np.sum(predicted & positive)
    )
    fn = int(
        np.sum(~predicted & positive)
    )
    fp_nonsplice = int(
        np.sum(predicted & nonsplice)
    )
    fp_other = int(
        np.sum(predicted & other_splice)
    )

    sampled_denominator = (
        tp
        + fp_nonsplice
        + fp_other
    )

    weighted_fp = (
        fp_other
        + nonsplice_weight * fp_nonsplice
    )
    weighted_denominator = (
        tp
        + weighted_fp
    )

    weights = np.where(
        nonsplice,
        nonsplice_weight,
        1.0,
    )

    return {
        "tp": tp,
        "fn": fn,
        "fp_other_splice": fp_other,
        "fp_sampled_nonsplice": fp_nonsplice,
        "recall": tp / (tp + fn),
        "sampled_precision": (
            tp / sampled_denominator
            if sampled_denominator
            else np.nan
        ),
        "target_prior_precision": (
            tp / weighted_denominator
            if weighted_denominator
            else np.nan
        ),
        "sampled_auprc": average_precision_score(
            positive,
            scores,
        ),
        "target_prior_auprc": average_precision_score(
            positive,
            scores,
            sample_weight=weights,
        ),
        "estimated_target_false_positives": weighted_fp,
    }


def saved_auprc_rows(
    model,
    config,
    metadata,
):
    frame = pd.read_csv(
        config["detection"]
    )

    required = {
        "method",
        "class",
        "auprc",
        "positives",
    }

    if not required.issubset(frame.columns):
        raise KeyError(
            f"{config['detection']}: missing "
            f"{sorted(required - set(frame))}"
        )

    rows = []
    positives = {}

    for row in frame.to_dict("records"):
        if row["method"] not in SAVED_METHODS:
            raise ValueError(
                f"Unknown saved method: "
                f"{row['method']}"
            )

        positives.setdefault(
            row["class"],
            int(row["positives"]),
        )

        rows.append(
            {
                "model_role": config["role"],
                "model": model,
                "method": SAVED_METHODS[
                    row["method"]
                ],
                "class": row["class"],
                "sampled_test_auprc": float(
                    row["auprc"]
                ),
                "positives": int(
                    row["positives"]
                ),
                "source": str(
                    config["detection"]
                ),
            }
        )

    for class_name in CLASSES.values():
        rows.append(
            {
                "model_role": config["role"],
                "model": model,
                "method": (
                    "OpenSpliceAI-style vector-T"
                ),
                "class": class_name,
                "sampled_test_auprc": float(
                    metadata[
                        f"{class_name}_auprc"
                    ]
                ),
                "positives": positives[
                    class_name
                ],
                "source": str(
                    config["style"]
                ),
            }
        )

    return rows


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_total_positions = (
        read_frozen_total_positions()
    )

    auprc_rows = []
    operating_rows = []

    for model, config in MODELS.items():
        for path in [
            config["detection"],
            config["style"],
        ]:
            if not path.is_file():
                raise FileNotFoundError(path)

        style = pd.read_csv(
            config["style"]
        )

        if len(style) != 1:
            raise ValueError(
                f"{config['style']}: "
                "expected one row"
            )

        metadata = style.iloc[0]

        needed_columns = {
            "T_nonsplice",
            "T_acceptor",
            "T_donor",
            "acceptor_auprc",
            "donor_auprc",
            "total_positions_seen",
            "positives_seen",
            "sampled_negatives",
            "negative_weight",
        }

        if not needed_columns.issubset(
            style.columns
        ):
            raise KeyError(
                f"{config['style']}: missing "
                f"{sorted(
                    needed_columns
                    - set(style.columns)
                )}"
            )

        temperatures = metadata[
            [
                "T_nonsplice",
                "T_acceptor",
                "T_donor",
            ]
        ].to_numpy(
            dtype=np.float64
        )

        validation_logits, validation_labels = (
            load_cache(
                config["cache"]
                / "validation_sampled_logits.npz"
            )
        )

        test_logits, test_labels = load_cache(
            config["cache"]
            / "test_sampled_logits.npz"
        )

        n_splice = int(
            np.sum(test_labels != 0)
        )
        n_nonsplice = int(
            np.sum(test_labels == 0)
        )

        if n_splice != int(
            metadata["positives_seen"]
        ):
            raise ValueError(
                f"{model}: positive-count mismatch"
            )

        if n_nonsplice != int(
            metadata["sampled_negatives"]
        ):
            raise ValueError(
                f"{model}: "
                "sampled-nonsplice-count mismatch"
            )

        evaluator_total_positions = int(
            metadata["total_positions_seen"]
        )
        evaluator_weight = float(
            metadata["negative_weight"]
        )

        derived_evaluator_weight = (
            evaluator_total_positions
            - int(metadata["positives_seen"])
        ) / int(
            metadata["sampled_negatives"]
        )

        if not np.isclose(
            evaluator_weight,
            derived_evaluator_weight,
            rtol=0,
            atol=5e-10,
        ):
            raise ValueError(
                f"{model}: "
                "inconsistent evaluator weight"
            )

        # The frozen cross-seed milestone is
        # authoritative for publication.
        weight = (
            frozen_total_positions
            - int(metadata["positives_seen"])
        ) / int(
            metadata["sampled_negatives"]
        )

        validation_methods = {
            "Uncalibrated": softmax(
                validation_logits,
                1.0,
            ),
            "Fixed global T=1.1": softmax(
                validation_logits,
                1.1,
            ),
            "OpenSpliceAI-style vector-T": (
                softmax(
                    validation_logits,
                    temperatures,
                )
            ),
        }

        test_methods = {
            "Uncalibrated": softmax(
                test_logits,
                1.0,
            ),
            "Fixed global T=1.1": softmax(
                test_logits,
                1.1,
            ),
            "OpenSpliceAI-style vector-T": (
                softmax(
                    test_logits,
                    temperatures,
                )
            ),
        }

        for (
            method,
            validation_probabilities,
        ) in validation_methods.items():

            for (
                class_index,
                class_name,
            ) in CLASSES.items():

                validation_scores = (
                    validation_probabilities[
                        :,
                        class_index,
                    ]
                )

                test_scores = (
                    test_methods[method][
                        :,
                        class_index,
                    ]
                )

                threshold = select_threshold(
                    validation_scores,
                    (
                        validation_labels
                        == class_index
                    ),
                )

                validation_result = evaluate(
                    validation_scores,
                    validation_labels,
                    class_index,
                    threshold,
                    weight,
                )

                test_result = evaluate(
                    test_scores,
                    test_labels,
                    class_index,
                    threshold,
                    weight,
                )

                if (
                    validation_result["recall"]
                    < TARGET_RECALL
                ):
                    raise AssertionError(
                        "Validation recall target "
                        "not attained"
                    )

                operating_rows.append(
                    {
                        "model_role": (
                            config["role"]
                        ),
                        "model": model,
                        "method": method,
                        "class": class_name,
                        "target_validation_recall": (
                            TARGET_RECALL
                        ),
                        "selected_threshold": (
                            threshold
                        ),
                        "validation_recall": (
                            validation_result[
                                "recall"
                            ]
                        ),
                        "test_recall": (
                            test_result["recall"]
                        ),
                        "test_sampled_precision": (
                            test_result[
                                "sampled_precision"
                            ]
                        ),
                        "test_target_prior_precision": (
                            test_result[
                                "target_prior_precision"
                            ]
                        ),
                        "test_sampled_auprc": (
                            test_result[
                                "sampled_auprc"
                            ]
                        ),
                        "test_target_prior_auprc": (
                            test_result[
                                "target_prior_auprc"
                            ]
                        ),
                        "test_tp": (
                            test_result["tp"]
                        ),
                        "test_fn": (
                            test_result["fn"]
                        ),
                        "test_fp_other_splice": (
                            test_result[
                                "fp_other_splice"
                            ]
                        ),
                        "test_fp_sampled_nonsplice": (
                            test_result[
                                "fp_sampled_nonsplice"
                            ]
                        ),
                        "estimated_test_target_false_positives": (
                            test_result[
                                "estimated_target_false_positives"
                            ]
                        ),
                        "nonsplice_weight": weight,
                        "frozen_total_positions": (
                            frozen_total_positions
                        ),
                        "evaluator_total_positions": (
                            evaluator_total_positions
                        ),
                        "evaluator_nonsplice_weight": (
                            evaluator_weight
                        ),
                    }
                )

        auprc_rows.extend(
            saved_auprc_rows(
                model,
                config,
                metadata,
            )
        )

        del validation_logits
        del test_logits
        del validation_methods
        del test_methods

    auprc = pd.DataFrame(
        auprc_rows
    )
    operating = pd.DataFrame(
        operating_rows
    )

    recomputed = operating[
        [
            "model",
            "method",
            "class",
            "test_sampled_auprc",
        ]
    ]

    check = auprc.merge(
        recomputed,
        on=[
            "model",
            "method",
            "class",
        ],
        how="inner",
        validate="one_to_one",
    )

    check["absolute_difference"] = (
        check["sampled_test_auprc"]
        - check["test_sampled_auprc"]
    ).abs()

    if len(check) != 12:
        raise AssertionError(
            "Expected 12 AUPRC checks; "
            f"found {len(check)}"
        )

    max_difference = float(
        check["absolute_difference"].max()
    )

    if max_difference > 2e-7:
        raise AssertionError(
            "Saved/recomputed AUPRC mismatch: "
            f"{max_difference:.3e}"
        )

    spread = (
        auprc.groupby(
            [
                "model_role",
                "model",
                "class",
            ],
            as_index=False,
        )
        .agg(
            min_auprc=(
                "sampled_test_auprc",
                "min",
            ),
            max_auprc=(
                "sampled_test_auprc",
                "max",
            ),
        )
    )

    spread["method_spread"] = (
        spread["max_auprc"]
        - spread["min_auprc"]
    )

    cross_seed = (
        auprc.groupby(
            [
                "method",
                "class",
            ],
            as_index=False,
        )
        .agg(
            mean_auprc=(
                "sampled_test_auprc",
                "mean",
            ),
            min_auprc=(
                "sampled_test_auprc",
                "min",
            ),
            max_auprc=(
                "sampled_test_auprc",
                "max",
            ),
        )
    )

    cross_seed["seed_range"] = (
        cross_seed["max_auprc"]
        - cross_seed["min_auprc"]
    )

    primary_comparison = operating[
        operating["method"].isin(
            [
                "Uncalibrated",
                "OpenSpliceAI-style vector-T",
            ]
        )
    ]

    outputs = {
        "auprc_by_seed_class_method.csv": (
            auprc
        ),
        "validation_selected_95_recall_operating_points.csv": (
            operating
        ),
        "primary_operating_point_comparison.csv": (
            primary_comparison
        ),
        "auprc_within_model_method_spread.csv": (
            spread
        ),
        "auprc_cross_seed_summary.csv": (
            cross_seed
        ),
        "saved_vs_recomputed_auprc_check.csv": (
            check
        ),
    }

    for filename, frame in outputs.items():
        frame.to_csv(
            OUT / filename,
            index=False,
        )

    print(
        "PASS: saved and recomputed "
        "sampled-test AUPRC values agree"
    )
    print(
        "PASS: maximum AUPRC difference "
        f"= {max_difference:.3e}"
    )
    print(
        "PASS: validation-selected "
        "operating-point analysis completed"
    )
    print(
        "Frozen target positions: "
        f"{frozen_total_positions:,}"
    )

    evaluator_totals = sorted(
        operating[
            "evaluator_total_positions"
        ].unique()
    )

    print(
        "Older evaluator position totals: "
        f"{evaluator_totals}"
    )
    print(
        f"Output directory: {OUT}"
    )

    print(
        "\nPrimary operating-point comparison:"
    )
    print(
        primary_comparison[
            [
                "model",
                "class",
                "method",
                "selected_threshold",
                "test_recall",
                "test_target_prior_precision",
                "test_target_prior_auprc",
            ]
        ].to_string(index=False)
    )

    print(
        "\nWithin-model AUPRC method spread:"
    )
    print(
        spread.to_string(index=False)
    )


if __name__ == "__main__":
    main()
