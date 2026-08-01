#!/usr/bin/env python3
"""Shared validation and metric helpers for calibration pipeline v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA_VERSION = 2
CLASS_NAMES = ("nonsplice", "acceptor", "donor")


@dataclass(frozen=True)
class CacheData:
    path: Path
    sha256: str
    logits: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    metadata: dict[str, Any]
    class_counts: np.ndarray
    negative_weight: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(cache: np.lib.npyio.NpzFile, name: str) -> Any:
    if name not in cache.files:
        raise RuntimeError(f"Cache is missing required field: {name}")
    values = np.asarray(cache[name]).reshape(-1)
    if values.size != 1:
        raise RuntimeError(
            f"Cache field {name!r} must contain one scalar; "
            f"found shape {np.asarray(cache[name]).shape}"
        )
    return values[0].item()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits64 = np.asarray(logits, dtype=np.float64)
    shifted = logits64 - np.max(logits64, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def apply_global_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    validate_temperature_vector(
        np.repeat(float(temperature), 3),
        name="global temperature",
    )
    return softmax_np(np.asarray(logits, dtype=np.float64) / temperature)


def apply_vector_temperature(
    logits: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    temperatures = validate_temperature_vector(temperatures)
    return softmax_np(
        np.asarray(logits, dtype=np.float64)
        / temperatures.reshape(1, 3)
    )


def validate_temperature_vector(
    temperatures: np.ndarray,
    *,
    name: str = "temperature vector",
) -> np.ndarray:
    temperatures = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    if temperatures.shape != (3,):
        raise RuntimeError(
            f"{name} must contain exactly three values; "
            f"found shape {temperatures.shape}"
        )
    if not np.all(np.isfinite(temperatures)):
        raise RuntimeError(f"{name} contains non-finite values")
    if not np.all(temperatures > 0):
        raise RuntimeError(f"{name} values must all be positive")
    return temperatures


def load_temperature_text(path: str | Path) -> np.ndarray:
    temperature_path = Path(path)
    if not temperature_path.is_file():
        raise FileNotFoundError(
            f"Temperature file does not exist: {temperature_path}"
        )
    raw = temperature_path.read_text(encoding="utf-8").strip()
    clean = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    temperatures = np.fromstring(clean, sep=" ", dtype=np.float64)
    return validate_temperature_vector(
        temperatures,
        name=f"temperature file {temperature_path}",
    )


def write_temperature_text(
    path: str | Path,
    temperatures: np.ndarray,
) -> None:
    temperatures = validate_temperature_vector(temperatures)
    Path(path).write_text(
        np.array2string(
            temperatures,
            precision=10,
            separator=" ",
            max_line_width=200,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_metadata(metadata: dict[str, Any], path: Path) -> None:
    expected_flags = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "all_sequences_included": 1,
        "padding_excluded_from_sampling": 1,
    }
    for name, expected in expected_flags.items():
        actual = int(metadata[name])
        if actual != expected:
            raise RuntimeError(
                f"{path}: {name} must be {expected}; found {actual}"
            )

    integer_fields = (
        "total_sequences_seen",
        "total_positions_seen",
        "valid_positions_seen",
        "padding_positions_seen",
        "positive_positions_seen",
        "valid_nonsplice_positions_seen",
        "negatives_seen_total",
        "sampled_negatives",
        "random_seed",
    )
    for name in integer_fields:
        value = int(metadata[name])
        if value < 0:
            raise RuntimeError(f"{path}: {name} cannot be negative")

    if int(metadata["total_sequences_seen"]) <= 0:
        raise RuntimeError(f"{path}: total_sequences_seen must be positive")
    if int(metadata["sampled_negatives"]) <= 0:
        raise RuntimeError(f"{path}: sampled_negatives must be positive")

    total_positions = int(metadata["total_positions_seen"])
    valid_positions = int(metadata["valid_positions_seen"])
    padding_positions = int(metadata["padding_positions_seen"])
    positives = int(metadata["positive_positions_seen"])
    valid_nonsplice = int(metadata["valid_nonsplice_positions_seen"])
    negatives_total = int(metadata["negatives_seen_total"])

    if total_positions != valid_positions + padding_positions:
        raise RuntimeError(
            f"{path}: total_positions_seen does not equal "
            "valid_positions_seen + padding_positions_seen"
        )
    if valid_positions != positives + valid_nonsplice:
        raise RuntimeError(
            f"{path}: valid_positions_seen does not equal "
            "positive_positions_seen + valid_nonsplice_positions_seen"
        )
    if negatives_total != valid_nonsplice:
        raise RuntimeError(
            f"{path}: negatives_seen_total does not equal "
            "valid_nonsplice_positions_seen"
        )


def load_and_validate_cache(path: str | Path) -> CacheData:
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache does not exist: {cache_path}")

    required_scalar_fields = (
        "cache_schema_version",
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
        "all_sequences_included",
        "padding_excluded_from_sampling",
        "dataset_path",
    )
    required_array_fields = (
        "logits_sample",
        "labels_sample",
        "probs_sample",
    )

    with np.load(cache_path, allow_pickle=False) as cache:
        missing = [
            name
            for name in (*required_scalar_fields, *required_array_fields)
            if name not in cache.files
        ]
        if missing:
            raise RuntimeError(
                f"{cache_path}: missing required fields: {missing}"
            )

        metadata = {
            name: scalar(cache, name)
            for name in required_scalar_fields
        }
        logits = np.asarray(cache["logits_sample"]).copy()
        labels = np.asarray(cache["labels_sample"]).copy()
        probabilities = np.asarray(cache["probs_sample"]).copy()

    _validate_metadata(metadata, cache_path)

    if logits.ndim != 2 or logits.shape[1] != 3:
        raise RuntimeError(
            f"{cache_path}: logits_sample must have shape (N, 3); "
            f"found {logits.shape}"
        )
    if labels.shape != logits.shape:
        raise RuntimeError(
            f"{cache_path}: labels_sample shape {labels.shape} does not "
            f"match logits_sample shape {logits.shape}"
        )
    if probabilities.shape != logits.shape:
        raise RuntimeError(
            f"{cache_path}: probs_sample shape {probabilities.shape} does "
            f"not match logits_sample shape {logits.shape}"
        )
    if not np.all(np.isfinite(logits)):
        raise RuntimeError(f"{cache_path}: logits_sample contains non-finite values")
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"{cache_path}: probs_sample contains non-finite values")
    if not np.all((labels == 0) | (labels == 1)):
        raise RuntimeError(f"{cache_path}: labels_sample is not binary one-hot data")
    if not np.all(labels.sum(axis=1) == 1):
        raise RuntimeError(
            f"{cache_path}: labels_sample contains padding or malformed rows"
        )

    y = labels.argmax(axis=1)
    class_counts = np.bincount(y, minlength=3)
    sampled_negatives = int(metadata["sampled_negatives"])
    positives = int(metadata["positive_positions_seen"])
    expected_rows = sampled_negatives + positives

    if len(labels) != expected_rows:
        raise RuntimeError(
            f"{cache_path}: cache has {len(labels):,} rows but metadata "
            f"requires {expected_rows:,}"
        )
    if int(class_counts[0]) != sampled_negatives:
        raise RuntimeError(
            f"{cache_path}: sampled non-splice count {class_counts[0]:,} "
            f"does not match sampled_negatives {sampled_negatives:,}"
        )
    if int(class_counts[1] + class_counts[2]) != positives:
        raise RuntimeError(
            f"{cache_path}: sampled positive count "
            f"{int(class_counts[1] + class_counts[2]):,} does not match "
            f"positive_positions_seen {positives:,}"
        )

    negative_weight = (
        float(metadata["negatives_seen_total"]) / sampled_negatives
    )
    stored_weight = float(metadata["valid_nonsplice_weight"])
    if not np.isclose(
        stored_weight,
        negative_weight,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{cache_path}: stored valid_nonsplice_weight "
            f"{stored_weight:.12f} does not match reconstructed weight "
            f"{negative_weight:.12f}"
        )

    reconstructed = softmax_np(logits)
    max_probability_difference = float(
        np.max(np.abs(reconstructed - probabilities))
    )
    if max_probability_difference > 5e-6:
        raise RuntimeError(
            f"{cache_path}: softmax(logits_sample) and probs_sample differ "
            f"by {max_probability_difference:.8g}, exceeding 5e-6"
        )
    probability_row_error = float(
        np.max(np.abs(probabilities.sum(axis=1) - 1.0))
    )
    if probability_row_error > 5e-6:
        raise RuntimeError(
            f"{cache_path}: probability row-sum error "
            f"{probability_row_error:.8g} exceeds 5e-6"
        )

    metadata = dict(metadata)
    metadata["dataset_path"] = str(metadata["dataset_path"])
    metadata["max_softmax_probability_difference"] = (
        max_probability_difference
    )
    metadata["max_probability_row_sum_error"] = probability_row_error

    return CacheData(
        path=cache_path,
        sha256=sha256_file(cache_path),
        logits=logits,
        labels=labels,
        probabilities=probabilities,
        metadata=metadata,
        class_counts=class_counts,
        negative_weight=negative_weight,
    )


def validate_cache_pair(
    validation: CacheData,
    test: CacheData,
) -> None:
    if validation.path.resolve() == test.path.resolve():
        raise RuntimeError(
            "Validation and test cache paths resolve to the same file"
        )
    for name in (
        "cache_schema_version",
        "random_seed",
        "sampled_negatives",
    ):
        validation_value = validation.metadata[name]
        test_value = test.metadata[name]
        if validation_value != test_value:
            raise RuntimeError(
                f"Validation/test cache mismatch for {name}: "
                f"{validation_value!r} versus {test_value!r}"
            )
    if validation.metadata["dataset_path"] == test.metadata["dataset_path"]:
        raise RuntimeError(
            "Validation and test caches report the same dataset_path"
        )


def make_target_weights(cache: CacheData) -> np.ndarray:
    y = cache.labels.argmax(axis=1)
    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = cache.negative_weight
    return weights


def multiclass_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    y = labels.argmax(axis=1)
    p_true = np.clip(
        probabilities[np.arange(len(y)), y],
        1e-12,
        1.0,
    )
    losses = -np.log(p_true)
    if weights is None:
        return float(np.mean(losses))
    return float(np.sum(weights * losses) / np.sum(weights))


def multiclass_brier(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    scores = np.sum((probabilities - labels) ** 2, axis=1)
    if weights is None:
        return float(np.mean(scores))
    return float(np.sum(weights * scores) / np.sum(weights))


def binary_nll(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    losses = -(
        labels * np.log(probabilities)
        + (1.0 - labels) * np.log(1.0 - probabilities)
    )
    if weights is None:
        return float(np.mean(losses))
    return float(np.sum(weights * losses) / np.sum(weights))


def binary_brier(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    scores = (probabilities - labels) ** 2
    if weights is None:
        return float(np.mean(scores))
    return float(np.sum(weights * scores) / np.sum(weights))


def multiclass_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    weights: np.ndarray | None = None,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y_true = labels.argmax(axis=1)
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (y_true == y_pred).astype(np.float64)
    if weights is None:
        weights = np.ones(len(y_true), dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_weight = np.sum(weights)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not np.any(mask):
            continue
        bin_weights = weights[mask]
        bin_weight = np.sum(bin_weights)
        average_confidence = (
            np.sum(bin_weights * confidence[mask]) / bin_weight
        )
        average_accuracy = (
            np.sum(bin_weights * correct[mask]) / bin_weight
        )
        ece += (
            bin_weight
            / total_weight
            * abs(average_accuracy - average_confidence)
        )
    return float(ece)


def binary_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    weights: np.ndarray | None = None,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if weights is None:
        weights = np.ones(len(labels), dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_weight = np.sum(weights)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        bin_weights = weights[mask]
        bin_weight = np.sum(bin_weights)
        average_probability = (
            np.sum(bin_weights * probabilities[mask]) / bin_weight
        )
        observed_frequency = (
            np.sum(bin_weights * labels[mask]) / bin_weight
        )
        ece += (
            bin_weight
            / total_weight
            * abs(observed_frequency - average_probability)
        )
    return float(ece)


def calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "multiclass_ece": multiclass_ece(
            probabilities,
            labels,
            n_bins,
            weights,
        ),
        "multiclass_nll": multiclass_nll(
            probabilities,
            labels,
            weights,
        ),
        "multiclass_brier": multiclass_brier(
            probabilities,
            labels,
            weights,
        ),
    }
    for class_index, class_name in ((1, "acceptor"), (2, "donor")):
        binary_labels = labels[:, class_index]
        class_probabilities = probabilities[:, class_index]
        metrics[f"{class_name}_ece"] = binary_ece(
            class_probabilities,
            binary_labels,
            n_bins,
            weights,
        )
        metrics[f"{class_name}_nll"] = binary_nll(
            class_probabilities,
            binary_labels,
            weights,
        )
        metrics[f"{class_name}_brier"] = binary_brier(
            class_probabilities,
            binary_labels,
            weights,
        )
    return metrics


def argmax_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, int]]:
    y_true = labels.argmax(axis=1)
    y_pred = probabilities.argmax(axis=1)
    return {
        class_name: {
            "true": int(np.count_nonzero(y_true == class_index)),
            "predicted": int(np.count_nonzero(y_pred == class_index)),
            "tp": int(
                np.count_nonzero(
                    (y_true == class_index) & (y_pred == class_index)
                )
            ),
        }
        for class_index, class_name in enumerate(CLASS_NAMES)
    }


def create_new_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.exists():
        if not output_path.is_dir():
            raise RuntimeError(
                f"Output path exists and is not a directory: {output_path}"
            )
        if any(output_path.iterdir()):
            raise RuntimeError(
                f"Refusing to write into non-empty output directory: "
                f"{output_path}"
            )
    else:
        output_path.mkdir(parents=True)
    return output_path


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
