import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from openspliceai.train.train import initialize_model_and_optim
from openspliceai.train_base.utils import CL_max


CACHE_SCHEMA_VERSION = 2


def get_h5_indices(h5f):
    idxs = sorted(
        int(key[1:])
        for key in h5f.keys()
        if key.startswith("X") and key[1:].isdigit()
    )

    if not idxs:
        raise RuntimeError("No X<number> datasets found.")

    missing_y = [
        shard_idx
        for shard_idx in idxs
        if f"Y{shard_idx}" not in h5f
    ]
    if missing_y:
        raise RuntimeError(
            f"Missing label datasets for shards: {missing_y}"
        )

    return idxs


# TCBB v9 duplicate-sensitivity exclusions.
# Exact validation H5 rows whose sequence+label records duplicate training
# records. This extractor is validation-only.
EXCLUDED_ROWS_BY_SHARD = {
    2: frozenset({651}),
    11: frozenset({900}),
    12: frozenset({518}),
}
EXCLUDED_ROW_PAIRS = tuple(
    (shard_idx, row_idx)
    for shard_idx in sorted(EXCLUDED_ROWS_BY_SHARD)
    for row_idx in sorted(EXCLUDED_ROWS_BY_SHARD[shard_idx])
)
EXPECTED_EXCLUDED_SEQUENCE_COUNT = 3

if len(EXCLUDED_ROW_PAIRS) != EXPECTED_EXCLUDED_SEQUENCE_COUNT:
    raise RuntimeError(
        "Duplicate-sensitivity exclusion configuration must contain "
        f"{EXPECTED_EXCLUDED_SEQUENCE_COUNT} rows; found "
        f"{len(EXCLUDED_ROW_PAIRS)}"
    )


def iter_complete_shard_batches(h5f, shard_idx, batch_size):
    """
    Stream every sequence in a shard, including the incomplete final batch.

    This intentionally does not use OpenSpliceAI's load_data_from_shard(),
    because that helper hard-codes DataLoader(drop_last=True).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    x_ds = h5f[f"X{shard_idx}"]
    y_ds = h5f[f"Y{shard_idx}"]

    if x_ds.ndim != 3:
        raise ValueError(
            f"X{shard_idx}: expected 3 dimensions, got {x_ds.shape}"
        )
    if y_ds.ndim != 4 or y_ds.shape[0] != 1:
        raise ValueError(
            f"Y{shard_idx}: expected shape (1, N, L, 3), "
            f"got {y_ds.shape}"
        )
    if y_ds.shape[-1] != 3:
        raise ValueError(
            f"Y{shard_idx}: expected three label classes, "
            f"got {y_ds.shape}"
        )
    if x_ds.shape[0] != y_ds.shape[1]:
        raise ValueError(
            f"Shard {shard_idx}: X/Y sequence counts differ: "
            f"{x_ds.shape[0]} versus {y_ds.shape[1]}"
        )

    sequence_count = int(x_ds.shape[0])

    excluded_rows = EXCLUDED_ROWS_BY_SHARD.get(
        shard_idx,
        frozenset(),
    )

    invalid_excluded_rows = sorted(
        row_idx
        for row_idx in excluded_rows
        if row_idx < 0 or row_idx >= sequence_count
    )
    if invalid_excluded_rows:
        raise RuntimeError(
            f"Shard {shard_idx}: configured exclusion row(s) out of bounds: "
            f"{invalid_excluded_rows}; sequence_count={sequence_count}"
        )

    for start in range(0, sequence_count, batch_size):
        stop = min(start + batch_size, sequence_count)

        row_indices = [
            i for i in range(start, stop)
            if i not in excluded_rows
        ]

        if not row_indices:
            continue

        dnas_np = np.asarray(x_ds[row_indices]).transpose(0, 2, 1)
        labels_np = np.asarray(
            y_ds[0, row_indices]
        ).transpose(0, 2, 1)

        dnas = torch.from_numpy(
            np.ascontiguousarray(dnas_np)
        )
        labels = torch.from_numpy(
            np.ascontiguousarray(labels_np)
        )

        yield dnas, labels


def clip_input_context(dnas, labels, flanking_size):
    """
    Apply the context clipping performed by clip_datapoints without dropping
    samples to force a multi-GPU batch multiple.

    Cache extraction runs on one CPU, so retaining every sample is required.
    """
    context_difference = int(CL_max) - int(flanking_size)

    if context_difference < 0 or context_difference % 2 != 0:
        raise ValueError(
            "CL_max - flanking_size must be a nonnegative even number; "
            f"got CL_max={CL_max}, flanking_size={flanking_size}"
        )

    clip = context_difference // 2

    if clip:
        if dnas.shape[-1] <= 2 * clip:
            raise ValueError(
                "Input sequence is too short for context clipping: "
                f"shape={tuple(dnas.shape)}, clip={clip}"
            )
        dnas = dnas[:, :, clip:-clip]

    return dnas, labels


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def validate_and_partition_labels(labels_flat):
    """
    Return valid, padding, positive, and non-splice masks.

    Valid rows must be binary one-hot vectors. All-zero rows are padding and
    are excluded before argmax and before negative reservoir sampling.
    """
    if labels_flat.ndim != 2 or labels_flat.shape[1] != 3:
        raise ValueError(
            f"Expected flattened labels with shape (N, 3), "
            f"got {labels_flat.shape}"
        )

    if not np.all(np.isfinite(labels_flat)):
        raise ValueError("Labels contain NaN or Inf.")

    if not np.all((labels_flat == 0) | (labels_flat == 1)):
        raise ValueError("Labels contain values other than zero and one.")

    row_sums = labels_flat.sum(axis=1)
    valid_mask = row_sums == 1
    padding_mask = row_sums == 0
    invalid_mask = ~(valid_mask | padding_mask)

    if np.any(invalid_mask):
        raise ValueError(
            "Labels contain "
            f"{int(invalid_mask.sum()):,} multi-hot or invalid rows."
        )

    y_true = np.full(len(labels_flat), -1, dtype=np.int8)
    y_true[valid_mask] = np.argmax(
        labels_flat[valid_mask],
        axis=1,
    ).astype(np.int8)

    pos_mask = valid_mask & (y_true != 0)
    neg_mask = valid_mask & (y_true == 0)

    return valid_mask, padding_mask, pos_mask, neg_mask


def load_model(model_path, flanking_size, device):
    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    model.apply_softmax = False

    return model, params


def extract_sampled_logits(
    model,
    params,
    dataset_path,
    out_path,
    device,
    max_negatives,
    random_seed,
    overwrite=False,
):
    dataset_path = Path(dataset_path)
    out_path = Path(out_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive")
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing cache: {out_path}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    pos_logits = []
    pos_probs = []
    pos_labels = []

    neg_logits_reservoir = np.empty(
        (max_negatives, 3),
        dtype=np.float32,
    )
    neg_probs_reservoir = np.empty(
        (max_negatives, 3),
        dtype=np.float32,
    )
    neg_labels_reservoir = np.empty(
        (max_negatives, 3),
        dtype=np.float32,
    )

    negatives_seen_total = 0
    negatives_kept = 0

    total_sequences_seen = 0
    total_positions_seen = 0
    valid_positions_seen = 0
    padding_positions_seen = 0
    positive_positions_seen = 0
    valid_nonsplice_positions_seen = 0

    with h5py.File(dataset_path, "r") as h5f:
        idxs = get_h5_indices(h5f)
        print("Dataset:", dataset_path, flush=True)
        print("Dataset shards:", idxs, flush=True)

        with torch.no_grad():
            for shard_idx in idxs:
                shard_sequence_count = int(
                    h5f[f"X{shard_idx}"].shape[0]
                )
                shard_sequences_seen = 0

                print(
                    f"Shard {shard_idx}: "
                    f"{shard_sequence_count:,} sequences",
                    flush=True,
                )

                batches = iter_complete_shard_batches(
                    h5f,
                    shard_idx,
                    params["BATCH_SIZE"],
                )

                for dnas, labels in batches:
                    batch_size = int(dnas.shape[0])
                    shard_sequences_seen += batch_size
                    total_sequences_seen += batch_size

                    dnas, labels = clip_input_context(
                        dnas,
                        labels,
                        params["CL"],
                    )

                    dnas = dnas.to(
                        device=device,
                        dtype=torch.float32,
                    )
                    labels = labels.to(
                        device=device,
                        dtype=torch.float32,
                    )

                    logits = model(dnas)
                    probs = torch.softmax(logits, dim=1)

                    if (
                        logits.ndim != 3
                        or logits.shape[1] != 3
                        or logits.shape[0] != labels.shape[0]
                        or logits.shape[2] != labels.shape[2]
                    ):
                        raise ValueError(
                            "Model-output/label shape mismatch: "
                            f"logits={tuple(logits.shape)}, "
                            f"labels={tuple(labels.shape)}"
                        )

                    logits_np = logits.detach().cpu().numpy()
                    probs_np = probs.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    logits_flat = np.transpose(
                        logits_np,
                        (0, 2, 1),
                    ).reshape(-1, 3)
                    probs_flat = np.transpose(
                        probs_np,
                        (0, 2, 1),
                    ).reshape(-1, 3)
                    labels_flat = np.transpose(
                        labels_np,
                        (0, 2, 1),
                    ).reshape(-1, 3)

                    (
                        valid_mask,
                        padding_mask,
                        pos_mask,
                        neg_mask,
                    ) = validate_and_partition_labels(labels_flat)

                    batch_raw = len(labels_flat)
                    batch_valid = int(valid_mask.sum())
                    batch_padding = int(padding_mask.sum())
                    batch_positive = int(pos_mask.sum())
                    batch_nonsplice = int(neg_mask.sum())

                    if (
                        batch_valid + batch_padding != batch_raw
                        or batch_positive + batch_nonsplice != batch_valid
                    ):
                        raise RuntimeError(
                            "Batch label-count reconciliation failed."
                        )

                    total_positions_seen += batch_raw
                    valid_positions_seen += batch_valid
                    padding_positions_seen += batch_padding
                    positive_positions_seen += batch_positive
                    valid_nonsplice_positions_seen += batch_nonsplice

                    if pos_mask.any():
                        pos_logits.append(
                            logits_flat[pos_mask].astype(np.float32)
                        )
                        pos_probs.append(
                            probs_flat[pos_mask].astype(np.float32)
                        )
                        pos_labels.append(
                            labels_flat[pos_mask].astype(np.float32)
                        )

                    neg_indices = np.flatnonzero(neg_mask)

                    if len(neg_indices) > 0:
                        n_neg = len(neg_indices)

                        if negatives_kept < max_negatives:
                            fill_n = min(
                                max_negatives - negatives_kept,
                                n_neg,
                            )
                            fill_indices = neg_indices[:fill_n]
                            destination = slice(
                                negatives_kept,
                                negatives_kept + fill_n,
                            )

                            neg_logits_reservoir[destination] = (
                                logits_flat[fill_indices].astype(
                                    np.float32
                                )
                            )
                            neg_probs_reservoir[destination] = (
                                probs_flat[fill_indices].astype(
                                    np.float32
                                )
                            )
                            neg_labels_reservoir[destination] = (
                                labels_flat[fill_indices].astype(
                                    np.float32
                                )
                            )

                            negatives_kept += fill_n
                            negatives_seen_total += fill_n

                            neg_indices = neg_indices[fill_n:]
                            n_neg = len(neg_indices)

                        if n_neg > 0:
                            global_positions = np.arange(
                                negatives_seen_total,
                                negatives_seen_total + n_neg,
                                dtype=np.int64,
                            )
                            slots = rng.integers(
                                0,
                                global_positions + 1,
                            )
                            replace_mask = slots < max_negatives

                            if replace_mask.any():
                                replace_slots = slots[replace_mask]
                                replace_indices = neg_indices[
                                    replace_mask
                                ]

                                neg_logits_reservoir[replace_slots] = (
                                    logits_flat[replace_indices].astype(
                                        np.float32
                                    )
                                )
                                neg_probs_reservoir[replace_slots] = (
                                    probs_flat[replace_indices].astype(
                                        np.float32
                                    )
                                )
                                neg_labels_reservoir[replace_slots] = (
                                    labels_flat[replace_indices].astype(
                                        np.float32
                                    )
                                )

                            negatives_seen_total += n_neg

                excluded_count = len(
                    EXCLUDED_ROWS_BY_SHARD.get(
                        shard_idx,
                        frozenset(),
                    )
                )
                expected_sequences_seen = (
                    shard_sequence_count - excluded_count
                )

                if shard_sequences_seen != expected_sequences_seen:
                    raise RuntimeError(
                        f"Shard {shard_idx}: evaluated "
                        f"{shard_sequences_seen:,}; expected "
                        f"{expected_sequences_seen:,} after "
                        f"{excluded_count} intentional exclusion(s)."
                    )

                print(
                    f"Shard {shard_idx}: retained "
                    f"{shard_sequences_seen:,} of "
                    f"{shard_sequence_count:,} sequences "
                    f"after {excluded_count} intentional exclusion(s)",
                    flush=True,
                )

    expected_sensitivity_census = {
        "total_sequences_seen": 18483,
        "total_positions_seen": 92415000,
        "valid_positions_seen": 89007976,
        "padding_positions_seen": 3407024,
        "positive_positions_seen": 25902,
        "valid_nonsplice_positions_seen": 88982074,
    }

    observed_sensitivity_census = {
        "total_sequences_seen": total_sequences_seen,
        "total_positions_seen": total_positions_seen,
        "valid_positions_seen": valid_positions_seen,
        "padding_positions_seen": padding_positions_seen,
        "positive_positions_seen": positive_positions_seen,
        "valid_nonsplice_positions_seen":
            valid_nonsplice_positions_seen,
    }

    if observed_sensitivity_census != expected_sensitivity_census:
        raise RuntimeError(
            "Duplicate-excluded validation census mismatch: "
            f"observed={observed_sensitivity_census}, "
            f"expected={expected_sensitivity_census}"
        )

    if not pos_logits:
        raise RuntimeError("No positive splice positions found.")

    if total_positions_seen != valid_positions_seen + padding_positions_seen:
        raise RuntimeError(
            "Raw-position reconciliation failed."
        )
    if (
        valid_positions_seen
        != positive_positions_seen + valid_nonsplice_positions_seen
    ):
        raise RuntimeError(
            "Valid-position reconciliation failed."
        )
    if negatives_seen_total != valid_nonsplice_positions_seen:
        raise RuntimeError(
            "Negative-reservoir population reconciliation failed."
        )

    expected_negatives_kept = min(
        max_negatives,
        valid_nonsplice_positions_seen,
    )
    if negatives_kept != expected_negatives_kept:
        raise RuntimeError(
            "Sampled-negative count reconciliation failed."
        )

    pos_logits = np.concatenate(pos_logits, axis=0)
    pos_probs = np.concatenate(pos_probs, axis=0)
    pos_labels = np.concatenate(pos_labels, axis=0)

    neg_logits = neg_logits_reservoir[:negatives_kept]
    neg_probs = neg_probs_reservoir[:negatives_kept]
    neg_labels = neg_labels_reservoir[:negatives_kept]

    logits_sample = np.concatenate(
        [pos_logits, neg_logits],
        axis=0,
    )
    probs_sample = np.concatenate(
        [pos_probs, neg_probs],
        axis=0,
    )
    labels_sample = np.concatenate(
        [pos_labels, neg_labels],
        axis=0,
    )

    perm = rng.permutation(len(labels_sample))
    logits_sample = logits_sample[perm]
    probs_sample = probs_sample[perm]
    labels_sample = labels_sample[perm]

    sampled_row_sums = labels_sample.sum(axis=1)
    if not np.all(sampled_row_sums == 1):
        raise RuntimeError(
            "Final cache contains padding or invalid label rows."
        )

    if (
        not np.all(np.isfinite(logits_sample))
        or not np.all(np.isfinite(probs_sample))
    ):
        raise RuntimeError(
            "Final cache contains NaN or Inf predictions."
        )

    probs_from_logits = softmax_np(logits_sample, axis=1)
    max_abs_diff = float(
        np.max(np.abs(probs_from_logits - probs_sample))
    )
    prob_row_sum_max_error = float(
        np.max(np.abs(probs_sample.sum(axis=1) - 1.0))
    )

    y = np.argmax(labels_sample, axis=1)
    class_counts = np.bincount(y, minlength=3)

    if (
        int(class_counts[0]) != negatives_kept
        or int(class_counts[1] + class_counts[2])
        != positive_positions_seen
    ):
        raise RuntimeError(
            "Final cache class-count reconciliation failed."
        )

    negative_weight = (
        valid_nonsplice_positions_seen / negatives_kept
    )

    print("\nExtraction summary")
    print("==================")
    print("Cache schema version:", CACHE_SCHEMA_VERSION)
    print("Total sequences seen:", total_sequences_seen)
    print("Raw positions seen:", total_positions_seen)
    print("Valid positions seen:", valid_positions_seen)
    print("Padding positions excluded:", padding_positions_seen)
    print("Positive positions seen:", positive_positions_seen)
    print(
        "Valid non-splice positions seen:",
        valid_nonsplice_positions_seen,
    )
    print("Sampled valid non-splice positions:", negatives_kept)
    print("Valid non-splice weight:", negative_weight)
    print("Total sampled positions:", len(labels_sample))
    print("Class counts:")
    print("  nonsplice:", int(class_counts[0]))
    print("  acceptor:", int(class_counts[1]))
    print("  donor:", int(class_counts[2]))
    print(
        "Max abs diff softmax(logits) vs probs:",
        max_abs_diff,
    )
    print(
        "Probability row-sum max error:",
        prob_row_sum_max_error,
    )

    temporary_path = out_path.with_suffix(
        out_path.suffix + ".tmp"
    )

    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                logits_sample=logits_sample.astype(np.float32),
                probs_sample=probs_sample.astype(np.float32),
                labels_sample=labels_sample.astype(np.float32),
                cache_schema_version=np.array(
                    [CACHE_SCHEMA_VERSION],
                    dtype=np.int64,
                ),
                dataset_path=np.array([str(dataset_path)]),
                total_shards_seen=np.array(
                    [len(idxs)],
                    dtype=np.int64,
                ),
                total_sequences_seen=np.array(
                    [total_sequences_seen],
                    dtype=np.int64,
                ),
                total_positions_seen=np.array(
                    [total_positions_seen],
                    dtype=np.int64,
                ),
                valid_positions_seen=np.array(
                    [valid_positions_seen],
                    dtype=np.int64,
                ),
                padding_positions_seen=np.array(
                    [padding_positions_seen],
                    dtype=np.int64,
                ),
                positive_positions_seen=np.array(
                    [positive_positions_seen],
                    dtype=np.int64,
                ),
                valid_nonsplice_positions_seen=np.array(
                    [valid_nonsplice_positions_seen],
                    dtype=np.int64,
                ),
                # Backward-compatible alias used by downstream scripts.
                negatives_seen_total=np.array(
                    [valid_nonsplice_positions_seen],
                    dtype=np.int64,
                ),
                sampled_negatives=np.array(
                    [negatives_kept],
                    dtype=np.int64,
                ),
                max_negatives=np.array(
                    [max_negatives],
                    dtype=np.int64,
                ),
                random_seed=np.array(
                    [random_seed],
                    dtype=np.int64,
                ),
                all_sequences_included=np.array(
                    [0],
                    dtype=np.int8,
                ),
                sensitivity_exclusion_applied=np.array(
                    [1],
                    dtype=np.int8,
                ),
                excluded_sequence_count=np.array(
                    [len(EXCLUDED_ROW_PAIRS)],
                    dtype=np.int64,
                ),
                excluded_shard_indices=np.array(
                    [pair[0] for pair in EXCLUDED_ROW_PAIRS],
                    dtype=np.int64,
                ),
                excluded_row_indices=np.array(
                    [pair[1] for pair in EXCLUDED_ROW_PAIRS],
                    dtype=np.int64,
                ),
                padding_excluded_from_sampling=np.array(
                    [1],
                    dtype=np.int8,
                ),
                max_abs_diff_softmax_logits_probs=np.array(
                    [max_abs_diff],
                    dtype=np.float32,
                ),
                prob_row_sum_max_error=np.array(
                    [prob_row_sum_max_error],
                    dtype=np.float32,
                ),
                valid_nonsplice_weight=np.array(
                    [negative_weight],
                    dtype=np.float64,
                ),
            )

        temporary_path.replace(out_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print("Saved:", out_path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--max-negatives", type=int, default=500000)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "both"),
        default="both",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output cache.",
    )
    args = parser.parse_args()

    if args.split != "validation":
        raise RuntimeError(
            "This TCBB v9 duplicate-sensitivity extractor is "
            "validation-only. Use --split validation."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    if args.split in ("validation", "both"):
        selected.append(
            (
                Path(args.validation_dataset),
                out_dir / "validation_sampled_logits.npz",
            )
        )
    if args.split in ("test", "both"):
        selected.append(
            (
                Path(args.test_dataset),
                out_dir / "test_sampled_logits.npz",
            )
        )

    for dataset_path, out_path in selected:
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing cache: {out_path}"
            )

    device = torch.device("cpu")
    model, params = load_model(
        model_path=args.model,
        flanking_size=args.flanking_size,
        device=device,
    )

    print("\nModel loaded with apply_softmax =", model.apply_softmax)
    print("Device:", device)
    print("Batch size:", params["BATCH_SIZE"])
    print("Split selection:", args.split)

    for dataset_path, out_path in selected:
        extract_sampled_logits(
            model=model,
            params=params,
            dataset_path=dataset_path,
            out_path=out_path,
            device=device,
            max_negatives=args.max_negatives,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
