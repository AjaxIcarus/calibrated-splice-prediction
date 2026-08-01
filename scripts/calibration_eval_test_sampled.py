import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import log_loss, brier_score_loss

from openspliceai.train.train import initialize_model_and_optim
from openspliceai.train_base.utils import load_data_from_shard, clip_datapoints
from openspliceai.constants import CL_max


def get_h5_indices(h5f):
    x_keys = [k for k in h5f.keys() if k.startswith("X")]
    return np.array(sorted([int(k[1:]) for k in x_keys]))


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def apply_temperature_to_probs(probs, temperature):
    """
    Applies temperature scaling to already-softmaxed probabilities.

    Since we only have probabilities here, recover log-probabilities:
        logits_proxy = log(probs)
    Then apply:
        softmax(logits_proxy / T)
    """
    eps = 1e-12
    logits_proxy = np.log(np.clip(probs, eps, 1.0))
    return softmax_np(logits_proxy / temperature, axis=1)


def binary_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    if n == 0:
        return np.nan

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_conf = float(probs[mask].mean())
        bin_freq = float(labels[mask].mean())
        ece += (count / n) * abs(bin_freq - bin_conf)

    return ece


def multiclass_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)

    if n == 0:
        return np.nan

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return ece


def reliability_bins_binary(probs, labels, n_bins=15):
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = int(mask.sum())

        if count == 0:
            rows.append((i, lo, hi, 0, np.nan, np.nan, np.nan))
            continue

        conf = float(probs[mask].mean())
        freq = float(labels[mask].mean())
        gap = abs(freq - conf)
        rows.append((i, lo, hi, count, conf, freq, gap))

    return rows


def reliability_bins_multiclass(probs, labels, n_bins=15):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())

        if count == 0:
            rows.append((i, lo, hi, 0, np.nan, np.nan, np.nan))
            continue

        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        gap = abs(bin_acc - bin_conf)
        rows.append((i, lo, hi, count, bin_conf, bin_acc, gap))

    return rows


def write_binary_bins(path, rows):
    with open(path, "w") as f:
        f.write("bin,lo,hi,count,mean_confidence,empirical_frequency,abs_gap\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def write_multiclass_bins(path, rows):
    with open(path, "w") as f:
        f.write("bin,lo,hi,count,mean_confidence,empirical_accuracy,abs_gap\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def compute_metrics(probs, labels):
    eps = 1e-12
    y_true = np.argmax(labels, axis=1)

    probs_clip = np.clip(probs, eps, 1.0 - eps)

    out = {}
    out["multiclass_ece"] = multiclass_ece(probs, labels)
    out["multiclass_nll"] = log_loss(y_true, probs_clip, labels=[0, 1, 2])
    out["multiclass_brier"] = float(np.mean(np.sum((probs - labels) ** 2, axis=1)))

    for name, class_idx in [("acceptor", 1), ("donor", 2)]:
        p = probs[:, class_idx]
        y = labels[:, class_idx]

        out[f"{name}_ece"] = binary_ece(p, y)
        out[f"{name}_nll"] = log_loss(y, np.clip(p, eps, 1.0 - eps), labels=[0, 1])
        out[f"{name}_brier"] = brier_score_loss(y, p)

    return out


def assert_probabilities(probs, name):
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{name} contains NaN or Inf.")

    row_sums = probs.sum(axis=1)
    max_error = np.max(np.abs(row_sums - 1.0))

    print(f"{name} probability row-sum max error: {max_error}", flush=True)

    if max_error > 1e-3:
        raise ValueError(
            f"{name} does not look like probabilities. "
            f"Max row-sum error: {max_error}. "
            "Check whether model output is logits instead of softmax probabilities."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--max-negatives", type=int, default=500000)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=args.flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    print("Loading checkpoint:", args.model, flush=True)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Checkpoint loaded.", flush=True)

    params["RANDOM_SEED"] = args.random_seed

    pos_probs = []
    pos_labels = []

    # Reservoir for uniform sampling of non-splice positions across the full test set.
    neg_probs_reservoir = np.empty((args.max_negatives, 3), dtype=np.float32)
    neg_labels_reservoir = np.empty((args.max_negatives, 3), dtype=np.float32)
    negatives_seen_total = 0
    negatives_kept = 0

    total_positions_seen = 0
    positive_positions_seen = 0

    with h5py.File(args.dataset, "r") as h5f:
        idxs = get_h5_indices(h5f)
        print("Dataset shards:", idxs, flush=True)

        with torch.no_grad():
            for shard_idx in idxs:
                print(f"Shard {shard_idx}", flush=True)

                loader = load_data_from_shard(
                    h5f,
                    shard_idx,
                    device,
                    params["BATCH_SIZE"],
                    params,
                    shuffle=False,
                )

                for batch in loader:
                    DNAs, labels = batch[0].to(device), batch[1].to(device)

                    DNAs, labels = clip_datapoints(
                        DNAs,
                        labels,
                        params["CL"],
                        CL_max,
                        params["N_GPUS"],
                    )

                    DNAs = DNAs.to(torch.float32).to(device)
                    labels = labels.to(torch.float32).to(device)

                    probs = model(DNAs)

                    probs_np = probs.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    # Expected shape: B x 3 x L
                    # Convert to B*L x 3
                    probs_flat = np.transpose(probs_np, (0, 2, 1)).reshape(-1, 3)
                    labels_flat = np.transpose(labels_np, (0, 2, 1)).reshape(-1, 3)

                    total_positions_seen += len(labels_flat)

                    y_true = np.argmax(labels_flat, axis=1)
                    pos_mask = y_true != 0
                    neg_mask = y_true == 0

                    if pos_mask.any():
                        pos_probs.append(probs_flat[pos_mask].astype(np.float32))
                        pos_labels.append(labels_flat[pos_mask].astype(np.float32))
                        positive_positions_seen += int(pos_mask.sum())

                    neg_indices = np.where(neg_mask)[0]

                    if len(neg_indices) > 0:
                        n_neg = len(neg_indices)

                        # Fill reservoir first.
                        if negatives_kept < args.max_negatives:
                            fill_n = min(args.max_negatives - negatives_kept, n_neg)
                            fill_indices = neg_indices[:fill_n]

                            neg_probs_reservoir[
                                negatives_kept:negatives_kept + fill_n
                            ] = probs_flat[fill_indices].astype(np.float32)

                            neg_labels_reservoir[
                                negatives_kept:negatives_kept + fill_n
                            ] = labels_flat[fill_indices].astype(np.float32)

                            negatives_kept += fill_n
                            negatives_seen_total += fill_n

                            neg_indices = neg_indices[fill_n:]
                            n_neg = len(neg_indices)

                        # Reservoir sampling for remaining negatives.
                        if n_neg > 0:
                            global_positions = np.arange(
                                negatives_seen_total,
                                negatives_seen_total + n_neg,
                                dtype=np.int64,
                            )

                            # For item i, choose random slot in [0, i].
                            # Replace only if slot < reservoir size.
                            slots = rng.integers(0, global_positions + 1)
                            replace_mask = slots < args.max_negatives

                            if replace_mask.any():
                                replace_slots = slots[replace_mask]
                                replace_indices = neg_indices[replace_mask]

                                neg_probs_reservoir[replace_slots] = probs_flat[
                                    replace_indices
                                ].astype(np.float32)

                                neg_labels_reservoir[replace_slots] = labels_flat[
                                    replace_indices
                                ].astype(np.float32)

                            negatives_seen_total += n_neg

    if len(pos_probs) == 0:
        raise RuntimeError("No positive splice positions found.")

    pos_probs = np.concatenate(pos_probs, axis=0)
    pos_labels = np.concatenate(pos_labels, axis=0)

    neg_probs = neg_probs_reservoir[:negatives_kept]
    neg_labels = neg_labels_reservoir[:negatives_kept]

    probs_sample = np.concatenate([pos_probs, neg_probs], axis=0)
    labels_sample = np.concatenate([pos_labels, neg_labels], axis=0)

    print("Total positions seen:", total_positions_seen, flush=True)
    print("Positive positions:", len(pos_probs), flush=True)
    print("Positive positions seen:", positive_positions_seen, flush=True)
    print("Total negative positions seen:", negatives_seen_total, flush=True)
    print("Sampled negative positions:", len(neg_probs), flush=True)
    print("Total sampled positions:", len(probs_sample), flush=True)

    # Shuffle final sampled set.
    perm = rng.permutation(len(probs_sample))
    probs_sample = probs_sample[perm]
    labels_sample = labels_sample[perm]

    assert_probabilities(probs_sample, "uncalibrated probs")

    uncal = compute_metrics(probs_sample, labels_sample)

    calibrated_probs = apply_temperature_to_probs(probs_sample, args.temperature)
    assert_probabilities(calibrated_probs, "temperature-scaled probs")

    cal = compute_metrics(calibrated_probs, labels_sample)

    # Reliability bins.
    write_multiclass_bins(
        out_dir / "uncalibrated_multiclass_reliability_bins.csv",
        reliability_bins_multiclass(probs_sample, labels_sample, n_bins=args.n_bins),
    )
    write_binary_bins(
        out_dir / "uncalibrated_acceptor_reliability_bins.csv",
        reliability_bins_binary(
            probs_sample[:, 1], labels_sample[:, 1], n_bins=args.n_bins
        ),
    )
    write_binary_bins(
        out_dir / "uncalibrated_donor_reliability_bins.csv",
        reliability_bins_binary(
            probs_sample[:, 2], labels_sample[:, 2], n_bins=args.n_bins
        ),
    )

    write_multiclass_bins(
        out_dir / "temperature_scaled_multiclass_reliability_bins.csv",
        reliability_bins_multiclass(calibrated_probs, labels_sample, n_bins=args.n_bins),
    )
    write_binary_bins(
        out_dir / "temperature_scaled_acceptor_reliability_bins.csv",
        reliability_bins_binary(
            calibrated_probs[:, 1], labels_sample[:, 1], n_bins=args.n_bins
        ),
    )
    write_binary_bins(
        out_dir / "temperature_scaled_donor_reliability_bins.csv",
        reliability_bins_binary(
            calibrated_probs[:, 2], labels_sample[:, 2], n_bins=args.n_bins
        ),
    )

    summary_path = out_dir / "test_temperature_summary.txt"

    with open(summary_path, "w") as f:
        f.write("Sampled test calibration evaluation\n")
        f.write("===================================\n")
        f.write(f"model: {args.model}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"temperature_from_validation: {args.temperature}\n")
        f.write(f"random_seed: {args.random_seed}\n")
        f.write(f"max_negatives: {args.max_negatives}\n")
        f.write(f"n_bins: {args.n_bins}\n")
        f.write(f"total_positions_seen: {total_positions_seen}\n")
        f.write(f"positive_positions: {len(pos_probs)}\n")
        f.write(f"positive_positions_seen: {positive_positions_seen}\n")
        f.write(f"total_negative_positions_seen: {negatives_seen_total}\n")
        f.write(f"sampled_negative_positions: {len(neg_probs)}\n")
        f.write(f"total_sampled_positions: {len(probs_sample)}\n")

        f.write("\nUncalibrated sampled test metrics:\n")
        for k, v in uncal.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTemperature-scaled sampled test metrics:\n")
        for k, v in cal.items():
            f.write(f"{k}: {v}\n")

        f.write("\nDelta calibrated_minus_uncalibrated:\n")
        for k in uncal:
            f.write(f"{k}: {cal[k] - uncal[k]}\n")

    print(summary_path.read_text(), flush=True)


if __name__ == "__main__":
    main()