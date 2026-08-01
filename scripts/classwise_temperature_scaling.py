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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def apply_global_temperature(probs, temperature):
    eps = 1e-12
    logits_proxy = np.log(np.clip(probs, eps, 1.0))
    return softmax_np(logits_proxy / temperature, axis=1)


def apply_binary_temperature(p, temperature):
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return sigmoid(logit / temperature)


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


def binary_metrics(p, y, n_bins=15):
    eps = 1e-12
    p_clip = np.clip(p, eps, 1.0 - eps)
    return {
        "ece": binary_ece(p_clip, y, n_bins=n_bins),
        "nll": log_loss(y, p_clip, labels=[0, 1]),
        "brier": brier_score_loss(y, p_clip),
    }


def extract_or_load_sampled_predictions(
    cache_path,
    model_path,
    dataset_path,
    flanking_size,
    random_seed,
    max_negatives,
):
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Loading cache: {cache_path}", flush=True)
        z = np.load(cache_path)
        return z["probs_sample"], z["labels_sample"]

    print(f"Creating cache: {cache_path}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    print("Loading checkpoint:", model_path, flush=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Checkpoint loaded.", flush=True)

    params["RANDOM_SEED"] = random_seed

    pos_probs = []
    pos_labels = []

    neg_probs_reservoir = np.empty((max_negatives, 3), dtype=np.float32)
    neg_labels_reservoir = np.empty((max_negatives, 3), dtype=np.float32)
    negatives_seen_total = 0
    negatives_kept = 0

    total_positions_seen = 0
    positive_positions_seen = 0

    with h5py.File(dataset_path, "r") as h5f:
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

                        if negatives_kept < max_negatives:
                            fill_n = min(max_negatives - negatives_kept, n_neg)
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

                        if n_neg > 0:
                            global_positions = np.arange(
                                negatives_seen_total,
                                negatives_seen_total + n_neg,
                                dtype=np.int64,
                            )

                            slots = rng.integers(0, global_positions + 1)
                            replace_mask = slots < max_negatives

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

    perm = rng.permutation(len(probs_sample))
    probs_sample = probs_sample[perm]
    labels_sample = labels_sample[perm]

    print("Total positions seen:", total_positions_seen, flush=True)
    print("Positive positions:", len(pos_probs), flush=True)
    print("Positive positions seen:", positive_positions_seen, flush=True)
    print("Total negative positions seen:", negatives_seen_total, flush=True)
    print("Sampled negative positions:", len(neg_probs), flush=True)
    print("Total sampled positions:", len(probs_sample), flush=True)

    np.savez_compressed(
        cache_path,
        probs_sample=probs_sample.astype(np.float32),
        labels_sample=labels_sample.astype(np.float32),
        total_positions_seen=np.array([total_positions_seen]),
        positive_positions_seen=np.array([positive_positions_seen]),
        negatives_seen_total=np.array([negatives_seen_total]),
        sampled_negatives=np.array([len(neg_probs)]),
        random_seed=np.array([random_seed]),
        max_negatives=np.array([max_negatives]),
    )

    print(f"Saved cache: {cache_path}", flush=True)
    return probs_sample, labels_sample


def fit_temperature_grid(p_val, y_val, grid):
    best_t = None
    best_nll = float("inf")

    for t in grid:
        p_cal = apply_binary_temperature(p_val, t)
        nll = log_loss(y_val, np.clip(p_cal, 1e-12, 1.0 - 1e-12), labels=[0, 1])

        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return best_t, best_nll


def evaluate_split(name, probs, labels, global_t, t_acceptor, t_donor, n_bins):
    y_acc = labels[:, 1]
    y_don = labels[:, 2]

    p_acc_uncal = probs[:, 1]
    p_don_uncal = probs[:, 2]

    probs_global = apply_global_temperature(probs, global_t)
    p_acc_global = probs_global[:, 1]
    p_don_global = probs_global[:, 2]

    p_acc_class = apply_binary_temperature(p_acc_uncal, t_acceptor)
    p_don_class = apply_binary_temperature(p_don_uncal, t_donor)

    rows = []

    for method, p_acc, p_don in [
        ("uncalibrated", p_acc_uncal, p_don_uncal),
        (f"global_T_{global_t}", p_acc_global, p_don_global),
        (f"classwise_T_acc_{t_acceptor}_don_{t_donor}", p_acc_class, p_don_class),
    ]:
        acc = binary_metrics(p_acc, y_acc, n_bins=n_bins)
        don = binary_metrics(p_don, y_don, n_bins=n_bins)

        rows.append({
            "split": name,
            "method": method,
            "acceptor_ece": acc["ece"],
            "acceptor_nll": acc["nll"],
            "acceptor_brier": acc["brier"],
            "donor_ece": don["ece"],
            "donor_nll": don["nll"],
            "donor_brier": don["brier"],
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--max-negatives", type=int, default=500000)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--global-temperature", type=float, default=1.1)
    parser.add_argument("--grid-min", type=float, default=0.5)
    parser.add_argument("--grid-max", type=float, default=3.0)
    parser.add_argument("--grid-step", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_cache = out_dir / "validation_sampled_predictions.npz"
    test_cache = out_dir / "test_sampled_predictions.npz"

    val_probs, val_labels = extract_or_load_sampled_predictions(
        cache_path=val_cache,
        model_path=args.model,
        dataset_path=args.validation_dataset,
        flanking_size=args.flanking_size,
        random_seed=args.random_seed,
        max_negatives=args.max_negatives,
    )

    test_probs, test_labels = extract_or_load_sampled_predictions(
        cache_path=test_cache,
        model_path=args.model,
        dataset_path=args.test_dataset,
        flanking_size=args.flanking_size,
        random_seed=args.random_seed,
        max_negatives=args.max_negatives,
    )

    grid = np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step)

    t_acceptor, val_acc_nll = fit_temperature_grid(
        val_probs[:, 1],
        val_labels[:, 1],
        grid,
    )

    t_donor, val_don_nll = fit_temperature_grid(
        val_probs[:, 2],
        val_labels[:, 2],
        grid,
    )

    print("Best validation-fitted classwise temperatures:")
    print("T_acceptor:", t_acceptor, "validation acceptor NLL:", val_acc_nll)
    print("T_donor:", t_donor, "validation donor NLL:", val_don_nll)

    rows = []
    rows.extend(
        evaluate_split(
            "validation_sampled",
            val_probs,
            val_labels,
            args.global_temperature,
            t_acceptor,
            t_donor,
            args.n_bins,
        )
    )
    rows.extend(
        evaluate_split(
            "test_sampled",
            test_probs,
            test_labels,
            args.global_temperature,
            t_acceptor,
            t_donor,
            args.n_bins,
        )
    )

    summary_path = out_dir / "classwise_temperature_summary.csv"

    with open(summary_path, "w") as f:
        header = [
            "split",
            "method",
            "acceptor_ece",
            "acceptor_nll",
            "acceptor_brier",
            "donor_ece",
            "donor_nll",
            "donor_brier",
        ]
        f.write(",".join(header) + "\n")

        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    txt_path = out_dir / "classwise_temperature_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Class-specific temperature scaling\n")
        f.write("==================================\n")
        f.write(f"model: {args.model}\n")
        f.write(f"validation_dataset: {args.validation_dataset}\n")
        f.write(f"test_dataset: {args.test_dataset}\n")
        f.write(f"global_temperature_baseline: {args.global_temperature}\n")
        f.write(f"T_acceptor: {t_acceptor}\n")
        f.write(f"T_donor: {t_donor}\n")
        f.write(f"temperature_grid: {args.grid_min} to {args.grid_max} step {args.grid_step}\n")
        f.write(f"random_seed: {args.random_seed}\n")
        f.write(f"max_negatives: {args.max_negatives}\n")
        f.write(f"n_bins: {args.n_bins}\n\n")

        for row in rows:
            f.write(str(row) + "\n")

    print("\nWrote:")
    print(summary_path)
    print(txt_path)
    print("\nCSV preview:")
    print(summary_path.read_text())


if __name__ == "__main__":
    main()