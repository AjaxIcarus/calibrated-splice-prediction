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
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))
    return softmax_np(logits / temperature, axis=1)


def binary_ece(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

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
        ece += (count / len(probs)) * abs(bin_freq - bin_conf)

    return ece


def multiclass_ece(probs, labels, n_bins=15):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

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
        ece += (count / len(conf)) * abs(bin_acc - bin_conf)

    return ece


def compute_metrics(probs, labels):
    eps = 1e-12
    y_true = np.argmax(labels, axis=1)

    probs_clip = np.clip(probs, eps, 1.0 - eps)

    out = {}
    out["multiclass_ece"] = multiclass_ece(probs, labels)
    out["multiclass_nll"] = log_loss(y_true, probs_clip, labels=[0, 1, 2])
    out["multiclass_brier"] = np.mean(np.sum((probs - labels) ** 2, axis=1))

    for name, class_idx in [("acceptor", 1), ("donor", 2)]:
        p = probs[:, class_idx]
        y = labels[:, class_idx]

        out[f"{name}_ece"] = binary_ece(p, y)
        out[f"{name}_nll"] = log_loss(y, np.clip(p, eps, 1.0 - eps), labels=[0, 1])
        out[f"{name}_brier"] = brier_score_loss(y, p)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--max-negatives", type=int, default=500000)
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
    params["RANDOM_SEED"] = args.random_seed

    pos_probs = []
    pos_labels = []
    neg_probs = []
    neg_labels = []
    negatives_seen = 0

    with h5py.File(args.dataset, "r") as h5f:
        idxs = get_h5_indices(h5f)
        print("Validation shards:", idxs, flush=True)

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

                    # B x 3 x L -> B*L x 3
                    probs_flat = np.transpose(probs_np, (0, 2, 1)).reshape(-1, 3)
                    labels_flat = np.transpose(labels_np, (0, 2, 1)).reshape(-1, 3)

                    y_true = np.argmax(labels_flat, axis=1)
                    pos_mask = y_true != 0
                    neg_mask = y_true == 0

                    if pos_mask.any():
                        pos_probs.append(probs_flat[pos_mask])
                        pos_labels.append(labels_flat[pos_mask])

                    neg_indices = np.where(neg_mask)[0]
                    remaining = args.max_negatives - negatives_seen

                    if remaining > 0 and len(neg_indices) > 0:
                        take = min(remaining, len(neg_indices), 5000)
                        chosen = rng.choice(neg_indices, size=take, replace=False)
                        neg_probs.append(probs_flat[chosen])
                        neg_labels.append(labels_flat[chosen])
                        negatives_seen += take

    pos_probs = np.concatenate(pos_probs, axis=0)
    pos_labels = np.concatenate(pos_labels, axis=0)

    neg_probs = np.concatenate(neg_probs, axis=0)
    neg_labels = np.concatenate(neg_labels, axis=0)

    probs_sample = np.concatenate([pos_probs, neg_probs], axis=0)
    labels_sample = np.concatenate([pos_labels, neg_labels], axis=0)

    print("Positive positions:", len(pos_probs), flush=True)
    print("Sampled negative positions:", len(neg_probs), flush=True)
    print("Total sampled positions:", len(probs_sample), flush=True)

    # Shuffle sample
    perm = rng.permutation(len(probs_sample))
    probs_sample = probs_sample[perm]
    labels_sample = labels_sample[perm]

    uncal = compute_metrics(probs_sample, labels_sample)

    y_true = np.argmax(labels_sample, axis=1)

    temperatures = np.concatenate([
        np.linspace(0.25, 1.0, 16),
        np.linspace(1.1, 5.0, 40),
    ])

    best_t = None
    best_nll = float("inf")
    rows = []

    for t in temperatures:
        cal_probs = apply_temperature_to_probs(probs_sample, t)
        nll = log_loss(
            y_true,
            np.clip(cal_probs, 1e-12, 1.0 - 1e-12),
            labels=[0, 1, 2],
        )
        rows.append((t, nll))
        print(f"T={t:.4f}, sampled multiclass NLL={nll:.8f}", flush=True)

        if nll < best_nll:
            best_nll = nll
            best_t = t

    cal_probs = apply_temperature_to_probs(probs_sample, best_t)
    cal = compute_metrics(cal_probs, labels_sample)

    with open(out_dir / "temperature_grid.csv", "w") as f:
        f.write("temperature,sampled_multiclass_nll\n")
        for t, nll in rows:
            f.write(f"{t},{nll}\n")

    with open(out_dir / "temperature_summary.txt", "w") as f:
        f.write("Sampled temperature scaling on validation\n")
        f.write("=========================================\n")
        f.write(f"positive_positions: {len(pos_probs)}\n")
        f.write(f"sampled_negative_positions: {len(neg_probs)}\n")
        f.write(f"best_temperature: {best_t}\n")
        f.write(f"best_sampled_multiclass_nll: {best_nll}\n")

        f.write("\nUncalibrated sampled metrics:\n")
        for k, v in uncal.items():
            f.write(f"{k}: {v}\n")

        f.write("\nCalibrated sampled metrics:\n")
        for k, v in cal.items():
            f.write(f"{k}: {v}\n")

    print((out_dir / "temperature_summary.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()