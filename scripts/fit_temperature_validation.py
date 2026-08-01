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


def expected_calibration_error(confidences, correct, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (count / len(confidences)) * abs(bin_acc - bin_conf)

    return ece


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


def compute_metrics(probs_flat, labels_flat):
    eps = 1e-12

    y_true = np.argmax(labels_flat, axis=1)
    y_pred = np.argmax(probs_flat, axis=1)
    conf = np.max(probs_flat, axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    probs_clip = np.clip(probs_flat, eps, 1.0 - eps)

    multiclass_ece = expected_calibration_error(conf, correct, n_bins=15)
    multiclass_nll = log_loss(y_true, probs_clip, labels=[0, 1, 2])
    multiclass_brier = np.mean(np.sum((probs_flat - labels_flat) ** 2, axis=1))

    acceptor_probs = probs_flat[:, 1]
    donor_probs = probs_flat[:, 2]
    acceptor_labels = labels_flat[:, 1]
    donor_labels = labels_flat[:, 2]

    acceptor_ece = binary_ece(acceptor_probs, acceptor_labels, n_bins=15)
    donor_ece = binary_ece(donor_probs, donor_labels, n_bins=15)

    acceptor_nll = log_loss(
        acceptor_labels,
        np.clip(acceptor_probs, eps, 1.0 - eps),
        labels=[0, 1],
    )
    donor_nll = log_loss(
        donor_labels,
        np.clip(donor_probs, eps, 1.0 - eps),
        labels=[0, 1],
    )

    acceptor_brier = brier_score_loss(acceptor_labels, acceptor_probs)
    donor_brier = brier_score_loss(donor_labels, donor_probs)

    return {
        "multiclass_ece": multiclass_ece,
        "multiclass_nll": multiclass_nll,
        "multiclass_brier": multiclass_brier,
        "acceptor_ece": acceptor_ece,
        "acceptor_nll": acceptor_nll,
        "acceptor_brier": acceptor_brier,
        "donor_ece": donor_ece,
        "donor_nll": donor_nll,
        "donor_brier": donor_brier,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=args.flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    print("Loading checkpoint:", args.model)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    params["RANDOM_SEED"] = args.random_seed

    all_probs = []
    all_labels = []

    with h5py.File(args.dataset, "r") as h5f:
        idxs = get_h5_indices(h5f)
        if args.max_shards is not None:
            idxs = idxs[: args.max_shards]

        print("Evaluating shards:", idxs)

        with torch.no_grad():
            for shard_idx in idxs:
                print(f"Shard {shard_idx}")
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

                    all_probs.append(probs.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    probs_flat = np.transpose(probs, (0, 2, 1)).reshape(-1, 3)
    labels_flat = np.transpose(labels, (0, 2, 1)).reshape(-1, 3)

    print("probs_flat:", probs_flat.shape)
    print("labels_flat:", labels_flat.shape)

    y_true = np.argmax(labels_flat, axis=1)

    uncal_metrics = compute_metrics(probs_flat, labels_flat)

    temperatures = np.concatenate([
        np.linspace(0.25, 1.0, 16),
        np.linspace(1.1, 5.0, 40),
    ])

    rows = []
    best_t = None
    best_nll = float("inf")

    for t in temperatures:
        calibrated_probs = apply_temperature_to_probs(probs_flat, t)
        nll = log_loss(y_true, np.clip(calibrated_probs, 1e-12, 1.0 - 1e-12), labels=[0, 1, 2])
        rows.append((t, nll))

        if nll < best_nll:
            best_nll = nll
            best_t = t

    calibrated_probs = apply_temperature_to_probs(probs_flat, best_t)
    cal_metrics = compute_metrics(calibrated_probs, labels_flat)

    with open(out_dir / "temperature_grid.csv", "w") as f:
        f.write("temperature,multiclass_nll\n")
        for t, nll in rows:
            f.write(f"{t},{nll}\n")

    with open(out_dir / "temperature_summary.txt", "w") as f:
        f.write("Temperature scaling on validation\n")
        f.write("=================================\n")
        f.write(f"best_temperature: {best_t}\n")
        f.write(f"best_validation_multiclass_nll: {best_nll}\n")
        f.write("\nUncalibrated metrics:\n")
        for k, v in uncal_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nCalibrated metrics:\n")
        for k, v in cal_metrics.items():
            f.write(f"{k}: {v}\n")

    print((out_dir / "temperature_summary.txt").read_text())


if __name__ == "__main__":
    main()