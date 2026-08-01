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


def expected_calibration_error(confidences, correct, n_bins=15):
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append((lo, hi, 0, np.nan, np.nan, np.nan))
            continue

        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (count / len(confidences)) * gap
        rows.append((lo, hi, count, bin_conf, bin_acc, gap))

    return ece, rows


def binary_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append((lo, hi, 0, np.nan, np.nan, np.nan))
            continue

        bin_conf = float(probs[mask].mean())
        bin_freq = float(labels[mask].mean())
        gap = abs(bin_freq - bin_conf)
        ece += (count / len(probs)) * gap
        rows.append((lo, hi, count, bin_conf, bin_freq, gap))

    return ece, rows


def save_bin_table(path, rows):
    with open(path, "w") as f:
        f.write("bin_low,bin_high,count,mean_confidence,empirical_accuracy_or_frequency,gap\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


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

                    # model output and labels are expected as B x 3 x L
                    all_probs.append(probs.detach().cpu().numpy())
                    all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("probs shape:", probs.shape)
    print("labels shape:", labels.shape)

    # Flatten positions
    probs_flat = np.transpose(probs, (0, 2, 1)).reshape(-1, 3)
    labels_flat = np.transpose(labels, (0, 2, 1)).reshape(-1, 3)

    y_true = np.argmax(labels_flat, axis=1)
    y_pred = np.argmax(probs_flat, axis=1)
    conf = np.max(probs_flat, axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    multiclass_ece, multiclass_rows = expected_calibration_error(conf, correct, n_bins=15)

    # NLL over all 3 classes
    eps = 1e-12
    probs_clip = np.clip(probs_flat, eps, 1.0 - eps)
    nll = log_loss(y_true, probs_clip, labels=[0, 1, 2])

    # multiclass Brier = mean sum_k (p_k - y_k)^2
    brier_multiclass = np.mean(np.sum((probs_flat - labels_flat) ** 2, axis=1))

    acceptor_probs = probs_flat[:, 1]
    donor_probs = probs_flat[:, 2]
    acceptor_labels = labels_flat[:, 1]
    donor_labels = labels_flat[:, 2]

    acceptor_ece, acceptor_rows = binary_ece(acceptor_probs, acceptor_labels, n_bins=15)
    donor_ece, donor_rows = binary_ece(donor_probs, donor_labels, n_bins=15)

    acceptor_brier = brier_score_loss(acceptor_labels, acceptor_probs)
    donor_brier = brier_score_loss(donor_labels, donor_probs)

    # Binary NLL for splice classes
    acceptor_nll = log_loss(acceptor_labels, np.clip(acceptor_probs, eps, 1.0 - eps), labels=[0, 1])
    donor_nll = log_loss(donor_labels, np.clip(donor_probs, eps, 1.0 - eps), labels=[0, 1])

    summary_path = out_dir / "calibration_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Uncalibrated validation calibration metrics\n")
        f.write("==========================================\n")
        f.write(f"multiclass_ece: {multiclass_ece}\n")
        f.write(f"multiclass_nll: {nll}\n")
        f.write(f"multiclass_brier: {brier_multiclass}\n")
        f.write(f"acceptor_ece: {acceptor_ece}\n")
        f.write(f"acceptor_nll: {acceptor_nll}\n")
        f.write(f"acceptor_brier: {acceptor_brier}\n")
        f.write(f"donor_ece: {donor_ece}\n")
        f.write(f"donor_nll: {donor_nll}\n")
        f.write(f"donor_brier: {donor_brier}\n")

    save_bin_table(out_dir / "multiclass_reliability_bins.csv", multiclass_rows)
    save_bin_table(out_dir / "acceptor_reliability_bins.csv", acceptor_rows)
    save_bin_table(out_dir / "donor_reliability_bins.csv", donor_rows)

    print(summary_path.read_text())


if __name__ == "__main__":
    main()