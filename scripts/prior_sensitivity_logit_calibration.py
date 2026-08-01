import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def apply_vector_temperature_logits(logits, temps):
    temps = np.asarray(temps, dtype=np.float32).reshape(1, 3)
    return softmax_np(logits / temps, axis=1)


def make_weights(labels, neg_weight):
    y = np.argmax(labels, axis=1)
    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = neg_weight
    return weights


def weighted_nll(probs, labels, weights):
    y = np.argmax(labels, axis=1)
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    loss = -np.log(p)
    return float(np.sum(weights * loss) / np.sum(weights))


def weighted_ece(probs, labels, weights, n_bins=15):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)

    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total_w = np.sum(weights)

    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if not np.any(mask):
            continue

        w = weights[mask]
        bin_w = np.sum(w)

        bin_conf = np.sum(w * conf[mask]) / bin_w
        bin_acc = np.sum(w * correct[mask]) / bin_w

        ece += (bin_w / total_w) * abs(bin_acc - bin_conf)

    return float(ece)


def argmax_counts(probs, labels):
    y_pred = np.argmax(probs, axis=1)

    return {
        "pred_nonsplice": int((y_pred == 0).sum()),
        "pred_acceptor": int((y_pred == 1).sum()),
        "pred_donor": int((y_pred == 2).sum()),
    }


def fit_vector_temperature(logits_np, labels_np, weights_np, lr=0.01, steps=600):
    logits = torch.tensor(logits_np, dtype=torch.float32)
    y = torch.tensor(np.argmax(labels_np, axis=1), dtype=torch.long)

    weights = torch.tensor(weights_np / np.mean(weights_np), dtype=torch.float32)

    raw_t = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    optimizer = torch.optim.Adam([raw_t], lr=lr)

    best_loss = float("inf")
    best_t = None

    for step in range(steps):
        optimizer.zero_grad()

        temps = 0.05 + 4.95 * torch.sigmoid(raw_t)
        scaled_logits = logits / temps.reshape(1, 3)

        per_sample_loss = torch.nn.functional.cross_entropy(
            scaled_logits,
            y,
            reduction="none",
        )

        loss = torch.sum(weights * per_sample_loss) / torch.sum(weights)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())

        if loss_value < best_loss:
            best_loss = loss_value
            best_t = temps.detach().cpu().numpy().copy()

    return best_t, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path("figures/logit_based")
    table_dir = Path("tables/logit_based")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    val = np.load(args.validation_cache)
    test = np.load(args.test_cache)

    val_logits = val["logits_sample"]
    val_labels = val["labels_sample"]

    test_logits = test["logits_sample"]
    test_labels = test["labels_sample"]

    validation_genome_neg_weight = float(val["negatives_seen_total"][0]) / float(
        val["sampled_negatives"][0]
    )

    test_genome_neg_weight = float(test["negatives_seen_total"][0]) / float(
        test["sampled_negatives"][0]
    )

    sweep_weights = [
        1.0,
        2.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        validation_genome_neg_weight,
    ]

    rows = []

    print("Validation genome negative weight:", validation_genome_neg_weight)
    print("Test genome negative weight:", test_genome_neg_weight)
    print("Sweep weights:", sweep_weights)

    for neg_weight in sweep_weights:
        print(f"\nFitting vector T with validation negative weight = {neg_weight:.4f}")

        val_weights = make_weights(val_labels, neg_weight)

        temps, best_loss = fit_vector_temperature(
            val_logits,
            val_labels,
            val_weights,
            lr=args.lr,
            steps=args.steps,
        )

        test_weights_same_prior = make_weights(test_labels, neg_weight)
        test_weights_genome_prior = make_weights(test_labels, test_genome_neg_weight)

        probs_test = apply_vector_temperature_logits(test_logits, temps)

        same_prior_ece = weighted_ece(
            probs_test,
            test_labels,
            test_weights_same_prior,
            n_bins=args.n_bins,
        )

        same_prior_nll = weighted_nll(
            probs_test,
            test_labels,
            test_weights_same_prior,
        )

        genome_prior_ece = weighted_ece(
            probs_test,
            test_labels,
            test_weights_genome_prior,
            n_bins=args.n_bins,
        )

        genome_prior_nll = weighted_nll(
            probs_test,
            test_labels,
            test_weights_genome_prior,
        )

        counts = argmax_counts(probs_test, test_labels)

        row = {
            "validation_negative_weight": neg_weight,
            "T_nonsplice": float(temps[0]),
            "T_acceptor": float(temps[1]),
            "T_donor": float(temps[2]),
            "validation_fit_loss": float(best_loss),
            "test_ece_same_prior": same_prior_ece,
            "test_nll_same_prior": same_prior_nll,
            "test_ece_genome_prior": genome_prior_ece,
            "test_nll_genome_prior": genome_prior_nll,
        }

        row.update(counts)
        rows.append(row)

        print(row)

    df = pd.DataFrame(rows)

    csv_out = out_dir / "prior_sensitivity_summary.csv"
    md_out = table_dir / "logit_prior_sensitivity_summary.md"

    df.to_csv(csv_out, index=False)

    md_df = df.copy()
    for col in md_df.columns:
        if col.startswith("T_") or "ece" in col or "nll" in col or "loss" in col:
            md_df[col] = md_df[col].map(lambda x: f"{x:.6f}")
        elif col == "validation_negative_weight":
            md_df[col] = md_df[col].map(lambda x: f"{x:.3f}")

    with open(md_out, "w") as f:
        f.write("# Logit Prior-Sensitivity Summary\n\n")
        f.write(f"Validation genome negative weight: {validation_genome_neg_weight:.6f}\n\n")
        f.write(f"Test genome negative weight: {test_genome_neg_weight:.6f}\n\n")
        f.write(md_df.to_markdown(index=False))
        f.write("\n")

    print("\nWrote:")
    print(csv_out)
    print(md_out)

    # Figure 1: temperatures vs negative weight
    plt.figure(figsize=(8, 5))
    plt.plot(df["validation_negative_weight"], df["T_nonsplice"], marker="o", label="T_nonsplice")
    plt.plot(df["validation_negative_weight"], df["T_acceptor"], marker="s", label="T_acceptor")
    plt.plot(df["validation_negative_weight"], df["T_donor"], marker="^", label="T_donor")
    plt.xscale("log")
    plt.xlabel("Validation negative-class weight")
    plt.ylabel("Learned temperature")
    plt.title("Prior sensitivity of learned vector temperatures")
    plt.legend()
    plt.tight_layout()
    temp_fig = fig_dir / "logit_prior_sensitivity_temperatures.png"
    plt.savefig(temp_fig, dpi=300)
    plt.close()

    # Figure 2: genome-prior calibration metrics vs negative weight
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["validation_negative_weight"],
        df["test_ece_genome_prior"],
        marker="o",
        label="Genome-prior ECE",
    )
    plt.plot(
        df["validation_negative_weight"],
        df["test_nll_genome_prior"],
        marker="s",
        label="Genome-prior NLL",
    )
    plt.xscale("log")
    plt.xlabel("Validation negative-class weight")
    plt.ylabel("Metric value")
    plt.title("Genome-prior calibration vs calibration prior")
    plt.legend()
    plt.tight_layout()
    metric_fig = fig_dir / "logit_prior_sensitivity_genome_metrics.png"
    plt.savefig(metric_fig, dpi=300)
    plt.close()

    # Figure 3: argmax counts vs negative weight
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["validation_negative_weight"],
        df["pred_acceptor"],
        marker="o",
        label="Predicted acceptor",
    )
    plt.plot(
        df["validation_negative_weight"],
        df["pred_donor"],
        marker="s",
        label="Predicted donor",
    )
    plt.xscale("log")
    plt.xlabel("Validation negative-class weight")
    plt.ylabel("Argmax predicted splice positions")
    plt.title("Prior sensitivity of argmax splice predictions")
    plt.legend()
    plt.tight_layout()
    count_fig = fig_dir / "logit_prior_sensitivity_argmax_counts.png"
    plt.savefig(count_fig, dpi=300)
    plt.close()

    print(temp_fig)
    print(metric_fig)
    print(count_fig)


if __name__ == "__main__":
    main()