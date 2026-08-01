#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def load_temperature(path):
    txt = open(path).read().strip()
    txt = txt.replace("[", "").replace("]", "")
    return np.fromstring(txt, sep=" ").astype(np.float64)


def reliability_bins_multiclass(probs, y, weights, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)

    rows = []
    edges = np.linspace(0, 1, n_bins + 1)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue

        w = weights[mask]
        bin_w = w.sum()
        acc = np.sum(w * correct[mask]) / bin_w
        avg_conf = np.sum(w * conf[mask]) / bin_w

        rows.append({
            "bin": i,
            "bin_low": lo,
            "bin_high": hi,
            "count": int(mask.sum()),
            "weighted_count": bin_w,
            "mean_confidence": avg_conf,
            "observed_accuracy": acc,
            "gap": acc - avg_conf,
        })

    return pd.DataFrame(rows)


def reliability_bins_binary(p, y_bin, weights, n_bins=15):
    rows = []
    edges = np.linspace(0, 1, n_bins + 1)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p > lo) & (p <= hi)
        if not np.any(mask):
            continue

        w = weights[mask]
        bin_w = w.sum()
        obs = np.sum(w * y_bin[mask]) / bin_w
        pred = np.sum(w * p[mask]) / bin_w

        rows.append({
            "bin": i,
            "bin_low": lo,
            "bin_high": hi,
            "count": int(mask.sum()),
            "weighted_count": bin_w,
            "mean_prediction": pred,
            "observed_frequency": obs,
            "gap": obs - pred,
        })

    return pd.DataFrame(rows)


def plot_multiclass(df, outpath, title):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(df["mean_confidence"], df["observed_accuracy"])
    plt.xlabel("Mean confidence")
    plt.ylabel("Observed accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_binary(df, outpath, title):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(df["mean_prediction"], df["observed_frequency"])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--temperature-txt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="temperature_method")
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cache = np.load(args.cache)
    logits = cache["logits_sample"].astype(np.float64)
    y = cache["labels_sample"].argmax(axis=1).astype(int)

    sampled_negatives = int(cache["sampled_negatives"][0])
    negatives_seen_total = int(cache["negatives_seen_total"][0])
    neg_weight = negatives_seen_total / sampled_negatives

    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = neg_weight

    T = load_temperature(args.temperature_txt)
    probs = softmax(logits / T.reshape(1, 3))

    multi = reliability_bins_multiclass(probs, y, weights, args.n_bins)
    acc = reliability_bins_binary(probs[:, 1], (y == 1).astype(int), weights, args.n_bins)
    don = reliability_bins_binary(probs[:, 2], (y == 2).astype(int), weights, args.n_bins)

    multi.to_csv(os.path.join(args.output_dir, "reliability_multiclass_bins.csv"), index=False)
    acc.to_csv(os.path.join(args.output_dir, "reliability_acceptor_bins.csv"), index=False)
    don.to_csv(os.path.join(args.output_dir, "reliability_donor_bins.csv"), index=False)

    plot_multiclass(
        multi,
        os.path.join(args.output_dir, "reliability_multiclass.png"),
        f"{args.method_name}: multiclass reliability",
    )
    plot_binary(
        acc,
        os.path.join(args.output_dir, "reliability_acceptor.png"),
        f"{args.method_name}: acceptor reliability",
    )
    plot_binary(
        don,
        os.path.join(args.output_dir, "reliability_donor.png"),
        f"{args.method_name}: donor reliability",
    )

    print("Saved reliability outputs to:", args.output_dir)
    print("\nMulticlass bins:")
    print(multi.to_string(index=False))
    print("\nAcceptor bins:")
    print(acc.to_string(index=False))
    print("\nDonor bins:")
    print(don.to_string(index=False))


if __name__ == "__main__":
    main()
