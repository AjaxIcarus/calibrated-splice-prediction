#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def weighted_nll(probs, y, weights):
    eps = 1e-12
    p_true = probs[np.arange(len(y)), y]
    return np.sum(weights * -np.log(np.clip(p_true, eps, 1.0))) / np.sum(weights)


def multiclass_ece(probs, y, weights, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)

    ece = 0.0
    total_w = weights.sum()
    edges = np.linspace(0, 1, n_bins + 1)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        w = weights[mask]
        bin_w = w.sum()
        acc = np.sum(w * correct[mask]) / bin_w
        avg_conf = np.sum(w * conf[mask]) / bin_w
        ece += (bin_w / total_w) * abs(acc - avg_conf)

    return ece


def binary_ece(p, y_bin, weights, n_bins=15):
    ece = 0.0
    total_w = weights.sum()
    edges = np.linspace(0, 1, n_bins + 1)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi)
        if not np.any(mask):
            continue
        w = weights[mask]
        bin_w = w.sum()
        obs = np.sum(w * y_bin[mask]) / bin_w
        pred = np.sum(w * p[mask]) / bin_w
        ece += (bin_w / total_w) * abs(obs - pred)

    return ece


def summarize(values):
    values = np.asarray(values)
    return {
        "mean": np.mean(values),
        "ci_low": np.percentile(values, 2.5),
        "ci_high": np.percentile(values, 97.5),
    }


def load_temperature(path):
    txt = open(path).read().strip()
    txt = txt.replace("[", "").replace("]", "")
    return np.fromstring(txt, sep=" ").astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--temperature-txt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-name", default="temperature_method")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cache = np.load(args.cache)
    logits = cache["logits_sample"].astype(np.float64)
    labels_oh = cache["labels_sample"]
    y = labels_oh.argmax(axis=1).astype(int)

    sampled_negatives = int(cache["sampled_negatives"][0])
    negatives_seen_total = int(cache["negatives_seen_total"][0])
    neg_weight = negatives_seen_total / sampled_negatives

    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = neg_weight

    T = load_temperature(args.temperature_txt)
    probs = softmax(logits / T.reshape(1, 3))

    rng = np.random.default_rng(args.seed)
    n = len(y)

    rows = []

    print(f"Bootstrapping {args.method_name}")
    print(f"N = {n:,}")
    print(f"Bootstrap samples = {args.n_bootstrap}")
    print(f"Temperature = {T}")

    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)

        pb = probs[idx]
        yb = y[idx]
        wb = weights[idx]

        row = {
            "bootstrap": b,
            "weighted_multiclass_ece": multiclass_ece(pb, yb, wb, args.n_bins),
            "weighted_nll": weighted_nll(pb, yb, wb),
            "acceptor_ece": binary_ece(pb[:, 1], (yb == 1).astype(int), wb, args.n_bins),
            "donor_ece": binary_ece(pb[:, 2], (yb == 2).astype(int), wb, args.n_bins),
            "acceptor_auprc": average_precision_score((yb == 1).astype(int), pb[:, 1]),
            "donor_auprc": average_precision_score((yb == 2).astype(int), pb[:, 2]),
        }
        rows.append(row)

        if (b + 1) % 25 == 0 or (b + 1) == args.n_bootstrap:
            print(f"Completed {b + 1}/{args.n_bootstrap}")

    boot = pd.DataFrame(rows)
    boot_path = os.path.join(args.output_dir, "bootstrap_samples.csv")
    boot.to_csv(boot_path, index=False)

    summary_rows = []
    for metric in [
        "weighted_multiclass_ece",
        "weighted_nll",
        "acceptor_ece",
        "donor_ece",
        "acceptor_auprc",
        "donor_auprc",
    ]:
        s = summarize(boot[metric])
        summary_rows.append({
            "method": args.method_name,
            "metric": metric,
            "mean": s["mean"],
            "ci_low": s["ci_low"],
            "ci_high": s["ci_high"],
            "T_nonsplice": T[0],
            "T_acceptor": T[1],
            "T_donor": T[2],
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "bootstrap_summary.csv")
    summary.to_csv(summary_path, index=False)

    md_path = os.path.join(args.output_dir, "bootstrap_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Bootstrap summary: {args.method_name}\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print("\nBootstrap summary:")
    print(summary.to_markdown(index=False))
    print("\nSaved:")
    print(boot_path)
    print(summary_path)
    print(md_path)


if __name__ == "__main__":
    main()
