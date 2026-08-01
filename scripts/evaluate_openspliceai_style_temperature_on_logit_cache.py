#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import torch
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
        if mask.sum() == 0:
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
        if mask.sum() == 0:
            continue
        w = weights[mask]
        bin_w = w.sum()
        obs = np.sum(w * y_bin[mask]) / bin_w
        pred = np.sum(w * p[mask]) / bin_w
        ece += (bin_w / total_w) * abs(obs - pred)

    return ece


def threshold_metrics(probs, y, cls, thresholds):
    rows = []
    y_bin = (y == cls).astype(int)
    p = probs[:, cls]

    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y_bin == 1))
        fp = np.sum(pred & (y_bin == 0))
        fn = np.sum((~pred) & (y_bin == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        rows.append({
            "class": cls,
            "threshold": t,
            "predicted_positive": int(pred.sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--temperature-txt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method-name", default="openspliceai_style_vectorT_long")
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cache = np.load(args.cache)
    logits = cache["logits_sample"].astype(np.float64)
    labels_oh = cache["labels_sample"]
    y = labels_oh.argmax(axis=1).astype(int)

    total_positions_seen = int(cache["total_positions_seen"][0])
    positives_seen = int(cache["positive_positions_seen"][0])
    sampled_negatives = int(cache["sampled_negatives"][0])
    negatives_seen_total = int(cache["negatives_seen_total"][0])

    # Weight sampled negatives back to their genome-level count.
    weights = np.ones(len(y), dtype=np.float64)
    neg_mask = y == 0
    neg_weight = negatives_seen_total / sampled_negatives
    weights[neg_mask] = neg_weight

    # Load temperature like: [0.3932362  0.35275677 0.36842206]
    txt = open(args.temperature_txt).read().strip()
    txt = txt.replace("[", "").replace("]", "")
    T = np.fromstring(txt, sep=" ").astype(np.float64)

    probs = softmax(logits / T.reshape(1, 3))

    rows = []
    rows.append({
        "method": args.method_name,
        "T_nonsplice": T[0],
        "T_acceptor": T[1],
        "T_donor": T[2],
        "weighted_multiclass_ece": multiclass_ece(probs, y, weights, args.n_bins),
        "weighted_nll": weighted_nll(probs, y, weights),
        "acceptor_ece": binary_ece(probs[:, 1], (y == 1).astype(int), weights, args.n_bins),
        "donor_ece": binary_ece(probs[:, 2], (y == 2).astype(int), weights, args.n_bins),
        "acceptor_auprc": average_precision_score((y == 1).astype(int), probs[:, 1]),
        "donor_auprc": average_precision_score((y == 2).astype(int), probs[:, 2]),
        "total_positions_seen": total_positions_seen,
        "positives_seen": positives_seen,
        "sampled_negatives": sampled_negatives,
        "negative_weight": neg_weight,
    })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, "openspliceai_style_temperature_test_metrics.csv")
    summary.to_csv(summary_path, index=False)

    thresh_rows = []
    for cls in [1, 2]:
        thresh_rows.extend(threshold_metrics(probs, y, cls, [0.01, 0.05, 0.1, 0.5]))

    thresh = pd.DataFrame(thresh_rows)
    thresh_path = os.path.join(args.output_dir, "openspliceai_style_temperature_threshold_metrics.csv")
    thresh.to_csv(thresh_path, index=False)

    argmax = probs.argmax(axis=1)
    argmax_rows = []
    for cls in [0, 1, 2]:
        m = argmax == cls
        argmax_rows.append({
            "predicted_class": cls,
            "predicted_count": int(m.sum()),
            "true_positive_count": int(np.sum(m & (y == cls))),
        })

    argmax_df = pd.DataFrame(argmax_rows)
    argmax_path = os.path.join(args.output_dir, "openspliceai_style_temperature_argmax.csv")
    argmax_df.to_csv(argmax_path, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nThreshold metrics:")
    print(thresh.to_string(index=False))
    print("\nArgmax:")
    print(argmax_df.to_string(index=False))
    print("\nSaved:")
    print(summary_path)
    print(thresh_path)
    print(argmax_path)


if __name__ == "__main__":
    main()
