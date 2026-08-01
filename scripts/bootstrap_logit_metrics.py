import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def apply_global_temperature_logits(logits, t):
    return softmax_np(logits / t, axis=1)


def apply_vector_temperature_logits(logits, temps):
    temps = np.asarray(temps, dtype=np.float32).reshape(1, 3)
    return softmax_np(logits / temps, axis=1)


def make_weights(labels, cache):
    y = np.argmax(labels, axis=1)

    total_neg = float(cache["negatives_seen_total"][0])
    sampled_neg = float(cache["sampled_negatives"][0])
    neg_weight = total_neg / sampled_neg

    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] = neg_weight

    return weights, neg_weight


def weighted_multiclass_nll(probs, labels, weights):
    y = np.argmax(labels, axis=1)
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    loss = -np.log(p)
    return float(np.sum(weights * loss) / np.sum(weights))


def weighted_multiclass_ece(probs, labels, weights, n_bins=15):
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


def binary_auprc(probs, labels, class_idx):
    y = labels[:, class_idx].astype(int)
    scores = probs[:, class_idx]
    return float(average_precision_score(y, scores))


def compute_metrics(probs, labels, weights, n_bins):
    return {
        "weighted_multiclass_ece": weighted_multiclass_ece(
            probs, labels, weights, n_bins=n_bins
        ),
        "weighted_multiclass_nll": weighted_multiclass_nll(
            probs, labels, weights
        ),
        "acceptor_auprc": binary_auprc(probs, labels, class_idx=1),
        "donor_auprc": binary_auprc(probs, labels, class_idx=2),
    }


def summarize_bootstrap(df):
    rows = []

    for method in df["method"].unique():
        mdf = df[df["method"] == method]

        for metric in [
            "weighted_multiclass_ece",
            "weighted_multiclass_nll",
            "acceptor_auprc",
            "donor_auprc",
        ]:
            values = mdf[metric].values

            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "ci_lower_2.5": float(np.percentile(values, 2.5)),
                    "ci_upper_97.5": float(np.percentile(values, 97.5)),
                }
            )

    return pd.DataFrame(rows)


def format_summary_for_markdown(summary_df):
    df = summary_df.copy()

    method_rename = {
        "uncalibrated": "Uncalibrated",
        "global_T_1.1": "Global T=1.1",
        "logit_unweighted_vector": "Logit unweighted vector T",
        "logit_weighted_vector": "Logit genome-weighted vector T",
    }

    metric_rename = {
        "weighted_multiclass_ece": "Weighted multiclass ECE",
        "weighted_multiclass_nll": "Weighted multiclass NLL",
        "acceptor_auprc": "Acceptor AUPRC",
        "donor_auprc": "Donor AUPRC",
    }

    df["Method"] = df["method"].map(method_rename)
    df["Metric"] = df["metric"].map(metric_rename)

    df["Mean"] = df["mean"].map(lambda x: f"{x:.6f}")
    df["95% CI"] = df.apply(
        lambda r: f"[{r['ci_lower_2.5']:.6f}, {r['ci_upper_97.5']:.6f}]",
        axis=1,
    )

    return df[["Method", "Metric", "Mean", "95% CI"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--n-bins", type=int, default=15)

    parser.add_argument("--global-temperature", type=float, default=1.1)

    parser.add_argument("--unweighted-t-nonsplice", type=float, required=True)
    parser.add_argument("--unweighted-t-acceptor", type=float, required=True)
    parser.add_argument("--unweighted-t-donor", type=float, required=True)

    parser.add_argument("--weighted-t-nonsplice", type=float, required=True)
    parser.add_argument("--weighted-t-acceptor", type=float, required=True)
    parser.add_argument("--weighted-t-donor", type=float, required=True)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_dir = Path("tables/logit_based")
    table_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)

    cache = np.load(args.cache)
    logits = cache["logits_sample"]
    labels = cache["labels_sample"]

    weights, neg_weight = make_weights(labels, cache)

    print("Loaded cache:", args.cache)
    print("logits:", logits.shape)
    print("labels:", labels.shape)
    print("negative weight:", neg_weight)
    print("n bootstrap:", args.n_bootstrap)

    full_methods = {
        "uncalibrated": softmax_np(logits, axis=1),
        f"global_T_{args.global_temperature}": apply_global_temperature_logits(
            logits,
            args.global_temperature,
        ),
        "logit_unweighted_vector": apply_vector_temperature_logits(
            logits,
            [
                args.unweighted_t_nonsplice,
                args.unweighted_t_acceptor,
                args.unweighted_t_donor,
            ],
        ),
        "logit_weighted_vector": apply_vector_temperature_logits(
            logits,
            [
                args.weighted_t_nonsplice,
                args.weighted_t_acceptor,
                args.weighted_t_donor,
            ],
        ),
    }

    point_rows = []

    print("\nPoint estimates on full sampled test cache:")
    for method_name, probs in full_methods.items():
        m = compute_metrics(probs, labels, weights, args.n_bins)
        row = {"method": method_name}
        row.update(m)
        point_rows.append(row)
        print(method_name, m)

    point_df = pd.DataFrame(point_rows)
    point_path = out_dir / "point_estimates.csv"
    point_df.to_csv(point_path, index=False)

    boot_rows = []
    n = len(labels)

    for b in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)

        b_labels = labels[idx]
        b_weights = weights[idx]

        for method_name, probs in full_methods.items():
            b_probs = probs[idx]
            m = compute_metrics(b_probs, b_labels, b_weights, args.n_bins)

            row = {
                "bootstrap": b,
                "method": method_name,
            }
            row.update(m)
            boot_rows.append(row)

        if (b + 1) % 10 == 0:
            print(f"Completed bootstrap {b + 1}/{args.n_bootstrap}", flush=True)

    boot_df = pd.DataFrame(boot_rows)

    boot_path = out_dir / "bootstrap_replicates.csv"
    summary_path = out_dir / "bootstrap_summary.csv"
    md_path = table_dir / "logit_bootstrap_summary.md"

    boot_df.to_csv(boot_path, index=False)

    summary = summarize_bootstrap(boot_df)
    summary.to_csv(summary_path, index=False)

    md_table = format_summary_for_markdown(summary)

    with open(md_path, "w") as f:
        f.write("# True-logit Bootstrap Summary\n\n")
        f.write(
            f"Bootstrap replicates: {args.n_bootstrap}\n\n"
            f"Negative sample weight: {neg_weight:.6f}\n\n"
        )
        f.write(md_table.to_markdown(index=False))
        f.write("\n")

    print("\nWrote:")
    print(point_path)
    print(boot_path)
    print(summary_path)
    print(md_path)

    print("\nBootstrap summary:")
    print(md_table.to_markdown(index=False))


if __name__ == "__main__":
    main()