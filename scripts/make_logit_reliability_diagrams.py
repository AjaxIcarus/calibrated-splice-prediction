from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


CACHE = Path("results/logit_cache_flank80_epoch2/test_sampled_logits.npz")
OUT_DIR = Path("figures/logit_based")
OUT_DIR.mkdir(parents=True, exist_ok=True)


GLOBAL_T = 1.1

UNWEIGHTED_T = np.array([0.81190228, 0.24863805, 0.34511948], dtype=np.float32)
WEIGHTED_T = np.array([0.43516919, 0.41615313, 0.42794070], dtype=np.float32)


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


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


def multiclass_reliability(probs, labels, weights, n_bins=15):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)

    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers = []
    bin_conf = []
    bin_acc = []
    bin_weight = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if not np.any(mask):
            continue

        w = weights[mask]
        total_w = np.sum(w)

        bin_centers.append((lo + hi) / 2.0)
        bin_conf.append(float(np.sum(w * conf[mask]) / total_w))
        bin_acc.append(float(np.sum(w * correct[mask]) / total_w))
        bin_weight.append(float(total_w))

    return np.array(bin_conf), np.array(bin_acc), np.array(bin_weight)


def binary_reliability(probs, labels, weights, class_idx, n_bins=15):
    y = labels[:, class_idx].astype(np.float64)
    p = probs[:, class_idx]

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_conf = []
    bin_freq = []
    bin_weight = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not np.any(mask):
            continue

        w = weights[mask]
        total_w = np.sum(w)

        bin_conf.append(float(np.sum(w * p[mask]) / total_w))
        bin_freq.append(float(np.sum(w * y[mask]) / total_w))
        bin_weight.append(float(total_w))

    return np.array(bin_conf), np.array(bin_freq), np.array(bin_weight)


def plot_multiclass_single(probs, labels, weights, title, out_path):
    conf, acc, bin_weight = multiclass_reliability(probs, labels, weights)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(conf, acc, marker="o", label="Observed")

    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Wrote:", out_path)


def plot_binary_compare(
    probs_a,
    probs_b,
    labels,
    weights,
    class_idx,
    class_name,
    method_a,
    method_b,
    out_path,
):
    conf_a, freq_a, _ = binary_reliability(probs_a, labels, weights, class_idx)
    conf_b, freq_b, _ = binary_reliability(probs_b, labels, weights, class_idx)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(conf_a, freq_a, marker="o", label=method_a)
    plt.plot(conf_b, freq_b, marker="s", label=method_b)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"{class_name} reliability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Wrote:", out_path)


def write_reliability_tables(methods, labels, weights):
    table_dir = Path("tables/logit_based")
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for method_name, probs in methods.items():
        conf, acc, bin_weight = multiclass_reliability(probs, labels, weights)

        for i in range(len(conf)):
            rows.append(
                {
                    "method": method_name,
                    "type": "multiclass",
                    "class": "argmax",
                    "bin": i,
                    "mean_confidence": conf[i],
                    "observed_frequency": acc[i],
                    "weighted_count": bin_weight[i],
                }
            )

        for class_idx, class_name in [(1, "acceptor"), (2, "donor")]:
            conf, freq, bin_weight = binary_reliability(
                probs,
                labels,
                weights,
                class_idx,
            )

            for i in range(len(conf)):
                rows.append(
                    {
                        "method": method_name,
                        "type": "binary",
                        "class": class_name,
                        "bin": i,
                        "mean_confidence": conf[i],
                        "observed_frequency": freq[i],
                        "weighted_count": bin_weight[i],
                    }
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    out_csv = table_dir / "logit_reliability_bins.csv"
    df.to_csv(out_csv, index=False)

    print("Wrote:", out_csv)


def main():
    cache = np.load(CACHE)
    logits = cache["logits_sample"]
    labels = cache["labels_sample"]

    weights, neg_weight = make_weights(labels, cache)

    print("Loaded:", CACHE)
    print("logits:", logits.shape)
    print("labels:", labels.shape)
    print("negative weight:", neg_weight)

    probs_uncal = softmax_np(logits, axis=1)
    probs_global = softmax_np(logits / GLOBAL_T, axis=1)
    probs_unweighted = apply_vector_temperature_logits(logits, UNWEIGHTED_T)
    probs_weighted = apply_vector_temperature_logits(logits, WEIGHTED_T)

    methods = {
        "uncalibrated": probs_uncal,
        "global_T_1.1": probs_global,
        "logit_unweighted_vector": probs_unweighted,
        "logit_weighted_vector": probs_weighted,
    }

    plot_multiclass_single(
        probs_uncal,
        labels,
        weights,
        "Multiclass reliability: uncalibrated",
        OUT_DIR / "reliability_multiclass_uncalibrated.png",
    )

    plot_multiclass_single(
        probs_weighted,
        labels,
        weights,
        "Multiclass reliability: logit genome-weighted vector T",
        OUT_DIR / "reliability_multiclass_logit_weighted_vector.png",
    )

    plot_binary_compare(
        probs_uncal,
        probs_weighted,
        labels,
        weights,
        class_idx=1,
        class_name="Acceptor",
        method_a="Uncalibrated",
        method_b="Logit genome-weighted vector T",
        out_path=OUT_DIR / "reliability_acceptor_uncalibrated_vs_weighted.png",
    )

    plot_binary_compare(
        probs_uncal,
        probs_weighted,
        labels,
        weights,
        class_idx=2,
        class_name="Donor",
        method_a="Uncalibrated",
        method_b="Logit genome-weighted vector T",
        out_path=OUT_DIR / "reliability_donor_uncalibrated_vs_weighted.png",
    )

    write_reliability_tables(methods, labels, weights)


if __name__ == "__main__":
    main()