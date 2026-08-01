from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CACHE = Path("results/logit_cache_flank400_epoch8/test_sampled_logits.npz")
FIG_DIR = Path("figures/logit_based_flank400")
TAB_DIR = Path("tables/logit_based_flank400")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 15
NEGATIVE_WEIGHT = 832.107824

METHODS = {
    "uncalibrated": np.array([1.0, 1.0, 1.0], dtype=np.float64),
    "true-logit genome-weighted vector T": np.array(
        [0.4728609323501587, 0.46725451946258545, 0.4916958808898926],
        dtype=np.float64,
    ),
    "OpenSpliceAI-style vector T": np.array(
        [0.3837826, 0.36834455, 0.38703716],
        dtype=np.float64,
    ),
}

CLASS_NAMES = {
    0: "nonsplice",
    1: "acceptor",
    2: "donor",
}

def first_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return npz[k]
    raise KeyError(f"None of these keys found: {candidates}. Available keys: {npz.files}")

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def weighted_mean(x, w):
    return float(np.sum(x * w) / np.sum(w))

def reliability_binary(prob, truth, weights, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    total_w = float(np.sum(weights))

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (prob >= lo) & (prob <= hi)
        else:
            m = (prob >= lo) & (prob < hi)

        if not np.any(m):
            rows.append({
                "bin": i,
                "bin_lower": lo,
                "bin_upper": hi,
                "count": 0,
                "weight_sum": 0.0,
                "mean_pred": np.nan,
                "observed": np.nan,
                "abs_gap": np.nan,
                "ece_contribution": 0.0,
            })
            continue

        ww = weights[m]
        mean_pred = weighted_mean(prob[m], ww)
        observed = weighted_mean(truth[m].astype(np.float64), ww)
        gap = abs(mean_pred - observed)
        rows.append({
            "bin": i,
            "bin_lower": lo,
            "bin_upper": hi,
            "count": int(np.sum(m)),
            "weight_sum": float(np.sum(ww)),
            "mean_pred": mean_pred,
            "observed": observed,
            "abs_gap": gap,
            "ece_contribution": float(np.sum(ww) / total_w * gap),
        })

    return pd.DataFrame(rows)

def reliability_multiclass(probs, labels, weights, n_bins):
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == labels).astype(np.float64)
    return reliability_binary(conf, correct, weights, n_bins)

data = np.load(CACHE)
print("Loaded:", CACHE)
print("Keys:", data.files)

logits = first_key(data, ["logits_sample", "logits", "sampled_logits", "test_logits"])
labels = first_key(data, ["labels_sample", "labels", "sampled_labels", "test_labels", "y", "Y"])

if labels.ndim == 2:
    labels = np.argmax(labels, axis=1)
labels = labels.astype(np.int64)

weights = np.ones_like(labels, dtype=np.float64)
weights[labels == 0] = NEGATIVE_WEIGHT

print("logits:", logits.shape)
print("labels:", labels.shape)
print("negative weight:", NEGATIVE_WEIGHT)

all_rows = []
summary_rows = []

computed = {}

for method, T in METHODS.items():
    probs = softmax(logits.astype(np.float64) / T.reshape(1, 3))
    computed[method] = probs

    mc = reliability_multiclass(probs, labels, weights, N_BINS)
    mc["method"] = method
    mc["reliability_type"] = "multiclass"
    mc["class"] = "argmax"
    all_rows.append(mc)

    summary_rows.append({
        "method": method,
        "metric": "weighted_multiclass_ece",
        "value": mc["ece_contribution"].sum(),
    })

    for cls in [1, 2]:
        prob = probs[:, cls]
        truth = labels == cls
        rel = reliability_binary(prob, truth, weights, N_BINS)
        rel["method"] = method
        rel["reliability_type"] = "binary_class"
        rel["class"] = CLASS_NAMES[cls]
        all_rows.append(rel)

        summary_rows.append({
            "method": method,
            "metric": f"{CLASS_NAMES[cls]}_ece",
            "value": rel["ece_contribution"].sum(),
        })

bins = pd.concat(all_rows, ignore_index=True)
summary = pd.DataFrame(summary_rows)

bins.to_csv(TAB_DIR / "reliability_bins_flank400_methods.csv", index=False)
summary.to_csv(TAB_DIR / "reliability_summary_flank400_methods.csv", index=False)

def plot_reliability(reliability_type, class_name, title, outfile):
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    for method in METHODS:
        sub = bins[
            (bins["method"] == method)
            & (bins["reliability_type"] == reliability_type)
            & (bins["class"] == class_name)
            & bins["mean_pred"].notna()
        ]
        plt.plot(sub["mean_pred"], sub["observed"], marker="o", label=method)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency / accuracy")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / outfile, dpi=300)
    plt.close()

plot_reliability(
    "multiclass",
    "argmax",
    "Flank-400 multiclass reliability",
    "reliability_multiclass_flank400_methods.png",
)

plot_reliability(
    "binary_class",
    "acceptor",
    "Flank-400 acceptor reliability",
    "reliability_acceptor_flank400_methods.png",
)

plot_reliability(
    "binary_class",
    "donor",
    "Flank-400 donor reliability",
    "reliability_donor_flank400_methods.png",
)

print("Wrote:")
print(FIG_DIR / "reliability_multiclass_flank400_methods.png")
print(FIG_DIR / "reliability_acceptor_flank400_methods.png")
print(FIG_DIR / "reliability_donor_flank400_methods.png")
print(TAB_DIR / "reliability_bins_flank400_methods.csv")
print(TAB_DIR / "reliability_summary_flank400_methods.csv")
print()
print(summary.to_string(index=False))
