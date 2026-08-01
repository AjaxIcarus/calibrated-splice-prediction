import argparse
from pathlib import Path

import numpy as np
import torch


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


def multiclass_nll(probs, labels, weights=None):
    y = np.argmax(labels, axis=1)
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    loss = -np.log(p)
    if weights is None:
        return float(np.mean(loss))
    return float(np.sum(weights * loss) / np.sum(weights))


def multiclass_brier(probs, labels, weights=None):
    b = np.sum((probs - labels) ** 2, axis=1)
    if weights is None:
        return float(np.mean(b))
    return float(np.sum(weights * b) / np.sum(weights))


def binary_nll(p, y, weights=None):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if weights is None:
        return float(np.mean(loss))
    return float(np.sum(weights * loss) / np.sum(weights))


def binary_brier(p, y, weights=None):
    b = (p - y) ** 2
    if weights is None:
        return float(np.mean(b))
    return float(np.sum(weights * b) / np.sum(weights))


def multiclass_ece(probs, labels, n_bins=15, weights=None):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float64)

    if weights is None:
        weights = np.ones(len(y_true), dtype=np.float64)

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


def binary_ece(p, y, n_bins=15, weights=None):
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total_w = np.sum(weights)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not np.any(mask):
            continue

        w = weights[mask]
        bin_w = np.sum(w)
        bin_conf = np.sum(w * p[mask]) / bin_w
        bin_freq = np.sum(w * y[mask]) / bin_w
        ece += (bin_w / total_w) * abs(bin_freq - bin_conf)

    return float(ece)


def metrics(probs, labels, n_bins, weights=None):
    out = {
        "multiclass_ece": multiclass_ece(probs, labels, n_bins, weights),
        "multiclass_nll": multiclass_nll(probs, labels, weights),
        "multiclass_brier": multiclass_brier(probs, labels, weights),
    }

    for idx, name in [(1, "acceptor"), (2, "donor")]:
        y = labels[:, idx]
        p = probs[:, idx]
        out[f"{name}_ece"] = binary_ece(p, y, n_bins, weights)
        out[f"{name}_nll"] = binary_nll(p, y, weights)
        out[f"{name}_brier"] = binary_brier(p, y, weights)

    return out


def fit_vector_temperature(logits_np, labels_np, weights_np=None, lr=0.01, steps=1000):
    logits = torch.tensor(logits_np, dtype=torch.float32)
    y = torch.tensor(np.argmax(labels_np, axis=1), dtype=torch.long)

    if weights_np is None:
        weights = torch.ones(len(y), dtype=torch.float32)
    else:
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

        if step % 100 == 0:
            print(
                f"step={step} loss={loss_value:.8f} "
                f"T_nonsplice={temps[0].item():.4f} "
                f"T_acceptor={temps[1].item():.4f} "
                f"T_donor={temps[2].item():.4f}",
                flush=True,
            )

    return best_t, best_loss


def argmax_summary(probs, labels):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)

    rows = {}
    for i, name in enumerate(["nonsplice", "acceptor", "donor"]):
        rows[name] = {
            "true": int((y_true == i).sum()),
            "predicted": int((y_pred == i).sum()),
            "tp": int(((y_true == i) & (y_pred == i)).sum()),
        }
    return rows


def write_csv(path, rows):
    header = [
        "split",
        "method",
        "multiclass_ece",
        "multiclass_nll",
        "multiclass_brier",
        "acceptor_ece",
        "acceptor_nll",
        "acceptor_brier",
        "donor_ece",
        "donor_nll",
        "donor_brier",
    ]

    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--global-temperature", type=float, default=1.1)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_cache = np.load(args.validation_cache)
    test_cache = np.load(args.test_cache)

    val_logits = val_cache["logits_sample"]
    val_labels = val_cache["labels_sample"]
    test_logits = test_cache["logits_sample"]
    test_labels = test_cache["labels_sample"]

    val_weights, val_neg_weight = make_weights(val_labels, val_cache)
    test_weights, test_neg_weight = make_weights(test_labels, test_cache)

    print("Validation logits:", val_logits.shape)
    print("Test logits:", test_logits.shape)
    print("Validation negative weight:", val_neg_weight)
    print("Test negative weight:", test_neg_weight)

    print("\nFitting unweighted true-logit vector temperature...")
    unweighted_t, unweighted_loss = fit_vector_temperature(
        val_logits,
        val_labels,
        weights_np=None,
        lr=args.lr,
        steps=args.steps,
    )

    print("\nFitting genome-weighted true-logit vector temperature...")
    weighted_t, weighted_loss = fit_vector_temperature(
        val_logits,
        val_labels,
        weights_np=val_weights,
        lr=args.lr,
        steps=args.steps,
    )

    print("\nBest unweighted true-logit vector temperature:")
    print("T_nonsplice:", unweighted_t[0])
    print("T_acceptor:", unweighted_t[1])
    print("T_donor:", unweighted_t[2])
    print("Best unweighted validation NLL:", unweighted_loss)

    print("\nBest genome-weighted true-logit vector temperature:")
    print("T_nonsplice:", weighted_t[0])
    print("T_acceptor:", weighted_t[1])
    print("T_donor:", weighted_t[2])
    print("Best weighted validation NLL:", weighted_loss)

    rows = []

    for split_name, logits, labels, weights in [
        ("validation_sampled_unweighted", val_logits, val_labels, None),
        ("test_sampled_unweighted", test_logits, test_labels, None),
        ("validation_sampled_weighted", val_logits, val_labels, val_weights),
        ("test_sampled_weighted", test_logits, test_labels, test_weights),
    ]:
        methods = {
            "uncalibrated": softmax_np(logits, axis=1),
            f"global_T_{args.global_temperature}": apply_global_temperature_logits(
                logits, args.global_temperature
            ),
            f"logit_unweighted_vector_T_{unweighted_t[0]:.4f}_{unweighted_t[1]:.4f}_{unweighted_t[2]:.4f}": apply_vector_temperature_logits(
                logits, unweighted_t
            ),
            f"logit_weighted_vector_T_{weighted_t[0]:.4f}_{weighted_t[1]:.4f}_{weighted_t[2]:.4f}": apply_vector_temperature_logits(
                logits, weighted_t
            ),
        }

        for method_name, probs in methods.items():
            m = metrics(probs, labels, args.n_bins, weights)
            row = {"split": split_name, "method": method_name}
            row.update(m)
            rows.append(row)

    csv_path = out_dir / "logit_vector_temperature_summary.csv"
    write_csv(csv_path, rows)

    diag_path = out_dir / "logit_vector_argmax_summary.txt"
    with open(diag_path, "w") as f:
        f.write("True-logit vector temperature argmax summary\n")
        f.write("===========================================\n")
        f.write(f"validation_negative_weight: {val_neg_weight}\n")
        f.write(f"test_negative_weight: {test_neg_weight}\n\n")

        f.write("Unweighted true-logit vector T:\n")
        f.write(f"T_nonsplice: {unweighted_t[0]}\n")
        f.write(f"T_acceptor: {unweighted_t[1]}\n")
        f.write(f"T_donor: {unweighted_t[2]}\n\n")

        f.write("Genome-weighted true-logit vector T:\n")
        f.write(f"T_nonsplice: {weighted_t[0]}\n")
        f.write(f"T_acceptor: {weighted_t[1]}\n")
        f.write(f"T_donor: {weighted_t[2]}\n\n")

        test_methods = {
            "uncalibrated": softmax_np(test_logits, axis=1),
            "global": apply_global_temperature_logits(test_logits, args.global_temperature),
            "logit_unweighted_vector": apply_vector_temperature_logits(test_logits, unweighted_t),
            "logit_weighted_vector": apply_vector_temperature_logits(test_logits, weighted_t),
        }

        for name, probs in test_methods.items():
            f.write(f"\n{name}\n")
            f.write("-" * len(name) + "\n")
            for cls, vals in argmax_summary(probs, test_labels).items():
                f.write(
                    f"{cls}: true={vals['true']}, predicted={vals['predicted']}, tp={vals['tp']}\n"
                )

    print("\nWrote:")
    print(csv_path)
    print(diag_path)

    print("\nCSV preview:")
    print(csv_path.read_text())

    print("\nArgmax preview:")
    print(diag_path.read_text())


if __name__ == "__main__":
    main()