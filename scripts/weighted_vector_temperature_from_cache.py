import argparse
from pathlib import Path

import numpy as np
import torch


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def apply_global_temperature(probs, t):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return softmax_np(logits / t, axis=1)


def apply_vector_temperature(probs, temps):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
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
    return float(np.sum(weights * (-np.log(p))) / np.sum(weights))


def weighted_multiclass_brier(probs, labels, weights):
    b = np.sum((probs - labels) ** 2, axis=1)
    return float(np.sum(weights * b) / np.sum(weights))


def weighted_binary_nll(p, y, weights):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.sum(weights * loss) / np.sum(weights))


def weighted_binary_brier(p, y, weights):
    b = (p - y) ** 2
    return float(np.sum(weights * b) / np.sum(weights))


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


def weighted_binary_ece(p, y, weights, n_bins=15):
    p = np.asarray(p)
    y = np.asarray(y)
    weights = np.asarray(weights)

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


def metrics(probs, labels, weights, n_bins):
    out = {}

    out["multiclass_ece"] = weighted_multiclass_ece(probs, labels, weights, n_bins)
    out["multiclass_nll"] = weighted_multiclass_nll(probs, labels, weights)
    out["multiclass_brier"] = weighted_multiclass_brier(probs, labels, weights)

    for class_idx, class_name in [(1, "acceptor"), (2, "donor")]:
        y = labels[:, class_idx]
        p = probs[:, class_idx]

        out[f"{class_name}_ece"] = weighted_binary_ece(p, y, weights, n_bins)
        out[f"{class_name}_nll"] = weighted_binary_nll(p, y, weights)
        out[f"{class_name}_brier"] = weighted_binary_brier(p, y, weights)

    return out


def fit_weighted_vector_temperature(val_probs, val_labels, val_weights, lr=0.01, steps=1000):
    device = torch.device("cpu")

    logits = torch.tensor(
        np.log(np.clip(val_probs, 1e-12, 1.0)),
        dtype=torch.float32,
        device=device,
    )

    y = torch.tensor(
        np.argmax(val_labels, axis=1),
        dtype=torch.long,
        device=device,
    )

    w = torch.tensor(
        val_weights / np.mean(val_weights),
        dtype=torch.float32,
        device=device,
    )

    raw_t = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
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

        loss = torch.sum(w * per_sample_loss) / torch.sum(w)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())

        if loss_value < best_loss:
            best_loss = loss_value
            best_t = temps.detach().cpu().numpy().copy()

        if step % 100 == 0:
            print(
                f"step={step} weighted_loss={loss_value:.8f} "
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
        true_count = int((y_true == i).sum())
        pred_count = int((y_pred == i).sum())
        tp = int(((y_true == i) & (y_pred == i)).sum())
        rows[name] = (true_count, pred_count, tp)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--global-temperature", type=float, default=1.1)
    parser.add_argument("--old-t-nonsplice", type=float, required=True)
    parser.add_argument("--old-t-acceptor", type=float, required=True)
    parser.add_argument("--old-t-donor", type=float, required=True)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_cache = np.load(args.validation_cache)
    test_cache = np.load(args.test_cache)

    val_probs = val_cache["probs_sample"]
    val_labels = val_cache["labels_sample"]
    test_probs = test_cache["probs_sample"]
    test_labels = test_cache["labels_sample"]

    val_weights, val_neg_weight = make_weights(val_labels, val_cache)
    test_weights, test_neg_weight = make_weights(test_labels, test_cache)

    print("Validation sample:", val_probs.shape)
    print("Test sample:", test_probs.shape)
    print("Validation negative weight:", val_neg_weight)
    print("Test negative weight:", test_neg_weight)

    weighted_t, weighted_val_loss = fit_weighted_vector_temperature(
        val_probs,
        val_labels,
        val_weights,
        lr=args.lr,
        steps=args.steps,
    )

    print("\nWeighted validation-fitted temperature:")
    print("T_nonsplice:", weighted_t[0])
    print("T_acceptor:", weighted_t[1])
    print("T_donor:", weighted_t[2])
    print("Best weighted validation NLL:", weighted_val_loss)

    old_t = [args.old_t_nonsplice, args.old_t_acceptor, args.old_t_donor]

    rows = []

    for split_name, probs, labels, weights in [
        ("validation_sampled_weighted", val_probs, val_labels, val_weights),
        ("test_sampled_weighted", test_probs, test_labels, test_weights),
    ]:
        methods = {
            "uncalibrated": probs,
            f"global_T_{args.global_temperature}": apply_global_temperature(
                probs,
                args.global_temperature,
            ),
            f"old_unweighted_vector_T_{old_t[0]:.4f}_{old_t[1]:.4f}_{old_t[2]:.4f}": apply_vector_temperature(
                probs,
                old_t,
            ),
            f"weighted_vector_T_{weighted_t[0]:.4f}_{weighted_t[1]:.4f}_{weighted_t[2]:.4f}": apply_vector_temperature(
                probs,
                weighted_t,
            ),
        }

        for method_name, method_probs in methods.items():
            m = metrics(method_probs, labels, weights, args.n_bins)
            row = {"split": split_name, "method": method_name}
            row.update(m)
            rows.append(row)

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

    csv_path = out_dir / "weighted_vector_temperature_summary.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    diag_path = out_dir / "weighted_argmax_summary.txt"
    with open(diag_path, "w") as f:
        f.write("Weighted vector temperature argmax summary\n")
        f.write("=========================================\n")
        f.write(f"validation_negative_weight: {val_neg_weight}\n")
        f.write(f"test_negative_weight: {test_neg_weight}\n")
        f.write(f"weighted_T_nonsplice: {weighted_t[0]}\n")
        f.write(f"weighted_T_acceptor: {weighted_t[1]}\n")
        f.write(f"weighted_T_donor: {weighted_t[2]}\n\n")

        for name, p in {
            "uncalibrated": test_probs,
            "global": apply_global_temperature(test_probs, args.global_temperature),
            "old_unweighted_vector": apply_vector_temperature(test_probs, old_t),
            "weighted_vector": apply_vector_temperature(test_probs, weighted_t),
        }.items():
            f.write(f"\n{name}\n")
            f.write("-" * len(name) + "\n")
            for cls, vals in argmax_summary(p, test_labels).items():
                true_count, pred_count, tp = vals
                f.write(
                    f"{cls}: true={true_count}, predicted={pred_count}, tp={tp}\n"
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