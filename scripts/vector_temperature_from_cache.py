import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss, brier_score_loss


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def multiclass_ece(probs, labels, n_bins=15):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (y_true == y_pred).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_conf = float(conf[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)

    return ece


def binary_ece(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        bin_conf = float(probs[mask].mean())
        bin_freq = float(labels[mask].mean())
        ece += (count / n) * abs(bin_freq - bin_conf)

    return ece


def apply_global_temperature(probs, t):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return softmax_np(logits / t, axis=1)


def apply_vector_temperature(probs, temps):
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    temps = np.asarray(temps, dtype=np.float32).reshape(1, 3)
    return softmax_np(logits / temps, axis=1)


def metrics(probs, labels, n_bins):
    y_true = np.argmax(labels, axis=1)

    out = {}
    out["multiclass_ece"] = multiclass_ece(probs, labels, n_bins=n_bins)
    out["multiclass_nll"] = log_loss(y_true, np.clip(probs, 1e-12, 1.0), labels=[0, 1, 2])
    out["multiclass_brier"] = np.mean(np.sum((probs - labels) ** 2, axis=1))

    for class_idx, class_name in [(1, "acceptor"), (2, "donor")]:
        y = labels[:, class_idx]
        p = np.clip(probs[:, class_idx], 1e-12, 1.0 - 1e-12)

        out[f"{class_name}_ece"] = binary_ece(p, y, n_bins=n_bins)
        out[f"{class_name}_nll"] = log_loss(y, p, labels=[0, 1])
        out[f"{class_name}_brier"] = brier_score_loss(y, p)

    return out


def fit_vector_temperature(val_probs, val_labels, lr=0.01, steps=1000):
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

    raw_t = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([raw_t], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_t = None

    for step in range(steps):
        optimizer.zero_grad()

        temps = 0.05 + 4.95 * torch.sigmoid(raw_t)
        scaled_logits = logits / temps.reshape(1, 3)

        loss = loss_fn(scaled_logits, y)
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

    val = np.load(args.validation_cache)
    test = np.load(args.test_cache)

    val_probs = val["probs_sample"]
    val_labels = val["labels_sample"]
    test_probs = test["probs_sample"]
    test_labels = test["labels_sample"]

    print("Validation sample:", val_probs.shape)
    print("Test sample:", test_probs.shape)

    best_t, best_loss = fit_vector_temperature(
        val_probs,
        val_labels,
        lr=args.lr,
        steps=args.steps,
    )

    print("\nBest vector temperature:")
    print("T_nonsplice:", best_t[0])
    print("T_acceptor:", best_t[1])
    print("T_donor:", best_t[2])
    print("Best validation NLL:", best_loss)

    rows = []

    for split_name, probs, labels in [
        ("validation_sampled", val_probs, val_labels),
        ("test_sampled", test_probs, test_labels),
    ]:
        methods = {
            "uncalibrated": probs,
            f"global_T_{args.global_temperature}": apply_global_temperature(
                probs, args.global_temperature
            ),
            f"vector_T_nonsplice_{best_t[0]:.4f}_acceptor_{best_t[1]:.4f}_donor_{best_t[2]:.4f}": apply_vector_temperature(
                probs, best_t
            ),
        }

        for method_name, method_probs in methods.items():
            m = metrics(method_probs, labels, args.n_bins)
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

    csv_path = out_dir / "vector_temperature_summary.csv"

    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    txt_path = out_dir / "vector_temperature_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Vector temperature scaling from cached predictions\n")
        f.write("=================================================\n")
        f.write(f"T_nonsplice: {best_t[0]}\n")
        f.write(f"T_acceptor: {best_t[1]}\n")
        f.write(f"T_donor: {best_t[2]}\n")
        f.write(f"Best validation NLL: {best_loss}\n\n")
        f.write(csv_path.read_text())

    print("\nWrote:")
    print(csv_path)
    print(txt_path)
    print("\nCSV preview:")
    print(csv_path.read_text())


if __name__ == "__main__":
    main()