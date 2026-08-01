import argparse
from pathlib import Path

import numpy as np
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


def detection_summary(scores, y_true, class_name):
    n_pos = int(y_true.sum())
    auprc = average_precision_score(y_true, scores)

    rows = []

    for k_mult in [1, 2, 5, 10]:
        k = min(len(scores), max(1, n_pos * k_mult))
        top_idx = np.argsort(scores)[::-1][:k]
        tp = int(y_true[top_idx].sum())
        precision = tp / k
        recall = tp / n_pos if n_pos > 0 else np.nan

        rows.append(
            {
                "class": class_name,
                "metric": f"top_{k_mult}x_positives",
                "k": k,
                "tp": tp,
                "precision": precision,
                "recall": recall,
            }
        )

    for threshold in [0.001, 0.01, 0.05, 0.1, 0.5, 0.9]:
        pred = scores >= threshold
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        n_pred = int(pred.sum())

        precision = tp / n_pred if n_pred > 0 else np.nan
        recall = tp / n_pos if n_pos > 0 else np.nan

        rows.append(
            {
                "class": class_name,
                "metric": f"threshold_{threshold}",
                "k": n_pred,
                "tp": tp,
                "precision": precision,
                "recall": recall,
            }
        )

    return auprc, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
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

    z = np.load(args.cache)
    logits = z["logits_sample"]
    labels = z["labels_sample"]

    methods = {
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

    auprc_rows = []
    detail_rows = []

    for method_name, probs in methods.items():
        for class_idx, class_name in [(1, "acceptor"), (2, "donor")]:
            y = labels[:, class_idx].astype(int)
            scores = probs[:, class_idx]

            auprc, rows = detection_summary(scores, y, class_name)

            auprc_rows.append(
                {
                    "method": method_name,
                    "class": class_name,
                    "auprc": auprc,
                    "positives": int(y.sum()),
                    "mean_score_positive": float(scores[y == 1].mean()),
                    "mean_score_negative": float(scores[y == 0].mean()),
                    "max_score_negative": float(scores[y == 0].max()),
                }
            )

            for row in rows:
                row["method"] = method_name
                detail_rows.append(row)

    auprc_path = out_dir / "logit_detection_auprc_summary.csv"
    detail_path = out_dir / "logit_detection_threshold_topk_summary.csv"

    with open(auprc_path, "w") as f:
        header = [
            "method",
            "class",
            "auprc",
            "positives",
            "mean_score_positive",
            "mean_score_negative",
            "max_score_negative",
        ]
        f.write(",".join(header) + "\n")
        for row in auprc_rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    with open(detail_path, "w") as f:
        header = [
            "method",
            "class",
            "metric",
            "k",
            "tp",
            "precision",
            "recall",
        ]
        f.write(",".join(header) + "\n")
        for row in detail_rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")

    print("Wrote:")
    print(auprc_path)
    print(detail_path)
    print("\nAUPRC summary:")
    print(auprc_path.read_text())
    print("\nThreshold/top-k summary:")
    print(detail_path.read_text())


if __name__ == "__main__":
    main()