import argparse
from pathlib import Path
import numpy as np


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def scalar(d, key, default=None):
    if key not in d.files:
        return default
    arr = d[key]
    return int(arr[0]) if arr.size else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-a", required=True)
    parser.add_argument("--cache-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-seed", type=int, default=999)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    a = np.load(args.cache_a)
    b = np.load(args.cache_b)

    for key in ["logits", "probs", "labels"]:
        if key not in a.files or key not in b.files:
            raise KeyError(f"Missing key {key} in one of the caches.")

    logits_a, probs_a, labels_a = a["logits"], a["probs"], a["labels"]
    logits_b, probs_b, labels_b = b["logits"], b["probs"], b["labels"]

    y_a = labels_a.argmax(axis=1)
    y_b = labels_b.argmax(axis=1)

    pos_a = y_a != 0
    neg_a = y_a == 0
    pos_b = y_b != 0
    neg_b = y_b == 0

    print("Cache A class counts:", np.bincount(y_a, minlength=3))
    print("Cache B class counts:", np.bincount(y_b, minlength=3))

    if pos_a.sum() != pos_b.sum():
        print("WARNING: positive count differs between caches.")

    # Keep positives once, combine negatives from both independent samples.
    logits = np.concatenate([logits_a[pos_a], logits_a[neg_a], logits_b[neg_b]], axis=0)
    probs = np.concatenate([probs_a[pos_a], probs_a[neg_a], probs_b[neg_b]], axis=0)
    labels = np.concatenate([labels_a[pos_a], labels_a[neg_a], labels_b[neg_b]], axis=0)

    perm = rng.permutation(len(labels))
    logits = logits[perm].astype(np.float32)
    probs = probs[perm].astype(np.float32)
    labels = labels[perm].astype(np.float32)

    y = labels.argmax(axis=1)
    class_counts = np.bincount(y, minlength=3)

    negatives_seen_total = scalar(a, "negatives_seen_total", None)
    total_positions_seen = scalar(a, "total_positions_seen", None)
    positive_positions_seen = scalar(a, "positive_positions_seen", int(class_counts[1] + class_counts[2]))

    max_abs_diff = float(np.max(np.abs(softmax_np(logits) - probs)))
    row_sum_error = float(np.max(np.abs(probs.sum(axis=1) - 1.0)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        logits=logits,
        probs=probs,
        labels=labels,
        split_name=np.array(["test"]),
        max_negatives=np.array([int(class_counts[0])], dtype=np.int64),
        sampled_negatives=np.array([int(class_counts[0])], dtype=np.int64),
        negatives_seen_total=np.array([negatives_seen_total], dtype=np.int64),
        total_positions_seen=np.array([total_positions_seen], dtype=np.int64),
        positive_positions_seen=np.array([positive_positions_seen], dtype=np.int64),
        random_seed=np.array([args.random_seed], dtype=np.int64),
        flanking_size=np.array([80], dtype=np.int64),
        max_abs_diff_softmax_logits_vs_probs=np.array([max_abs_diff], dtype=np.float64),
        prob_row_sum_max_error=np.array([row_sum_error], dtype=np.float64),
        source_cache_a=np.array([str(args.cache_a)]),
        source_cache_b=np.array([str(args.cache_b)]),
        note=np.array(["combined positives once + negatives from two independent 500k caches"]),
    )

    print("\nWrote:", out)
    print("Combined class counts [non, acceptor, donor]:", class_counts)
    print("Total sampled positions:", len(labels))
    print("max_abs_diff softmax(logits) vs probs:", max_abs_diff)
    print("prob row-sum max error:", row_sum_error)

    if negatives_seen_total is not None:
        print("Effective negative weight:", negatives_seen_total / int(class_counts[0]))


if __name__ == "__main__":
    main()
