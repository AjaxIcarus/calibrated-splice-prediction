import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


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


def summarize(name, probs, labels):
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(probs, axis=1)

    class_names = ["nonsplice", "acceptor", "donor"]

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("Accuracy:", accuracy_score(y_true, y_pred))

    print("\nTrue class counts:")
    for i, cname in enumerate(class_names):
        print(cname, int((y_true == i).sum()))

    print("\nPredicted class counts:")
    for i, cname in enumerate(class_names):
        print(cname, int((y_pred == i).sum()))

    print("\nMean predicted probabilities:")
    for i, cname in enumerate(class_names):
        print(cname, float(probs[:, i].mean()))

    print("\nMean predicted probability by true class:")
    for true_i, true_name in enumerate(class_names):
        mask = y_true == true_i
        print(f"\nTrue {true_name}, n={int(mask.sum())}")
        for pred_i, pred_name in enumerate(class_names):
            print(pred_name, float(probs[mask, pred_i].mean()))

    print("\nConfusion matrix rows=true, cols=pred:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    print("\nHigh-confidence splice predictions:")
    for i, cname in [(1, "acceptor"), (2, "donor")]:
        for threshold in [0.1, 0.5, 0.9]:
            print(
                f"{cname} p>={threshold}:",
                int((probs[:, i] >= threshold).sum()),
            )


def compare_predictions(name, base_probs, new_probs, labels):
    y_true = np.argmax(labels, axis=1)
    base_pred = np.argmax(base_probs, axis=1)
    new_pred = np.argmax(new_probs, axis=1)

    changed = base_pred != new_pred

    print("\n" + "=" * 80)
    print("Prediction changes:", name)
    print("=" * 80)
    print("Changed argmax count:", int(changed.sum()))
    print("Changed argmax fraction:", float(changed.mean()))

    print("\nChanged predictions by true class:")
    for i, cname in enumerate(["nonsplice", "acceptor", "donor"]):
        mask = y_true == i
        print(cname, int(changed[mask].sum()), "/", int(mask.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--global-temperature", type=float, default=1.1)
    parser.add_argument("--t-nonsplice", type=float, required=True)
    parser.add_argument("--t-acceptor", type=float, required=True)
    parser.add_argument("--t-donor", type=float, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.cache)
    probs = z["probs_sample"]
    labels = z["labels_sample"]

    global_probs = apply_global_temperature(probs, args.global_temperature)
    vector_probs = apply_vector_temperature(
        probs,
        [args.t_nonsplice, args.t_acceptor, args.t_donor],
    )

    log_path = out_dir / "diagnostics.txt"

    import sys
    old_stdout = sys.stdout

    with open(log_path, "w") as f:
        sys.stdout = f

        summarize("Uncalibrated", probs, labels)
        summarize(f"Global T={args.global_temperature}", global_probs, labels)
        summarize(
            f"Vector T=[{args.t_nonsplice}, {args.t_acceptor}, {args.t_donor}]",
            vector_probs,
            labels,
        )

        compare_predictions("global vs uncalibrated", probs, global_probs, labels)
        compare_predictions("vector vs uncalibrated", probs, vector_probs, labels)

    sys.stdout = old_stdout

    print("Wrote:", log_path)
    print(log_path.read_text())


if __name__ == "__main__":
    main()