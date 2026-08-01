import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_file(csv_path, out_path, title, empirical_col):
    df = pd.read_csv(csv_path)

    df = df[df["count"] > 0].copy()
    df["midpoint"] = (df["lo"] + df["hi"]) / 2

    plt.figure()
    plt.plot(df["midpoint"], df["mean_confidence"], marker="o", label="Mean confidence")
    plt.plot(df["midpoint"], df[empirical_col], marker="o", label="Empirical")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Confidence bin")
    plt.ylabel("Probability / frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_file(
        in_dir / "uncalibrated_acceptor_reliability_bins.csv",
        out_dir / "uncalibrated_acceptor_reliability.png",
        "Uncalibrated Acceptor Reliability",
        "empirical_frequency",
    )

    plot_file(
        in_dir / "temperature_scaled_acceptor_reliability_bins.csv",
        out_dir / "temperature_scaled_acceptor_reliability.png",
        "Temperature-Scaled Acceptor Reliability",
        "empirical_frequency",
    )

    plot_file(
        in_dir / "uncalibrated_donor_reliability_bins.csv",
        out_dir / "uncalibrated_donor_reliability.png",
        "Uncalibrated Donor Reliability",
        "empirical_frequency",
    )

    plot_file(
        in_dir / "temperature_scaled_donor_reliability_bins.csv",
        out_dir / "temperature_scaled_donor_reliability.png",
        "Temperature-Scaled Donor Reliability",
        "empirical_frequency",
    )

    plot_file(
        in_dir / "uncalibrated_multiclass_reliability_bins.csv",
        out_dir / "uncalibrated_multiclass_reliability.png",
        "Uncalibrated Multiclass Reliability",
        "empirical_accuracy",
    )

    plot_file(
        in_dir / "temperature_scaled_multiclass_reliability_bins.csv",
        out_dir / "temperature_scaled_multiclass_reliability.png",
        "Temperature-Scaled Multiclass Reliability",
        "empirical_accuracy",
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()