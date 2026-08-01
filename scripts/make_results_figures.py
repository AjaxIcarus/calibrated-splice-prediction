from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


WEIGHTED_CALIBRATION_CSV = Path(
    "results/weighted_vector_temperature_flank80_epoch2/weighted_vector_temperature_summary.csv"
)

DETECTION_AUPRC_CSV = Path(
    "results/detection_metrics_flank80_epoch2/detection_auprc_summary.csv"
)

DETECTION_THRESHOLD_CSV = Path(
    "results/detection_metrics_flank80_epoch2/detection_threshold_topk_summary.csv"
)

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


METHOD_RENAME = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Global T=1.1",
    "old_unweighted_vector_T_0.0591_0.8414_0.7686": "Unweighted vector T",
    "weighted_vector_T_3.9132_0.5750_0.5771": "Genome-weighted vector T",
    "old_unweighted_vector": "Unweighted vector T",
    "weighted_vector": "Genome-weighted vector T",
}


def clean_method_name(x):
    return METHOD_RENAME.get(x, x)


def plot_calibration_bars():
    df = pd.read_csv(WEIGHTED_CALIBRATION_CSV)

    test_df = df[df["split"] == "test_sampled_weighted"].copy()
    test_df["method_clean"] = test_df["method"].apply(clean_method_name)

    order = [
        "Uncalibrated",
        "Global T=1.1",
        "Unweighted vector T",
        "Genome-weighted vector T",
    ]

    test_df["method_clean"] = pd.Categorical(
        test_df["method_clean"], categories=order, ordered=True
    )
    test_df = test_df.sort_values("method_clean")

    metrics = [
        ("multiclass_ece", "Weighted Multiclass ECE"),
        ("multiclass_nll", "Weighted Multiclass NLL"),
        ("acceptor_ece", "Weighted Acceptor ECE"),
        ("donor_ece", "Weighted Donor ECE"),
    ]

    for metric, ylabel in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(test_df["method_clean"], test_df[metric])
        plt.ylabel(ylabel)
        plt.xlabel("Calibration method")
        plt.xticks(rotation=25, ha="right")
        plt.title(f"Test-set calibration: {ylabel}")
        plt.tight_layout()

        out_path = FIG_DIR / f"flank80_epoch2_{metric}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Wrote:", out_path)


def plot_detection_auprc():
    df = pd.read_csv(DETECTION_AUPRC_CSV)
    df["method_clean"] = df["method"].apply(clean_method_name)

    order = [
        "Uncalibrated",
        "Global T=1.1",
        "Unweighted vector T",
        "Genome-weighted vector T",
    ]

    acceptor = df[df["class"] == "acceptor"].copy()
    donor = df[df["class"] == "donor"].copy()

    acceptor["method_clean"] = pd.Categorical(
        acceptor["method_clean"], categories=order, ordered=True
    )
    donor["method_clean"] = pd.Categorical(
        donor["method_clean"], categories=order, ordered=True
    )

    acceptor = acceptor.sort_values("method_clean")
    donor = donor.sort_values("method_clean")

    x = range(len(order))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], acceptor["auprc"], width, label="Acceptor")
    plt.bar([i + width / 2 for i in x], donor["auprc"], width, label="Donor")

    plt.xticks(list(x), order, rotation=25, ha="right")
    plt.ylabel("AUPRC")
    plt.xlabel("Calibration method")
    plt.ylim(0.98, 1.0)
    plt.title("Detection/ranking performance is stable across calibration methods")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "flank80_epoch2_detection_auprc.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Wrote:", out_path)


def plot_threshold_precision_recall():
    df = pd.read_csv(DETECTION_THRESHOLD_CSV)

    threshold_df = df[df["metric"].str.startswith("threshold_")].copy()
    threshold_df["threshold"] = threshold_df["metric"].str.replace(
        "threshold_", "", regex=False
    ).astype(float)

    threshold_df = threshold_df[
        threshold_df["method"].isin(["uncalibrated", "weighted_vector"])
    ].copy()

    threshold_df["method_clean"] = threshold_df["method"].apply(clean_method_name)

    for class_name in ["acceptor", "donor"]:
        class_df = threshold_df[threshold_df["class"] == class_name].copy()

        plt.figure(figsize=(8, 5))

        for method_name in ["Uncalibrated", "Genome-weighted vector T"]:
            mdf = class_df[class_df["method_clean"] == method_name].sort_values(
                "threshold"
            )

            plt.plot(
                mdf["threshold"],
                mdf["precision"],
                marker="o",
                label=f"{method_name} precision",
            )

            plt.plot(
                mdf["threshold"],
                mdf["recall"],
                marker="s",
                linestyle="--",
                label=f"{method_name} recall",
            )

        plt.xscale("log")
        plt.xlabel("Score threshold")
        plt.ylabel("Precision / Recall")
        plt.ylim(0, 1.05)
        plt.title(f"{class_name.capitalize()} threshold behavior")
        plt.legend()
        plt.tight_layout()

        out_path = FIG_DIR / f"flank80_epoch2_{class_name}_threshold_precision_recall.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Wrote:", out_path)


def make_summary_table():
    cal = pd.read_csv(WEIGHTED_CALIBRATION_CSV)
    det = pd.read_csv(DETECTION_AUPRC_CSV)

    test_cal = cal[cal["split"] == "test_sampled_weighted"].copy()
    test_cal["method_clean"] = test_cal["method"].apply(clean_method_name)

    det["method_clean"] = det["method"].apply(clean_method_name)

    acceptor = det[det["class"] == "acceptor"][
        ["method_clean", "auprc"]
    ].rename(columns={"auprc": "acceptor_auprc"})

    donor = det[det["class"] == "donor"][
        ["method_clean", "auprc"]
    ].rename(columns={"auprc": "donor_auprc"})

    merged = (
        test_cal[
            [
                "method_clean",
                "multiclass_ece",
                "multiclass_nll",
                "acceptor_ece",
                "donor_ece",
            ]
        ]
        .merge(acceptor, on="method_clean")
        .merge(donor, on="method_clean")
    )

    out_path = FIG_DIR / "flank80_epoch2_main_results_table.csv"
    merged.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print(merged)


def main():
    plot_calibration_bars()
    plot_detection_auprc()
    plot_threshold_precision_recall()
    make_summary_table()


if __name__ == "__main__":
    main()