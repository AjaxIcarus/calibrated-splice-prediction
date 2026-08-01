from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CALIBRATION_CSV = Path(
    "results/logit_vector_temperature_flank80_epoch2/logit_vector_temperature_summary.csv"
)

AUPRC_CSV = Path(
    "results/logit_detection_metrics_flank80_epoch2/logit_detection_auprc_summary.csv"
)

THRESHOLD_CSV = Path(
    "results/logit_detection_metrics_flank80_epoch2/logit_detection_threshold_topk_summary.csv"
)

FIG_DIR = Path("figures/logit_based")
TABLE_DIR = Path("tables/logit_based")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


METHOD_RENAME = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Global T=1.1",
    "logit_unweighted_vector_T_0.8119_0.2486_0.3451": "Logit unweighted vector T",
    "logit_weighted_vector_T_0.4352_0.4162_0.4279": "Logit genome-weighted vector T",
    "logit_unweighted_vector": "Logit unweighted vector T",
    "logit_weighted_vector": "Logit genome-weighted vector T",
}


ORDER = [
    "Uncalibrated",
    "Global T=1.1",
    "Logit unweighted vector T",
    "Logit genome-weighted vector T",
]


def clean_method(x):
    return METHOD_RENAME.get(x, x)


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def load_main_table():
    cal = pd.read_csv(CALIBRATION_CSV)
    auprc = pd.read_csv(AUPRC_CSV)

    cal = cal[cal["split"] == "test_sampled_weighted"].copy()
    cal["Method"] = cal["method"].apply(clean_method)

    auprc["Method"] = auprc["method"].apply(clean_method)

    acc_auprc = auprc[auprc["class"] == "acceptor"][["Method", "auprc"]].rename(
        columns={"auprc": "Acceptor AUPRC"}
    )
    don_auprc = auprc[auprc["class"] == "donor"][["Method", "auprc"]].rename(
        columns={"auprc": "Donor AUPRC"}
    )

    table = cal[
        [
            "Method",
            "multiclass_ece",
            "multiclass_nll",
            "acceptor_ece",
            "donor_ece",
        ]
    ].rename(
        columns={
            "multiclass_ece": "Weighted multiclass ECE",
            "multiclass_nll": "Weighted multiclass NLL",
            "acceptor_ece": "Weighted acceptor ECE",
            "donor_ece": "Weighted donor ECE",
        }
    )

    table = table.merge(acc_auprc, on="Method").merge(don_auprc, on="Method")
    table["Method"] = pd.Categorical(table["Method"], categories=ORDER, ordered=True)
    table = table.sort_values("Method")

    return table


def write_tables():
    main = load_main_table()

    rounded = main.copy()

    calibration_cols = [
        "Weighted multiclass ECE",
        "Weighted multiclass NLL",
        "Weighted acceptor ECE",
        "Weighted donor ECE",
    ]

    auprc_cols = [
        "Acceptor AUPRC",
        "Donor AUPRC",
    ]

    for col in calibration_cols:
        rounded[col] = rounded[col].map(lambda x: f"{x:.6f}")

    for col in auprc_cols:
        rounded[col] = rounded[col].map(lambda x: f"{x:.4f}")

    csv_out = TABLE_DIR / "logit_main_results_table.csv"
    md_out = TABLE_DIR / "logit_main_results_table.md"

    rounded.to_csv(csv_out, index=False)

    with open(md_out, "w") as f:
        f.write("# True-logit Main Results Table\n\n")
        f.write(rounded.to_markdown(index=False, disable_numparse=True))
        f.write("\n")

    thresh = pd.read_csv(THRESHOLD_CSV)
    thresh = thresh[
        (thresh["metric"].isin(["threshold_0.01", "threshold_0.05", "threshold_0.1", "threshold_0.5"]))
        & (thresh["method"].isin(["uncalibrated", "logit_weighted_vector"]))
    ].copy()

    thresh["Method"] = thresh["method"].apply(clean_method)
    thresh["Class"] = thresh["class"].str.capitalize()
    thresh["Threshold"] = thresh["metric"].str.replace("threshold_", "", regex=False)

    threshold_table = thresh[
        ["Method", "Class", "Threshold", "precision", "recall", "k", "tp"]
    ].rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "k": "Predicted positives",
            "tp": "True positives",
        }
    )

    threshold_table["Precision"] = threshold_table["Precision"].map(lambda x: fmt(x, 4))
    threshold_table["Recall"] = threshold_table["Recall"].map(lambda x: fmt(x, 4))

    threshold_csv_out = TABLE_DIR / "logit_threshold_results_table.csv"
    threshold_md_out = TABLE_DIR / "logit_threshold_results_table.md"

    threshold_table.to_csv(threshold_csv_out, index=False)

    with open(threshold_md_out, "w") as f:
        f.write("# True-logit Threshold Results Table\n\n")
        f.write(threshold_table.to_markdown(index=False, disable_numparse=True))
        f.write("\n")

    print("Wrote:")
    print(csv_out)
    print(md_out)
    print(threshold_csv_out)
    print(threshold_md_out)

    print("\nMain table:")
    print(rounded.to_markdown(index=False))

    print("\nThreshold table:")
    print(threshold_table.to_markdown(index=False))


def plot_calibration_bars():
    main = load_main_table()

    plots = [
        ("Weighted multiclass ECE", "logit_multiclass_ece.png"),
        ("Weighted multiclass NLL", "logit_multiclass_nll.png"),
        ("Weighted acceptor ECE", "logit_acceptor_ece.png"),
        ("Weighted donor ECE", "logit_donor_ece.png"),
    ]

    for metric, filename in plots:
        plt.figure(figsize=(8, 5))
        plt.bar(main["Method"], main[metric])
        plt.ylabel(metric)
        plt.xlabel("Calibration method")
        plt.xticks(rotation=25, ha="right")
        plt.title(f"True-logit calibration: {metric}")
        plt.tight_layout()

        out_path = FIG_DIR / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Wrote:", out_path)


def plot_detection_auprc():
    df = pd.read_csv(AUPRC_CSV)
    df["Method"] = df["method"].apply(clean_method)

    acceptor = df[df["class"] == "acceptor"].copy()
    donor = df[df["class"] == "donor"].copy()

    acceptor["Method"] = pd.Categorical(acceptor["Method"], categories=ORDER, ordered=True)
    donor["Method"] = pd.Categorical(donor["Method"], categories=ORDER, ordered=True)

    acceptor = acceptor.sort_values("Method")
    donor = donor.sort_values("Method")

    x = range(len(ORDER))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], acceptor["auprc"], width, label="Acceptor")
    plt.bar([i + width / 2 for i in x], donor["auprc"], width, label="Donor")

    plt.xticks(list(x), ORDER, rotation=25, ha="right")
    plt.ylabel("AUPRC")
    plt.xlabel("Calibration method")
    plt.ylim(0.985, 1.0)
    plt.title("True-logit detection AUPRC across calibration methods")
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "logit_detection_auprc.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Wrote:", out_path)


def plot_threshold_precision_recall():
    df = pd.read_csv(THRESHOLD_CSV)

    threshold_df = df[df["metric"].str.startswith("threshold_")].copy()
    threshold_df["threshold"] = threshold_df["metric"].str.replace(
        "threshold_", "", regex=False
    ).astype(float)

    threshold_df = threshold_df[
        threshold_df["method"].isin(["uncalibrated", "logit_weighted_vector"])
    ].copy()

    threshold_df["Method"] = threshold_df["method"].apply(clean_method)

    for class_name in ["acceptor", "donor"]:
        class_df = threshold_df[threshold_df["class"] == class_name].copy()

        plt.figure(figsize=(8, 5))

        for method_name in ["Uncalibrated", "Logit genome-weighted vector T"]:
            mdf = class_df[class_df["Method"] == method_name].sort_values("threshold")

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
        plt.title(f"True-logit {class_name.capitalize()} threshold behavior")
        plt.legend()
        plt.tight_layout()

        out_path = FIG_DIR / f"logit_{class_name}_threshold_precision_recall.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Wrote:", out_path)


def main():
    write_tables()
    plot_calibration_bars()
    plot_detection_auprc()
    plot_threshold_precision_recall()


if __name__ == "__main__":
    main()