from pathlib import Path

import pandas as pd


OUT_DIR = Path("tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

calibration_csv = Path(
    "results/weighted_vector_temperature_flank80_epoch2/weighted_vector_temperature_summary.csv"
)

auprc_csv = Path(
    "results/detection_metrics_flank80_epoch2/detection_auprc_summary.csv"
)

threshold_csv = Path(
    "results/detection_metrics_flank80_epoch2/detection_threshold_topk_summary.csv"
)


METHOD_RENAME = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Global T=1.1",
    "old_unweighted_vector_T_0.0591_0.8414_0.7686": "Unweighted vector T",
    "weighted_vector_T_3.9132_0.5750_0.5771": "Genome-weighted vector T",
    "old_unweighted_vector": "Unweighted vector T",
    "weighted_vector": "Genome-weighted vector T",
}


def clean_method(x):
    return METHOD_RENAME.get(x, x)


def fmt(x, digits=4):
    return f"{x:.{digits}f}"


def main():
    cal = pd.read_csv(calibration_csv)
    auprc = pd.read_csv(auprc_csv)
    thresh = pd.read_csv(threshold_csv)

    cal = cal[cal["split"] == "test_sampled_weighted"].copy()
    cal["Method"] = cal["method"].apply(clean_method)

    auprc["Method"] = auprc["method"].apply(clean_method)

    acc_auprc = auprc[auprc["class"] == "acceptor"][["Method", "auprc"]].rename(
        columns={"auprc": "Acceptor AUPRC"}
    )
    don_auprc = auprc[auprc["class"] == "donor"][["Method", "auprc"]].rename(
        columns={"auprc": "Donor AUPRC"}
    )

    main = cal[
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

    main = main.merge(acc_auprc, on="Method").merge(don_auprc, on="Method")

    method_order = [
        "Uncalibrated",
        "Global T=1.1",
        "Unweighted vector T",
        "Genome-weighted vector T",
    ]

    main["Method"] = pd.Categorical(main["Method"], categories=method_order, ordered=True)
    main = main.sort_values("Method")

    rounded = main.copy()
    for col in rounded.columns:
        if col != "Method":
            rounded[col] = rounded[col].map(lambda x: fmt(x, 4))

    csv_out = OUT_DIR / "main_results_table.csv"
    md_out = OUT_DIR / "main_results_table.md"

    rounded.to_csv(csv_out, index=False)

    with open(md_out, "w") as f:
        f.write("# Main Results Table\n\n")
        f.write(rounded.to_markdown(index=False))
        f.write("\n")

    # Threshold table: only selected practical thresholds
    thresh = thresh[
        (thresh["metric"].isin(["threshold_0.01", "threshold_0.05", "threshold_0.1", "threshold_0.5"]))
        & (thresh["method"].isin(["uncalibrated", "weighted_vector"]))
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

    threshold_csv_out = OUT_DIR / "threshold_results_table.csv"
    threshold_md_out = OUT_DIR / "threshold_results_table.md"

    threshold_table.to_csv(threshold_csv_out, index=False)

    with open(threshold_md_out, "w") as f:
        f.write("# Threshold Results Table\n\n")
        f.write(threshold_table.to_markdown(index=False))
        f.write("\n")

    print("Wrote:")
    print(csv_out)
    print(md_out)
    print(threshold_csv_out)
    print(threshold_md_out)

    print("\nMain results table:")
    print(rounded.to_markdown(index=False))

    print("\nThreshold table:")
    print(threshold_table.to_markdown(index=False))


if __name__ == "__main__":
    main()