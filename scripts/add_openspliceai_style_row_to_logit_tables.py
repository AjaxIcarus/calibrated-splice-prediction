#!/usr/bin/env python3

import os
import pandas as pd

MAIN_IN = "tables/logit_based/logit_main_results_table.csv"
THRESH_IN = "tables/logit_based/logit_threshold_results_table.csv"

NEW_MAIN = "results/openspliceai_style_vectorT_flank80_epoch2_test_eval/openspliceai_style_temperature_test_metrics.csv"
NEW_THRESH = "results/openspliceai_style_vectorT_flank80_epoch2_test_eval/openspliceai_style_temperature_threshold_metrics.csv"

MAIN_OUT_CSV = "tables/logit_based/logit_main_results_table_with_openspliceai_style.csv"
MAIN_OUT_MD = "tables/logit_based/logit_main_results_table_with_openspliceai_style.md"

THRESH_OUT_CSV = "tables/logit_based/logit_threshold_results_table_with_openspliceai_style.csv"
THRESH_OUT_MD = "tables/logit_based/logit_threshold_results_table_with_openspliceai_style.md"


def fmt4(x):
    return f"{float(x):.4f}"


def fmt6(x):
    return f"{float(x):.6f}"


def main():
    os.makedirs("tables/logit_based", exist_ok=True)

    main = pd.read_csv(MAIN_IN)
    new = pd.read_csv(NEW_MAIN).iloc[0]

    new_main_row = {
        "Method": "OpenSpliceAI-style unweighted vector T",
        "Weighted multiclass ECE": fmt6(new["weighted_multiclass_ece"]),
        "Weighted multiclass NLL": fmt6(new["weighted_nll"]),
        "Weighted acceptor ECE": fmt6(new["acceptor_ece"]),
        "Weighted donor ECE": fmt6(new["donor_ece"]),
        "Acceptor AUPRC": fmt4(new["acceptor_auprc"]),
        "Donor AUPRC": fmt4(new["donor_auprc"]),
    }

    main2 = pd.concat([main, pd.DataFrame([new_main_row])], ignore_index=True)
    main2.to_csv(MAIN_OUT_CSV, index=False)

    with open(MAIN_OUT_MD, "w") as f:
        f.write("# True-logit Main Results Table with OpenSpliceAI-style Baseline\n\n")
        f.write(main2.to_markdown(index=False))
        f.write("\n")

    thresh = pd.read_csv(THRESH_IN)
    newt = pd.read_csv(NEW_THRESH)

    cls_map = {1: "Acceptor", 2: "Donor"}
    method = "OpenSpliceAI-style unweighted vector T"

    rows = []
    for _, r in newt.iterrows():
        rows.append({
            "Method": method,
            "Class": cls_map[int(r["class"])],
            "Threshold": r["threshold"],
            "Precision": fmt4(r["precision"]),
            "Recall": fmt4(r["recall"]),
            "Predicted positives": int(r["predicted_positive"]),
            "True positives": int(r["tp"]),
        })

    thresh2 = pd.concat([thresh, pd.DataFrame(rows)], ignore_index=True)
    thresh2.to_csv(THRESH_OUT_CSV, index=False)

    with open(THRESH_OUT_MD, "w") as f:
        f.write("# True-logit Threshold Results Table with OpenSpliceAI-style Baseline\n\n")
        f.write(thresh2.to_markdown(index=False))
        f.write("\n")

    print("Wrote:")
    print(MAIN_OUT_CSV)
    print(MAIN_OUT_MD)
    print(THRESH_OUT_CSV)
    print(THRESH_OUT_MD)

    print("\nNew main table:")
    print(main2.to_markdown(index=False))

    print("\nNew OpenSpliceAI-style threshold rows:")
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
