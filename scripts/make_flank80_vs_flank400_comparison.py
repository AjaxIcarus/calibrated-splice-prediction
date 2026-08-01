from pathlib import Path
import pandas as pd

OUT_DIR = Path("tables")
OUT_DIR.mkdir(exist_ok=True)

CONFIGS = [
    {
        "context": "flank80",
        "checkpoint": "epoch2",
        "vector_summary": Path("results/logit_vector_temperature_flank80_epoch2/logit_vector_temperature_summary.csv"),
        "detection": Path("results/logit_detection_metrics_flank80_epoch2/logit_detection_auprc_summary.csv"),
        "osai_style": Path("results/openspliceai_style_vectorT_flank80_epoch2_test_eval/openspliceai_style_temperature_test_metrics.csv"),
    },
    {
        "context": "flank400",
        "checkpoint": "epoch8",
        "vector_summary": Path("results/logit_vector_temperature_flank400_epoch8/logit_vector_temperature_summary.csv"),
        "detection": Path("results/logit_detection_metrics_flank400_epoch8/logit_detection_auprc_summary.csv"),
        "osai_style": Path("results/openspliceai_style_vectorT_flank400_epoch8_test_eval_converged/openspliceai_style_temperature_test_metrics.csv"),
    },
]

METHOD_MAP = [
    ("uncalibrated", "uncalibrated"),
    ("global_T_1.1", "global_T_1.1"),
    ("logit_unweighted_vector", "true-logit unweighted vector T"),
    ("logit_weighted_vector", "true-logit genome-weighted vector T"),
]

def get_detection_auprc(det_df, method_prefix):
    rows = det_df[det_df["method"].astype(str).str.startswith(method_prefix)]
    out = {}
    for cls in ["acceptor", "donor"]:
        r = rows[rows["class"] == cls]
        out[f"{cls}_auprc"] = float(r["auprc"].iloc[0]) if len(r) else float("nan")
    return out

def pick_vector_row(df, method_prefix):
    rows = df[
        (df["split"] == "test_sampled_weighted")
        & (df["method"].astype(str).str.startswith(method_prefix))
    ]
    if len(rows) == 0:
        raise RuntimeError(f"No row found for {method_prefix}")
    return rows.iloc[0]

records = []

for cfg in CONFIGS:
    vector_df = pd.read_csv(cfg["vector_summary"])
    det_df = pd.read_csv(cfg["detection"])

    for method_prefix, method_label in METHOD_MAP:
        row = pick_vector_row(vector_df, method_prefix)
        auprc = get_detection_auprc(det_df, method_prefix)

        records.append({
            "context": cfg["context"],
            "checkpoint": cfg["checkpoint"],
            "method": method_label,
            "weighted_multiclass_ece": row["multiclass_ece"],
            "weighted_nll": row["multiclass_nll"],
            "acceptor_ece": row["acceptor_ece"],
            "donor_ece": row["donor_ece"],
            "acceptor_auprc": auprc["acceptor_auprc"],
            "donor_auprc": auprc["donor_auprc"],
        })

    osai = pd.read_csv(cfg["osai_style"]).iloc[0]
    records.append({
        "context": cfg["context"],
        "checkpoint": cfg["checkpoint"],
        "method": "OpenSpliceAI-style vector T",
        "weighted_multiclass_ece": osai["weighted_multiclass_ece"],
        "weighted_nll": osai["weighted_nll"],
        "acceptor_ece": osai["acceptor_ece"],
        "donor_ece": osai["donor_ece"],
        "acceptor_auprc": osai["acceptor_auprc"],
        "donor_auprc": osai["donor_auprc"],
    })

df = pd.DataFrame(records)

# Sort in a stable paper-friendly order.
method_order = {
    "uncalibrated": 0,
    "global_T_1.1": 1,
    "true-logit unweighted vector T": 2,
    "true-logit genome-weighted vector T": 3,
    "OpenSpliceAI-style vector T": 4,
}
context_order = {"flank80": 0, "flank400": 1}

df["_context_order"] = df["context"].map(context_order)
df["_method_order"] = df["method"].map(method_order)
df = df.sort_values(["_context_order", "_method_order"]).drop(columns=["_context_order", "_method_order"])

exact_path = OUT_DIR / "flank80_vs_flank400_calibration_detection_exact.csv"
rounded_path = OUT_DIR / "flank80_vs_flank400_calibration_detection_rounded.csv"
md_path = OUT_DIR / "flank80_vs_flank400_calibration_detection.md"

df.to_csv(exact_path, index=False)

rounded = df.copy()
for col in [
    "weighted_multiclass_ece",
    "weighted_nll",
    "acceptor_ece",
    "donor_ece",
    "acceptor_auprc",
    "donor_auprc",
]:
    rounded[col] = rounded[col].map(lambda x: f"{float(x):.6g}")

rounded.to_csv(rounded_path, index=False)

with open(md_path, "w") as f:
    f.write("# Flank-80 vs flank-400 calibration and detection comparison\n\n")
    f.write(rounded.to_markdown(index=False))
    f.write("\n")

print("Wrote:")
print(exact_path)
print(rounded_path)
print(md_path)
print()
print(rounded.to_string(index=False))
