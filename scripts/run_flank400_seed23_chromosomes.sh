#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv_wsl/bin/activate
mkdir -p logs

chroms=(chr1 chr3 chr5 chr7 chr9)

for chrom in "${chroms[@]}"; do
    out_dir="results/chromosome_eval_flank400_seed23_epoch14_${chrom}"
    log_file="logs/chromosome_eval_flank400_seed23_epoch14_${chrom}.log"
    success_file="${out_dir}/_SUCCESS"

    if [[ -f "$success_file" ]]; then
        echo "SKIP: ${chrom} already completed"
        continue
    fi

    mkdir -p "$out_dir"

    echo
    echo "============================================================"
    echo "STARTING ${chrom}: $(date -Is)"
    echo "============================================================"

    python scripts/evaluate_chromosome_streaming_flank400_methods.py \
      --datafile data/processed_h5_flank400/datafile_test.h5 \
      --dataset data/processed_h5_flank400/dataset_test.h5 \
      --model results/best_models/flank400_seed23_focal_epoch14_best.pt \
      --temperature-txt \
        results/openspliceai_style_vectorT_flank400_seed23_epoch14_fullval/temperature_best.txt \
      --unweighted-temperature \
        0.7575227618217468 \
        0.2961099445819855 \
        0.4080635905265808 \
      --weighted-temperature \
        0.47173458337783813 \
        0.5043124556541443 \
        0.473145455121994 \
      --global-temperature 1.1 \
      --chrom "$chrom" \
      --out-dir "$out_dir" \
      --flanking-size 400 \
      --n-bins 15 \
      2>&1 | tee "$log_file"

    python - "$out_dir" "$chrom" <<'PY'
from pathlib import Path
import sys

import numpy as np
import pandas as pd

out_dir = Path(sys.argv[1])
chrom = sys.argv[2]

required = [
    out_dir / f"{chrom}_gene_segment_map.csv",
    out_dir / f"{chrom}_streaming_metrics.csv",
    out_dir / f"{chrom}_threshold_metrics.csv",
    out_dir / f"{chrom}_argmax.csv",
    out_dir / f"{chrom}_multiclass_reliability_bins.csv",
]

missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"Missing outputs: {missing}")

metrics = pd.read_csv(out_dir / f"{chrom}_streaming_metrics.csv")

needed_columns = {
    "method",
    "chrom",
    "multiclass_ece",
    "multiclass_nll",
}
if not needed_columns.issubset(metrics.columns):
    raise SystemExit(
        f"Missing metric columns: "
        f"{sorted(needed_columns - set(metrics.columns))}"
    )

if len(metrics) != 5 or metrics["method"].nunique() != 5:
    raise SystemExit(
        f"Expected five methods; found {len(metrics)} rows and "
        f"{metrics['method'].nunique()} unique methods"
    )

if set(metrics["chrom"].astype(str)) != {chrom}:
    raise SystemExit("Chromosome label mismatch")

values = metrics[
    ["multiclass_ece", "multiclass_nll"]
].apply(pd.to_numeric, errors="coerce").to_numpy()

if not np.isfinite(values).all():
    raise SystemExit("Non-finite multiclass ECE/NLL detected")

print("\nVALIDATED")
print(
    metrics[
        ["method", "multiclass_ece", "multiclass_nll"]
    ].to_string(index=False)
)
PY

    date -Is > "$success_file"
    echo "COMPLETED ${chrom}: $(date -Is)"
done

echo
echo "ALL FIVE SEED-23 CHROMOSOME RUNS COMPLETED"
