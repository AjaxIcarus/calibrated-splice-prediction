#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$HOME/projects/calibrated-splice-prediction"
SEED=23

cd "$PROJECT_ROOT"
source .venv_wsl/bin/activate

mkdir -p results/best_models logs

for epoch in $(seq 1 9); do
    OUTDIR="results/full_train_flank400_seed23_focal_epoch${epoch}_lowmem"
    PROJECT_NAME="flank400_seed23_full_focal_epoch${epoch}_lowmem"
    LOG="logs/train_flank400_seed23_focal_epoch${epoch}.log"
    SAVED_CHECKPOINT="results/best_models/flank400_seed23_focal_epoch${epoch}_best.pt"

    if [[ -f "$SAVED_CHECKPOINT" ]]; then
        echo "Epoch ${epoch} already complete: $SAVED_CHECKPOINT"
        continue
    fi

    if [[ -e "$OUTDIR" ]]; then
        echo "ERROR: Incomplete output directory already exists:"
        echo "$OUTDIR"
        echo "Inspect it before retrying; nothing has been deleted."
        exit 1
    fi

    TRAIN_ARGS=(
        openspliceai train
        --epochs 1
        --scheduler MultiStepLR
        --output-dir "$OUTDIR/"
        --project-name "$PROJECT_NAME"
        --exp-num 0
        --flanking-size 400
        --random-seed "$SEED"
        --train-dataset data/processed_h5_flank400/dataset_train.h5
        --test-dataset data/processed_h5_flank400/dataset_validation.h5
        --loss focal_loss
    )

    if (( epoch > 1 )); then
        PREVIOUS_EPOCH=$((epoch - 1))
        PREVIOUS_CHECKPOINT="results/best_models/flank400_seed23_focal_epoch${PREVIOUS_EPOCH}_best.pt"

        if [[ ! -f "$PREVIOUS_CHECKPOINT" ]]; then
            echo "ERROR: Missing continuation checkpoint:"
            echo "$PREVIOUS_CHECKPOINT"
            exit 1
        fi

        TRAIN_ARGS+=(--model "$PREVIOUS_CHECKPOINT")
    fi

    echo "Starting seed ${SEED}, epoch ${epoch}"
    "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG"

    mapfile -t FOUND_CHECKPOINTS < <(
        find "$OUTDIR" -type f -path '*/models/model_best.pt' -print
    )

    if [[ ${#FOUND_CHECKPOINTS[@]} -ne 1 ]]; then
        echo "ERROR: Expected exactly one model_best.pt after epoch ${epoch}."
        printf '%s\n' "${FOUND_CHECKPOINTS[@]}"
        exit 1
    fi

    cp -p "${FOUND_CHECKPOINTS[0]}" "$SAVED_CHECKPOINT"
    sha256sum "$SAVED_CHECKPOINT"

    echo "Completed seed ${SEED}, epoch ${epoch}"
done

python - <<'PY'
from pathlib import Path
import csv
import hashlib
import re
import shutil

root = Path.home() / "projects" / "calibrated-splice-prediction"
pattern = re.compile(r"Validation Loss:\s*([0-9.eE+-]+)")
rows = []

for epoch in range(1, 10):
    log = root / "logs" / f"train_flank400_seed23_focal_epoch{epoch}.log"
    checkpoint = (
        root / "results" / "best_models"
        / f"flank400_seed23_focal_epoch{epoch}_best.pt"
    )

    if not log.is_file():
        raise FileNotFoundError(log)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    matches = pattern.findall(log.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"No validation loss found in {log}")

    rows.append(
        {
            "epoch": epoch,
            "validation_loss": float(matches[-1]),
            "checkpoint": str(checkpoint.relative_to(root)),
        }
    )

best = min(rows, key=lambda row: row["validation_loss"])

summary_csv = (
    root / "results" / "flank400_seed23_validation_selection.csv"
)
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["epoch", "validation_loss", "checkpoint"],
    )
    writer.writeheader()
    writer.writerows(rows)

source = root / best["checkpoint"]
alias = (
    root / "results" / "best_models"
    / "flank400_seed23_focal_best.pt"
)
shutil.copy2(source, alias)

digest = hashlib.sha256(alias.read_bytes()).hexdigest()
selection_file = (
    root / "results" / "best_models"
    / "flank400_seed23_selection.txt"
)
selection_file.write_text(
    f"seed: 23\n"
    f"selection criterion: minimum validation loss\n"
    f"best epoch: {best['epoch']}\n"
    f"validation loss: {best['validation_loss']:.17g}\n"
    f"source checkpoint: {best['checkpoint']}\n"
    f"alias checkpoint: {alias.relative_to(root)}\n"
    f"sha256: {digest}\n"
)

print("\nSeed-23 validation results:")
for row in rows:
    marker = " <-- selected" if row["epoch"] == best["epoch"] else ""
    print(
        f"epoch {row['epoch']}: "
        f"{row['validation_loss']:.17g}{marker}"
    )

print(f"\nSelected checkpoint: {alias}")
print(f"SHA256: {digest}")
print(f"Table: {summary_csv}")
PY
