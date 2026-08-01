#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/projects/calibrated-splice-prediction"
SEED=23
PATIENCE=2

cd "$ROOT"
source .venv_wsl/bin/activate
mkdir -p results/best_models logs

validation_state() {
    local current_epoch="$1"

    python - "$current_epoch" "$PATIENCE" <<'PY'
from pathlib import Path
import re
import sys

current = int(sys.argv[1])
patience = int(sys.argv[2])
root = Path.home() / "projects" / "calibrated-splice-prediction"
pattern = re.compile(r"Validation Loss:\s*([0-9.eE+-]+)")

rows = []
for epoch in range(1, current + 1):
    log = root / "logs" / f"train_flank400_seed23_focal_epoch{epoch}.log"
    matches = pattern.findall(log.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"Missing validation loss in {log}")
    rows.append((epoch, float(matches[-1])))

best_epoch, best_loss = min(rows, key=lambda row: row[1])
epochs_without_improvement = current - best_epoch
stop = int(epochs_without_improvement >= patience)

print(best_epoch, repr(best_loss), epochs_without_improvement, stop)
PY
}

for epoch in $(seq 10 15); do
    previous_epoch=$((epoch - 1))

    previous_checkpoint="results/best_models/flank400_seed23_focal_epoch${previous_epoch}_best.pt"
    checkpoint="results/best_models/flank400_seed23_focal_epoch${epoch}_best.pt"
    outdir="results/full_train_flank400_seed23_focal_epoch${epoch}_lowmem"
    project="flank400_seed23_full_focal_epoch${epoch}_lowmem"
    log="logs/train_flank400_seed23_focal_epoch${epoch}.log"

    if [[ ! -f "$previous_checkpoint" ]]; then
        echo "ERROR: Missing $previous_checkpoint"
        exit 1
    fi

    if [[ -f "$checkpoint" ]]; then
        echo "Epoch ${epoch} already complete; checking early stopping."
    else
        if [[ -e "$outdir" ]]; then
            echo "ERROR: Incomplete output directory exists: $outdir"
            exit 1
        fi

        echo "Starting seed 23, epoch ${epoch}"

        openspliceai train \
          --epochs 1 \
          --scheduler MultiStepLR \
          --output-dir "$outdir/" \
          --project-name "$project" \
          --exp-num 0 \
          --flanking-size 400 \
          --random-seed "$SEED" \
          --train-dataset data/processed_h5_flank400/dataset_train.h5 \
          --test-dataset data/processed_h5_flank400/dataset_validation.h5 \
          --loss focal_loss \
          --model "$previous_checkpoint" \
          2>&1 | tee "$log"

        mapfile -t candidates < <(
            find "$outdir" -type f -path '*/models/model_best.pt'
        )

        if [[ ${#candidates[@]} -ne 1 ]]; then
            echo "ERROR: Expected exactly one model_best.pt"
            exit 1
        fi

        cp -p "${candidates[0]}" "$checkpoint"
        sha256sum "$checkpoint"
    fi

    read -r best_epoch best_loss bad_epochs should_stop \
        < <(validation_state "$epoch")

    echo "Current best: epoch ${best_epoch}, validation loss ${best_loss}"
    echo "Epochs without improvement: ${bad_epochs}/${PATIENCE}"

    if [[ "$should_stop" == "1" ]]; then
        echo "Early stopping triggered after epoch ${epoch}."
        break
    fi
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

for epoch in range(1, 16):
    log = root / "logs" / f"train_flank400_seed23_focal_epoch{epoch}.log"
    checkpoint = (
        root / "results/best_models"
        / f"flank400_seed23_focal_epoch{epoch}_best.pt"
    )

    if not log.exists() or not checkpoint.exists():
        continue

    matches = pattern.findall(log.read_text(errors="replace"))
    if matches:
        rows.append({
            "epoch": epoch,
            "validation_loss": float(matches[-1]),
            "checkpoint": str(checkpoint.relative_to(root)),
        })

best = min(rows, key=lambda row: row["validation_loss"])

table = root / "results/flank400_seed23_validation_selection.csv"
with table.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

source = root / best["checkpoint"]
alias = root / "results/best_models/flank400_seed23_focal_best.pt"
shutil.copy2(source, alias)

digest = hashlib.sha256(alias.read_bytes()).hexdigest()

selection = root / "results/best_models/flank400_seed23_selection.txt"
selection.write_text(
    "seed: 23\n"
    "maximum epochs: 15\n"
    "early-stopping patience: 2\n"
    "criterion: minimum validation loss\n"
    f"selected epoch: {best['epoch']}\n"
    f"validation loss: {best['validation_loss']:.17g}\n"
    f"checkpoint: {best['checkpoint']}\n"
    f"sha256: {digest}\n"
)

print(f"Selected epoch: {best['epoch']}")
print(f"Validation loss: {best['validation_loss']:.17g}")
print(f"Checkpoint: {alias}")
print(f"SHA256: {digest}")
PY
