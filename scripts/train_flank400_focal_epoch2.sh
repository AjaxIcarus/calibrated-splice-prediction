#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/calibrated-splice-prediction
source .venv_wsl/bin/activate

mkdir -p results/full_train_flank400_focal_epoch2_lowmem logs

openspliceai train \
  --epochs 1 \
  --scheduler MultiStepLR \
  --output-dir results/full_train_flank400_focal_epoch2_lowmem/ \
  --project-name flank400_full_focal_epoch2_lowmem \
  --exp-num 0 \
  --flanking-size 400 \
  --random-seed 11 \
  --train-dataset data/processed_h5_flank400/dataset_train.h5 \
  --test-dataset data/processed_h5_flank400/dataset_validation.h5 \
  --loss focal_loss \
  --model results/best_models/flank400_focal_epoch1_best.pt \
  2>&1 | tee logs/train_flank400_focal_epoch2.log
