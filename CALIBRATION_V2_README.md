# Calibration pipeline v2 patch

Created: 2026-07-28

This patch is a non-destructive correction for the calibrated-splice-prediction
calibration workflow. It adds new `_v2` scripts only. It does not replace the
historical scripts or modify any existing result directory.

## Why this patch is required

The source audit identified two confirmed defects:

1. `weighted_vector_temperature_from_cache.py` reconstructed logits with
   `log(probs)`. That is not equivalent to using the stored pre-softmax logits
   for class-specific temperatures.
2. `fit_openspliceai_style_vector_temperature_long.py` applied `argmax` to
   zero-labelled padding, turning padding into the non-splice class.

The validation-loader audit separately confirmed that
`get_validation_loader(...).drop_last` is `False`. Partial DataLoader batches
are retained. However, the installed `clip_datapoints` helper can remove batch
rows to make a batch divisible by the reported GPU count. The v2 fitter
therefore performs context clipping without removing sequences.

## New files

| New file | Role |
|---|---|
| `scripts/calibration_v2_common.py` | Schema-v2 cache validation, true-logit temperature application, metrics, and output guards |
| `scripts/vector_temperature_from_logits_cache_v2.py` | Fits both the sampled-cache unweighted vector temperature and target-position-weighted vector temperature from `logits_sample` |
| `scripts/fit_openspliceai_style_vector_temperature_long_v2.py` | Fits the full-validation OpenSpliceAI-style class-wise temperature after masking zero-labelled padding |
| `scripts/evaluate_openspliceai_style_temperature_on_logit_cache_v2.py` | Evaluates an OpenSpliceAI-style temperature on a corrected test cache |

The old probability-based weighted fitter is deliberately superseded by the
unified true-logit v2 fitter. Do not use
`scripts/weighted_vector_temperature_from_cache.py` for v2 results.

## Scientific method distinction preserved

- **Unweighted true-logit vector T:** fit on the sampled, splice-enriched
  validation cache.
- **Target-position-weighted true-logit vector T:** fit on the same sampled
  cache, with every sampled non-splice row weighted by
  `valid_nonsplice_positions_seen / sampled_negatives`.
- **OpenSpliceAI-style full-validation vector T:** fit without sampling on
  every valid labelled validation position. Zero-labelled padding is excluded.

The target-position weights reconstruct the valid-position population of the
evaluation H5. They do not claim a whole-chromosome intergenic prior.

## Installation

Place the patch archive in the repository root and run:

```bash
(
set -euo pipefail

cd ~/projects/calibrated-splice-prediction

archive="calibration_v2_patch_2026-07-28.tar.gz"

if [[ ! -f "$archive" ]]; then
    echo "ERROR: patch archive is missing: $archive"
    exit 1
fi

tar -tzf "$archive"
tar --no-same-owner -xzf "$archive"

sha256sum -c CALIBRATION_V2_MANIFEST.sha256

echo
echo "PASS: calibration v2 patch installed"
)
```

All installed script names are new. Existing historical scripts remain
untouched.

## Environment and import smoke test

```bash
(
set -euo pipefail

cd ~/projects/calibrated-splice-prediction
source .venv_wsl/bin/activate

python -m py_compile \
    scripts/calibration_v2_common.py \
    scripts/vector_temperature_from_logits_cache_v2.py \
    scripts/fit_openspliceai_style_vector_temperature_long_v2.py \
    scripts/evaluate_openspliceai_style_temperature_on_logit_cache_v2.py

python scripts/vector_temperature_from_logits_cache_v2.py --help >/dev/null
python scripts/fit_openspliceai_style_vector_temperature_long_v2.py --help >/dev/null
python scripts/evaluate_openspliceai_style_temperature_on_logit_cache_v2.py --help >/dev/null

echo "PASS: calibration v2 imports and CLI entry points are valid"
)
```

## Step 1: primary seed-11 true-logit calibration

Run this first. It validates both corrected caches before fitting either
temperature and refuses to write into an existing non-empty result directory.

```bash
(
set -euo pipefail

cd ~/projects/calibrated-splice-prediction
source .venv_wsl/bin/activate

script="scripts/vector_temperature_from_logits_cache_v2.py"
validation_cache="results/logit_cache_flank400_seed11_epoch11_fullpopulation_validonly_v2/validation_sampled_logits.npz"
test_cache="results/logit_cache_flank400_seed11_epoch11_fullpopulation_validonly_v2/test_sampled_logits.npz"
out_dir="results/logit_calibration_flank400_seed11_epoch11_fullpopulation_validonly_v2"
log="logs/logit_calibration_flank400_seed11_epoch11_fullpopulation_validonly_v2_2026-07-28.txt"

for required_file in \
    "$script" \
    "$validation_cache" \
    "$test_cache"
do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file missing: $required_file"
        exit 1
    fi
    ls -lh "$required_file"
done

if [[ -e "$out_dir" ]]; then
    echo "ERROR: refusing to reuse output directory: $out_dir"
    exit 1
fi

if [[ -e "$log" ]]; then
    echo "ERROR: refusing to overwrite log: $log"
    exit 1
fi

mkdir -p logs

python "$script" \
    --validation-cache "$validation_cache" \
    --test-cache "$test_cache" \
    --out-dir "$out_dir" \
    --global-temperature 1.1 \
    --n-bins 15 \
    --lr 0.01 \
    --steps 1000 \
    --print-every 100 \
    2>&1 | tee "$log"

for output in \
    "$out_dir/logit_vector_temperature_summary.csv" \
    "$out_dir/logit_vector_argmax_summary.txt" \
    "$out_dir/temperature_unweighted_best.txt" \
    "$out_dir/temperature_target_weighted_best.txt" \
    "$out_dir/calibration_v2_run_metadata.json"
do
    if [[ ! -s "$output" ]]; then
        echo "ERROR: expected output missing or empty: $output"
        exit 1
    fi
done

echo
echo "==== TEMPERATURES ===="
cat "$out_dir/temperature_unweighted_best.txt"
cat "$out_dir/temperature_target_weighted_best.txt"

echo
echo "==== OUTPUTS ===="
ls -lh "$out_dir" "$log"

echo
echo "PASS: primary seed-11 true-logit calibration v2 completed"
)
```

Expected cache census enforced by the script:

| Split | Rows | Non-splice | Acceptor | Donor | Non-splice weight |
|---|---:|---:|---:|---:|---:|
| Validation | 525,910 | 500,000 | 12,955 | 12,955 | 177.980664 |
| Test | 607,988 | 500,000 | 53,994 | 53,994 | 814.341268 |

The counts are read from and reconciled against the cache metadata; they are not
hard-coded into the fitter.

## Step 2: primary seed-11 full-validation OpenSpliceAI-style fit

Run this after Step 1. This is the slower model-inference fit. The explicit
expected census prevents a partial-population or padding-contaminated fit.

```bash
(
set -euo pipefail

cd ~/projects/calibrated-splice-prediction
source .venv_wsl/bin/activate

script="scripts/fit_openspliceai_style_vector_temperature_long_v2.py"
model="results/best_models/flank400_focal_epoch11_best.pt"
validation_h5="data/processed_h5_flank400/dataset_validation.h5"
out_dir="results/openspliceai_style_vectorT_flank400_seed11_epoch11_fullval_validonly_v2"
log="logs/openspliceai_style_vectorT_flank400_seed11_epoch11_fullval_validonly_v2_2026-07-28.txt"

for required_file in "$script" "$model" "$validation_h5"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file missing: $required_file"
        exit 1
    fi
    ls -lh "$required_file"
done

if [[ -e "$out_dir" ]]; then
    echo "ERROR: refusing to reuse output directory: $out_dir"
    exit 1
fi

if [[ -e "$log" ]]; then
    echo "ERROR: refusing to overwrite log: $log"
    exit 1
fi

mkdir -p logs

python "$script" \
    --pretrained-model "$model" \
    --validation-dataset "$validation_h5" \
    --output-dir "$out_dir" \
    --flanking-size 400 \
    --random-seed 11 \
    --epochs 1800 \
    --lr 0.01 \
    --print-every 25 \
    --save-every 200 \
    --expected-sequences 18486 \
    --expected-valid-positions 89016242 \
    --expected-padding-positions 3413758 \
    --expected-positive-positions 25910 \
    --expected-acceptor-positions 12955 \
    --expected-donor-positions 12955 \
    2>&1 | tee "$log"

for output in \
    "$out_dir/validation_census.json" \
    "$out_dir/training_log.tsv" \
    "$out_dir/temperature_best.txt" \
    "$out_dir/temperature_best.pt" \
    "$out_dir/fit_summary.json"
do
    if [[ ! -s "$output" ]]; then
        echo "ERROR: expected output missing or empty: $output"
        exit 1
    fi
done

echo
echo "==== BEST TEMPERATURE ===="
cat "$out_dir/temperature_best.txt"

echo
echo "==== OUTPUTS ===="
ls -lh "$out_dir" "$log"

echo
echo "PASS: primary full-validation OpenSpliceAI-style fit v2 completed"
)
```

To resume a stopped run, use the same `--output-dir`, set the final desired
`--epochs`, and add:

```text
--resume results/openspliceai_style_vectorT_flank400_seed11_epoch11_fullval_validonly_v2/checkpoint_latest.pt
```

The resume checkpoint must be inside the selected output directory and must
have been created by this v2 script.

## Step 3: evaluate the full-validation temperature on the corrected test cache

```bash
(
set -euo pipefail

cd ~/projects/calibrated-splice-prediction
source .venv_wsl/bin/activate

script="scripts/evaluate_openspliceai_style_temperature_on_logit_cache_v2.py"
cache="results/logit_cache_flank400_seed11_epoch11_fullpopulation_validonly_v2/test_sampled_logits.npz"
temperature="results/openspliceai_style_vectorT_flank400_seed11_epoch11_fullval_validonly_v2/temperature_best.txt"
out_dir="results/openspliceai_style_eval_flank400_seed11_epoch11_fullpopulation_validonly_v2"
log="logs/openspliceai_style_eval_flank400_seed11_epoch11_fullpopulation_validonly_v2_2026-07-28.txt"

for required_file in "$script" "$cache" "$temperature"; do
    if [[ ! -f "$required_file" ]]; then
        echo "ERROR: required file missing: $required_file"
        exit 1
    fi
    ls -lh "$required_file"
done

if [[ -e "$out_dir" ]]; then
    echo "ERROR: refusing to reuse output directory: $out_dir"
    exit 1
fi

if [[ -e "$log" ]]; then
    echo "ERROR: refusing to overwrite log: $log"
    exit 1
fi

mkdir -p logs

python "$script" \
    --cache "$cache" \
    --temperature-txt "$temperature" \
    --output-dir "$out_dir" \
    --method-name "openspliceai_style_full_validation_vectorT_v2" \
    --n-bins 15 \
    2>&1 | tee "$log"

for output in \
    "$out_dir/openspliceai_style_temperature_test_metrics.csv" \
    "$out_dir/openspliceai_style_temperature_threshold_metrics.csv" \
    "$out_dir/openspliceai_style_temperature_argmax.csv" \
    "$out_dir/evaluation_metadata.json"
do
    if [[ ! -s "$output" ]]; then
        echo "ERROR: expected output missing or empty: $output"
        exit 1
    fi
done

echo
echo "==== TEST METRICS ===="
cat "$out_dir/openspliceai_style_temperature_test_metrics.csv"

echo
echo "PASS: primary OpenSpliceAI-style test evaluation v2 completed"
)
```

## Important output definitions

- `weighted_multiclass_ece` and `weighted_multiclass_nll` reconstruct the
  corrected valid-position population using the cache metadata.
- `acceptor_auprc` and `donor_auprc` preserve the historical sampled-cache
  definition. Separately named target-weighted AUPRC columns are also written.
- Threshold output contains both sampled counts/metrics and reconstructed
  target-weighted counts/metrics.
- Every run records the cache SHA-256, schema version, population census,
  sampling seed, and fitted temperature.

## Historical-output boundary

Do not mix these v2 outputs with any result based on:

- 606,088 test rows;
- 106,088 test positives;
- non-splice test weight 832.107824;
- non-splice validation weight 181.749102;
- an old cache directory without `fullpopulation_validonly_v2`.

Reliability diagrams, bootstrap outputs, prior-sensitivity outputs, manuscript
tables, and cross-seed summaries must be regenerated from the corrected v2
caches after the three calibration methods have been refit.
