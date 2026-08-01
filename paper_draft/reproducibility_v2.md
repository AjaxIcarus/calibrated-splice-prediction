# Reproducibility

All analyses were run in a WSL Ubuntu environment using Python 3.12.3 and OpenSpliceAI 0.0.5. The analysis was performed under limited compute on CPU, without GPU acceleration.

## Data and model

The model was trained and evaluated using OpenSpliceAI-formatted HDF5 datasets derived from human GRCh38 sequence and MANE v1.3 splice annotations. The processed datasets were:

- `data/processed_h5/dataset_train.h5`
- `data/processed_h5/dataset_validation.h5`
- `data/processed_h5/dataset_test.h5`

All final analyses used the flank-80 focal-loss epoch-2 checkpoint:

- `results/best_models/flank80_focal_epoch2_best.pt`

The model performs per-position three-class classification over non-splice, acceptor, and donor labels.

## True-logit extraction

True pre-softmax logits were extracted by disabling the model’s final softmax layer during inference. The resulting validation and test caches were:

- `results/logit_cache_flank80_epoch2/validation_sampled_logits.npz`
- `results/logit_cache_flank80_epoch2/test_sampled_logits.npz`

Each cache contains sampled logits, probabilities, and one-hot labels. All positive splice-site positions were retained, and 500,000 non-splice positions were reservoir-sampled.

The full test set contained 416,160,000 genomic positions, including 106,088 splice-site positives and 416,053,912 non-splice positions. The sampled test cache contained 606,088 positions: 500,000 non-splice positions, 53,024 acceptor sites, and 53,064 donor sites.

## Calibration analysis

True-logit vector temperature scaling was fit on the sampled validation cache and evaluated on the sampled test cache. The final genome-prior weighted vector temperatures were:

- `T_nonsplice = 0.435169`
- `T_acceptor = 0.416153`
- `T_donor = 0.427941`

The main calibration outputs were:

- `results/logit_vector_temperature_flank80_epoch2/logit_vector_temperature_summary.csv`
- `results/logit_vector_temperature_flank80_epoch2/logit_vector_argmax_summary.txt`

## Detection, bootstrap, and prior-sensitivity outputs

Detection metrics were saved in:

- `results/logit_detection_metrics_flank80_epoch2/logit_detection_auprc_summary.csv`
- `results/logit_detection_metrics_flank80_epoch2/logit_detection_threshold_topk_summary.csv`

Bootstrap confidence intervals were computed using 200 nonparametric bootstrap replicates over the sampled test cache. The main bootstrap outputs were:

- `results/logit_bootstrap_flank80_epoch2/point_estimates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_replicates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_summary.csv`
- `tables/logit_based/logit_bootstrap_summary.md`

Reliability diagram outputs were saved in:

- `figures/logit_based/reliability_multiclass_uncalibrated.png`
- `figures/logit_based/reliability_multiclass_logit_weighted_vector.png`
- `figures/logit_based/reliability_acceptor_uncalibrated_vs_weighted.png`
- `figures/logit_based/reliability_donor_uncalibrated_vs_weighted.png`
- `tables/logit_based/logit_reliability_bins.csv`

Prior-sensitivity outputs were saved in:

- `results/logit_prior_sensitivity_flank80_epoch2/prior_sensitivity_summary.csv`
- `tables/logit_based/logit_prior_sensitivity_summary.md`
- `figures/logit_based/logit_prior_sensitivity_temperatures.png`
- `figures/logit_based/logit_prior_sensitivity_genome_metrics.png`
- `figures/logit_based/logit_prior_sensitivity_argmax_counts.png`

## Implementation

The final analysis used the following scripts:

- `scripts/extract_logits_cache.py`
- `scripts/vector_temperature_from_logits_cache.py`
- `scripts/detection_metrics_from_logits_cache.py`
- `scripts/make_logit_results_tables_and_figures.py`
- `scripts/bootstrap_logit_metrics.py`
- `scripts/make_logit_reliability_diagrams.py`
- `scripts/prior_sensitivity_logit_calibration.py`

Exact command-line calls for generating the caches, calibration outputs, detection metrics, bootstrap intervals, reliability diagrams, and prior-sensitivity analysis are provided in the supplementary materials.

## Reproducibility note

The large HDF5 datasets, model checkpoint, and `.npz` logit caches may be too large to store directly in a standard Git repository. The manuscript therefore tracks scripts, tables, figures, and result summaries, while large intermediate files should be stored separately or regenerated using the commands listed in the supplementary materials.
