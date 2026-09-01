> **ARCHIVED / SUPERSEDED RESEARCH DRAFT — NOT CURRENT TCBB SUBMISSION AUTHORITY**
>
> This file is retained as historical project-development material.
> Statements below using terms such as `current`, `final`, `primary`,
> `replication`, or older checkpoint identities describe the project state
> at the time this file was created; they do not define the current TCBB
> submission or corrected quantitative authority.
>
> **Current quantitative authority is seed 11/epoch 12 and seed 23/epoch 13.**
> See the repository root `README.md`,
> `CURRENT_SUBMISSION_AUTHORITY.md`, and
> `reproducibility/tcbb_v9/README.md`.
>
> The historical content below is intentionally preserved rather than
> rewritten so that project evolution remains auditable.

# Methods: Model calibration and evaluation

## Splice-site prediction model

We evaluated a flank-80 OpenSpliceAI-style splice-site prediction model trained on human GRCh38 sequence and MANE splice-site annotations. The task was formulated as per-position three-class classification over:

- non-splice
- acceptor
- donor

The model checkpoint used for all analyses was:

`results/best_models/flank80_focal_epoch2_best.pt`

The model was trained with focal loss to address the extreme class imbalance between abundant non-splice positions and rare acceptor/donor splice-site positions.

## Dataset construction and evaluation splits

The processed OpenSpliceAI HDF5 datasets were:

- `data/processed_h5/dataset_validation.h5`
- `data/processed_h5/dataset_test.h5`

The validation set was used for fitting calibration parameters. The test set was used only for final evaluation.

The full test set contained 416,160,000 genomic positions, of which 106,088 were splice-site positives and 416,053,912 were non-splice positions. Because evaluating every position repeatedly was computationally expensive and because positive splice sites are extremely sparse, we constructed sampled calibration/evaluation caches by retaining all positive splice-site positions and reservoir-sampling 500,000 non-splice positions.

The sampled test cache contained:

- 606,088 total sampled positions
- 500,000 sampled non-splice positions
- 53,024 acceptor positives
- 53,064 donor positives

This sampled distribution is strongly enriched for splice sites relative to the natural genome-wide class prior. Therefore, we evaluated both unweighted sampled calibration and genome-prior weighted calibration.

## True-logit extraction

Initial calibration experiments used probability-space proxies of the form:

`logits_proxy = log(probabilities)`

However, the trained model exposes true pre-softmax logits by disabling the final softmax operation. We therefore extracted true logits by setting:

`model.apply_softmax = False`

For sanity checking, we compared:

`softmax(true_logits)`

against the original model output with:

`model.apply_softmax = True`

The maximum absolute difference between the reconstructed probabilities and the original probabilities was approximately:

`1.19e-07`

for both validation and test caches, confirming that the extracted logits were valid.

The final true-logit caches were saved as:

- `results/logit_cache_flank80_epoch2/validation_sampled_logits.npz`
- `results/logit_cache_flank80_epoch2/test_sampled_logits.npz`

Each cache contained:

- `logits_sample`
- `probs_sample`
- `labels_sample`

## Temperature scaling

Temperature scaling was applied as a post-hoc calibration method. Model weights were frozen, and only calibration parameters were optimized on the sampled validation cache.

Given a logit vector:

`z = [z_nonsplice, z_acceptor, z_donor]`

and a temperature vector:

`T = [T_nonsplice, T_acceptor, T_donor]`

the calibrated probability vector was computed as:

`softmax(z / T)`

where division is elementwise.

We evaluated four calibration methods:

1. Uncalibrated model probabilities
2. Global scalar temperature scaling with `T = 1.1`
3. Unweighted true-logit vector temperature scaling
4. Genome-prior weighted true-logit vector temperature scaling

The unweighted vector model optimized calibration on the sampled validation distribution directly. The genome-weighted vector model corrected for the artificial enrichment of splice sites by assigning higher weight to sampled non-splice positions.

## Genome-prior weighting

Because the sampled validation and test caches retained all splice-site positives but only a subset of non-splice positions, the sampled class distribution did not reflect the natural genome-wide distribution.

For genome-prior weighted evaluation, sampled non-splice positions were weighted by the ratio:

`total_nonsplice_positions / sampled_nonsplice_positions`

For the test set, this weight was:

`416,053,912 / 500,000 = 832.107824`

The corresponding validation negative weight was:

`181.749102`

These weights approximate the natural genome-position prior while preserving a computationally manageable sampled cache.

## Calibration metrics

Calibration was evaluated using:

- multiclass expected calibration error
- multiclass negative log-likelihood
- class-specific acceptor ECE
- class-specific donor ECE
- reliability diagrams

For multiclass ECE, predictions were binned by confidence, where confidence was the maximum predicted class probability. For each bin, calibration error was computed as the absolute difference between average confidence and empirical accuracy. The final ECE was the weighted average of bin-wise calibration error.

For binary acceptor and donor reliability analyses, predicted acceptor or donor probabilities were binned directly and compared against the empirical frequency of the corresponding true class.

Negative log-likelihood was used to evaluate the likelihood assigned to the true class. Lower NLL indicates that predicted probabilities are better aligned with observed labels.

For genome-prior metrics, ECE and NLL were computed with non-splice sample weights to approximate the full genome-position prior.

## Detection and ranking metrics

Splice-site detection was evaluated separately from calibration. Detection asks whether true acceptor and donor positions are ranked above non-splice positions, rather than whether predicted probabilities are numerically calibrated.

We evaluated detection using:

- acceptor AUPRC
- donor AUPRC
- top-k precision and recall
- threshold-based precision and recall

Acceptor detection treated acceptor positions as positives and all other positions as negatives. Donor detection was evaluated analogously.

AUPRC was used as the main ranking metric because the task is highly imbalanced. Threshold precision and recall were computed for selected probability thresholds to understand operating-point behavior after calibration.

## Bootstrap confidence intervals

We estimated uncertainty in final calibration and detection metrics using nonparametric bootstrap resampling over the sampled test cache.

The bootstrap analysis used:

- 200 bootstrap replicates
- random seed 17

For each replicate, sampled positions were resampled with replacement. Metrics were recomputed for each calibration method. We reported point estimates and 95% percentile confidence intervals.

Bootstrap outputs were saved in:

- `results/logit_bootstrap_flank80_epoch2/point_estimates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_replicates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_summary.csv`

## Prior-sensitivity analysis

To test whether temperature scaling depends on the assumed class prior, we performed a prior-sensitivity analysis by sweeping the validation negative-class weight.

The tested validation negative weights were:

`1, 2, 5, 10, 25, 50, 100, 181.749`

For each weight, we refit the vector temperature model on the validation cache and evaluated genome-prior calibration on the test cache. We recorded:

- learned temperature parameters
- genome-prior ECE
- genome-prior NLL
- argmax prediction counts

This analysis tested whether calibration parameters and downstream interpretation change as the assumed non-splice prior approaches the natural genome-wide prior.

## Implementation

The main scripts used for true-logit calibration and evaluation were:

- `scripts/extract_logits_cache.py`
- `scripts/vector_temperature_from_logits_cache.py`
- `scripts/detection_metrics_from_logits_cache.py`
- `scripts/make_logit_results_tables_and_figures.py`
- `scripts/bootstrap_logit_metrics.py`
- `scripts/make_logit_reliability_diagrams.py`
- `scripts/prior_sensitivity_logit_calibration.py`

The main outputs were stored in:

- `results/logit_cache_flank80_epoch2/`
- `results/logit_vector_temperature_flank80_epoch2/`
- `results/logit_detection_metrics_flank80_epoch2/`
- `results/logit_bootstrap_flank80_epoch2/`
- `results/logit_prior_sensitivity_flank80_epoch2/`
- `tables/logit_based/`
- `figures/logit_based/`