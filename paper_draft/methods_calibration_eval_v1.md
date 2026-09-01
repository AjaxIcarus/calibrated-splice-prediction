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

# Methods

## Model and prediction task

We evaluated a flank-80 OpenSpliceAI-style splice-site prediction model trained on human GRCh38 sequence and MANE splice-site annotations. The model performs per-position three-class classification, assigning each nucleotide position to one of three classes: non-splice, acceptor, or donor. All analyses used the epoch-2 focal-loss checkpoint:

`results/best_models/flank80_focal_epoch2_best.pt`

Focal loss was used during training to address the extreme imbalance between abundant non-splice positions and rare splice-site positions.

## Evaluation data

Calibration parameters were fit on the validation set and evaluated on the held-out test set:

- `data/processed_h5/dataset_validation.h5`
- `data/processed_h5/dataset_test.h5`

The full test set contained 416,160,000 genomic positions, including 106,088 splice-site positives and 416,053,912 non-splice positions. To make repeated calibration and bootstrap analysis feasible, we constructed sampled caches that retained all positive splice-site positions and reservoir-sampled 500,000 non-splice positions.

The sampled test cache contained 606,088 positions: 500,000 non-splice positions, 53,024 acceptor sites, and 53,064 donor sites. Because this sampled distribution is strongly enriched for splice sites relative to the natural genome-wide distribution, we evaluated calibration both on the sampled distribution and under a genome-prior weighted distribution.

## True-logit extraction

Early calibration experiments used probability-derived logit proxies, computed as `log(probability)`. For the final analysis, we instead extracted true pre-softmax logits from the model. The OpenSpliceAI-style model returns softmax probabilities by default, but setting:

`model.apply_softmax = False`

returns the final pre-softmax logits. We verified the extraction by comparing `softmax(logits)` to the model’s default probability output. The maximum absolute difference was approximately `1.19e-07` for both validation and test caches.

The final true-logit caches were:

- `results/logit_cache_flank80_epoch2/validation_sampled_logits.npz`
- `results/logit_cache_flank80_epoch2/test_sampled_logits.npz`

Each cache stored sampled logits, probabilities, and one-hot labels.

## Temperature scaling

Temperature scaling was applied as a post-hoc calibration method. The trained model weights were frozen, and only temperature parameters were optimized on the sampled validation cache.

For a logit vector:

`z = [z_nonsplice, z_acceptor, z_donor]`

and a temperature vector:

`T = [T_nonsplice, T_acceptor, T_donor]`

calibrated probabilities were computed as:

`softmax(z / T)`

where division is elementwise. We compared four calibration settings:

1. uncalibrated model probabilities;
2. global scalar temperature scaling with `T = 1.1`;
3. unweighted true-logit vector temperature scaling;
4. genome-prior weighted true-logit vector temperature scaling.

The unweighted vector model optimized calibration on the sampled validation distribution. The genome-prior weighted vector model corrected for splice enrichment by assigning larger weights to sampled non-splice positions.

## Genome-prior weighting

Because the sampled caches included all splice sites but only a subset of non-splice positions, the sampled class distribution did not match the natural genome-wide class prior. To approximate the natural prior, sampled non-splice positions were weighted by:

`total non-splice positions / sampled non-splice positions`

For the test set, this gave:

`416,053,912 / 500,000 = 832.107824`

The corresponding validation negative weight was `181.749102`.

## Calibration metrics

Calibration was evaluated using expected calibration error, negative log-likelihood, and reliability diagrams.

For multiclass ECE, predictions were binned by confidence, defined as the maximum predicted class probability. Within each bin, we computed the absolute difference between mean confidence and empirical accuracy. The final ECE was the weighted average of these bin-level errors.

For acceptor and donor reliability analyses, predicted acceptor or donor probabilities were binned directly and compared with the empirical frequency of the corresponding class. Negative log-likelihood measured the probability assigned to the true class, with lower values indicating better calibrated likelihoods.

Genome-prior ECE and NLL were computed using non-splice sample weights to approximate the full genome-position distribution.

## Detection metrics

Splice-site detection was evaluated separately from probability calibration. Acceptor detection treated acceptor positions as positives and all other sampled positions as negatives. Donor detection was evaluated analogously.

Detection was measured using:

- acceptor AUPRC;
- donor AUPRC;
- top-k precision and recall;
- threshold-based precision and recall.

AUPRC was used as the primary ranking metric because splice-site prediction is highly imbalanced. Threshold precision and recall were used to evaluate how calibration changes operating-point behavior.

## Bootstrap confidence intervals

We estimated confidence intervals using nonparametric bootstrap resampling over the sampled test cache. We generated 200 bootstrap replicates using random seed 17. For each replicate, sampled positions were resampled with replacement and all calibration and detection metrics were recomputed.

Bootstrap outputs were saved in:

- `results/logit_bootstrap_flank80_epoch2/point_estimates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_replicates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_summary.csv`

We report 95% percentile confidence intervals.

## Prior-sensitivity analysis

To test whether calibration depends on the assumed class prior, we refit vector temperature scaling while sweeping the validation non-splice weight across:

`1, 2, 5, 10, 25, 50, 100, 181.749`

For each weight, we recorded the learned temperature vector, genome-prior ECE, genome-prior NLL, and argmax prediction counts on the test cache. This analysis assessed whether calibrated probabilities and downstream predictions change as the calibration prior approaches the natural genome-wide prior.

## Implementation

The final analysis used the following scripts:

- `scripts/extract_logits_cache.py`
- `scripts/vector_temperature_from_logits_cache.py`
- `scripts/detection_metrics_from_logits_cache.py`
- `scripts/make_logit_results_tables_and_figures.py`
- `scripts/bootstrap_logit_metrics.py`
- `scripts/make_logit_reliability_diagrams.py`
- `scripts/prior_sensitivity_logit_calibration.py`
