> **ARCHIVED / SUPERSEDED RESEARCH DRAFT — NOT CURRENT TCBB SUBMISSION AUTHORITY**
>
> This file is preserved unchanged below for development traceability.
> It may contain historical flank-80 results, earlier flank-400 checkpoints,
> earlier population definitions, or manuscript wording that was superseded
> during reproducibility closure and TCBB revision.
>
> **Do not use this file as current quantitative or submission authority.**
> Current repository quantitative authority is seed 11/epoch 12 and
> seed 23/epoch 13. See the root `README.md`,
> `CURRENT_SUBMISSION_AUTHORITY.md`, and
> `reproducibility/tcbb_v9/README.md`.

# Prior-dependent calibration for OpenSpliceAI-style splice-site prediction under extreme class imbalance

## Abstract

Deep learning models for splice-site prediction are commonly evaluated as detection systems, but their predicted probabilities are also often interpreted as confidence estimates. In this work, we study probability calibration for a flank-80 OpenSpliceAI-style model trained on human GRCh38/MANE splice annotations. We show that calibration depends strongly on the assumed class prior. Using true pre-softmax logits, genome-prior weighted vector temperature scaling substantially improves weighted multiclass calibration while preserving acceptor and donor ranking performance. These results suggest that, in highly imbalanced per-nucleotide splice-site prediction, probability calibration and splice-site detection should be evaluated as separate objectives.

## Introduction

Alternative splicing is a central mechanism for transcript diversity and gene regulation. Computational models of splicing aim to predict splice-site usage, splice junctions, and the effects of sequence variation from primary sequence. Recent deep learning models such as SpliceAI and OpenSpliceAI formulate splice-site prediction as a per-position sequence classification task, assigning probabilities to acceptor, donor, and non-splice positions.

Most evaluation of these models emphasizes detection: whether true splice sites are ranked above non-splice positions. However, model scores are often interpreted as confidence values. This raises a separate calibration question: when a model assigns a probability to a splice-site prediction, does that probability match the empirical frequency of the event?

This question is especially important because per-nucleotide splice-site prediction is extremely imbalanced. Across the genome, almost all positions are non-splice positions, while acceptor and donor sites are rare. A model can rank splice sites well while still producing probabilities that are miscalibrated under the natural genome-wide prior.

In this study, we evaluate this calibration-detection distinction using a flank-80 OpenSpliceAI-style model trained with focal loss. We extract true pre-softmax logits, fit vector temperature scaling models, compare unweighted and genome-prior weighted calibration, and evaluate detection using AUPRC and threshold precision/recall. We also use bootstrap confidence intervals, reliability diagrams, and prior-sensitivity analysis to test whether the calibration behavior is robust.

Our central claim is that probability calibration and splice-site detection are separate objectives in highly imbalanced splice-site prediction. Calibration must be interpreted relative to an explicit class prior, while detection should be evaluated using ranking and threshold-based metrics.

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

# Results: True-logit calibration reveals prior-dependent uncertainty in splice-site prediction

## True-logit calibration setup

We evaluated calibration for a flank-80 OpenSpliceAI-style splice-site predictor trained with focal loss on GRCh38/MANE splice annotations. The model predicts three per-nucleotide classes: non-splice, acceptor, and donor. Instead of calibrating log-probability proxies, we extracted true pre-softmax logits by disabling the model’s final softmax layer and verified that applying softmax to the extracted logits exactly reproduced the model probabilities.

For calibration and detection analysis, we used sampled validation and test sets that retained all positive splice-site positions and reservoir-sampled 500,000 non-splice positions. The sampled test set contained 606,088 positions: 500,000 non-splice positions, 53,024 acceptor sites, and 53,064 donor sites. Because this sampled set is strongly splice-enriched relative to the natural genome-wide class distribution, we evaluated both unweighted calibration and genome-prior weighted calibration.

## Genome-prior weighted vector scaling improves calibration

Global temperature scaling with T=1.1 did not improve genome-prior calibration. Weighted multiclass ECE increased from 0.005610 to 0.008538, and weighted multiclass NLL increased from 0.006240 to 0.009212. This suggests that a single scalar temperature is too coarse for the non-splice/acceptor/donor prediction problem.

True-logit vector temperature scaling substantially improved calibration when fit under the genome-wide class prior. The genome-weighted vector temperature model used:

- T_nonsplice = 0.435169
- T_acceptor = 0.416153
- T_donor = 0.427941

This reduced weighted multiclass ECE from 0.005610 to 0.000044 and weighted multiclass NLL from 0.006240 to 0.000818. Bootstrap confidence intervals confirmed that the improvement was stable: weighted multiclass ECE was 0.000049 [0.000039, 0.000058], and weighted multiclass NLL was 0.000818 [0.000804, 0.000832].

These results indicate that the model’s raw confidence scores are not directly interpretable under the natural genome-position prior, but can be strongly recalibrated using true-logit vector scaling with explicit prior weighting.

## Detection remains strong after calibration

Calibration changed the probability scale but did not substantially affect splice-site ranking. The uncalibrated model achieved acceptor AUPRC of 0.9909 and donor AUPRC of 0.9950 on the sampled test set. After genome-weighted true-logit vector scaling, acceptor AUPRC remained 0.9909 and donor AUPRC remained 0.9950.

Bootstrap analysis showed the same pattern. Under genome-weighted vector scaling, acceptor AUPRC was 0.990906 [0.990350, 0.991494], and donor AUPRC was 0.994977 [0.994573, 0.995355]. Thus, the model ranks true splice sites above non-splice positions very well, even when the calibrated probability values are adjusted.

This separates two evaluation goals: probability calibration and splice-site detection. Calibration asks whether predicted probabilities match empirical frequencies under a chosen prior. Detection asks whether true splice sites receive higher scores than non-splice positions. In this task, calibration is prior-sensitive, while ranking performance remains high.

## Threshold behavior depends on calibration

Although AUPRC changed little, threshold-based precision and recall changed after calibration. For example, the uncalibrated model at threshold 0.5 had very high precision but low recall: acceptor precision was 1.0000 with recall 0.0750, and donor precision was 0.9999 with recall 0.1432.

After genome-weighted vector calibration, a lower threshold such as 0.01 gave a more useful operating point: acceptor precision was 0.9801 with recall 0.8954, and donor precision was 0.9852 with recall 0.9403. This shows that calibrated probabilities should not be interpreted through arbitrary default thresholds such as 0.5. Instead, thresholds should be selected according to the target operating point and the assumed class prior.

## Prior sensitivity explains the calibration behavior

Prior-sensitivity analysis showed that learned temperatures and downstream predictions depend strongly on the assumed negative-class weight. As the validation negative weight increased toward the genome-wide prior, genome-prior ECE and NLL improved, while the number of argmax splice predictions decreased.

This demonstrates that temperature scaling is not prior-neutral in extremely imbalanced per-position splice-site prediction. Unweighted calibration fits the artificially splice-enriched sampled distribution. Genome-weighted calibration fits the natural class distribution, where non-splice positions dominate.

## Argmax prediction is misleading under extreme class imbalance

The genome-weighted vector-calibrated model produced excellent probability calibration, but argmax predictions remained dominated by the non-splice class. This is expected because the natural genome-wide prior contains vastly more non-splice positions than splice sites.

Therefore, argmax classification is not the right operating mode for splice-site discovery. A calibrated model can assign low absolute probabilities to splice sites while still ranking them highly relative to non-splice positions. Splice-site discovery should therefore use ranking metrics, threshold curves, and operating-point-specific precision/recall rather than multiclass argmax accuracy alone.

## Main result

These experiments show that in highly imbalanced per-nucleotide splice-site prediction, probability calibration and splice-site detection are separate objectives. The flank-80 focal-loss model ranks acceptor and donor sites very well, but calibrated probability interpretation depends strongly on the assumed class prior. Genome-weighted true-logit vector scaling gives the best genome-prior calibration, while AUPRC confirms that splice-site detection remains strong.

# Discussion: Calibration and detection are separate objectives in splice-site prediction

Deep learning models for splice-site prediction are usually evaluated as detection systems: do true acceptor and donor sites receive higher scores than non-splice positions? In this work, we asked a related but distinct question: when a model assigns a probability to a splice-site prediction, should that probability be interpreted as calibrated uncertainty?

Our results show that the answer depends strongly on the assumed class prior. The flank-80 focal-loss OpenSpliceAI-style model ranked splice sites very well, with acceptor and donor AUPRC near 0.991 and 0.995. However, its probability calibration changed substantially depending on whether calibration was performed under the splice-enriched sampled distribution or under the natural genome-wide prior.

## Main finding

The main finding is that genome-prior weighted true-logit vector temperature scaling substantially improved probability calibration while preserving detection performance. Weighted multiclass ECE decreased from 0.005610 to 0.000044, and weighted multiclass NLL decreased from 0.006240 to 0.000818. At the same time, acceptor and donor AUPRC remained essentially unchanged.

This supports a calibration-detection separation:

- detection/ranking asks whether true splice sites are scored above non-splice positions
- calibration asks whether predicted probabilities match empirical frequencies under an explicit prior

In highly imbalanced per-nucleotide splice-site prediction, these are not the same objective.

## Why prior weighting matters

The sampled validation and test caches intentionally retained all positive splice sites while subsampling non-splice positions. This made evaluation computationally feasible and gave enough positive examples for stable analysis. However, it also produced a splice-enriched distribution that differs sharply from the natural genome-wide distribution.

A calibration method fit on this sampled distribution learns probabilities for the sampled task, not for the genome-wide task. This is why unweighted vector temperature scaling could produce good calibration on the sampled distribution but worse genome-prior NLL. Once non-splice examples were weighted to approximate their natural abundance, the learned temperature vector shifted and genome-prior calibration improved strongly.

This suggests that calibration should always state the target prior. Without specifying the prior, a calibrated probability is ambiguous.

## Why argmax is misleading

Argmax classification is not a suitable operating mode for genome-wide splice-site discovery. Under the natural genome-wide prior, the non-splice class dominates. A well-calibrated model can assign low absolute probabilities to true splice sites while still ranking them above nearby non-splice candidates.

Therefore, a low calibrated probability does not necessarily mean a site is unimportant for discovery. It may simply reflect the extreme rarity of splice sites among all genomic positions. For discovery and annotation tasks, ranking metrics, AUPRC, top-k recall, and threshold-specific precision/recall are more appropriate than multiclass argmax accuracy.

## Thresholds require calibration-aware interpretation

Although AUPRC remained stable after calibration, threshold behavior changed. This matters because downstream users often apply fixed probability thresholds. A threshold such as 0.5 is arbitrary in an extremely imbalanced genome-wide prediction task. It produced very high precision but low recall.

After genome-prior calibration, lower thresholds such as 0.01 produced much more useful recall while maintaining high precision. This shows that calibration does not simply make scores “better”; it changes the probability scale. Thresholds should therefore be selected based on the desired operating point, class prior, and downstream application.

## Relationship to existing splice prediction work

Splice-site prediction models such as SpliceAI and OpenSpliceAI are usually motivated by accurate identification of acceptor and donor positions from DNA sequence. This work does not replace that detection framing. Instead, it adds a probability-interpretation layer: when the model gives a score, what distribution does that score correspond to?

OpenSpliceAI already includes temperature-scaling-based calibration. Our analysis extends this direction by focusing on true pre-softmax logits, explicit genome-prior weighting, bootstrap confidence intervals, reliability diagrams, and prior-sensitivity analysis for an extremely imbalanced per-position prediction setting.

The key distinction is that strong detection performance does not automatically imply well-calibrated genome-wide probabilities.

## Practical implication

For practical splice-site discovery, the calibrated model should be used in two stages:

1. Use ranking or threshold-based detection to identify candidate acceptor and donor sites.
2. Interpret calibrated probabilities only relative to the prior used during calibration.

This means that a model calibrated under a genome-wide prior is appropriate when probabilities are meant to represent absolute genome-position likelihoods. A model calibrated under a splice-enriched prior may still be useful for candidate-ranking or benchmark settings, but its probabilities should not be interpreted as genome-wide event probabilities.

## Limitations

This analysis used a flank-80 model trained for only two epochs under limited compute. Longer-context models are known to capture broader splice regulatory context, and future work should test whether the same calibration-prior behavior holds for flank-400, flank-2000, or flank-10000 models.

The analysis also used MANE annotation-derived splice labels. These labels capture annotated splice sites but do not fully represent tissue-specific alternative splicing, cryptic splice usage, or RNA-seq-derived junction variability. Calibration against tissue-specific RNA-seq-derived labels could lead to different probability interpretations.

Finally, the genome-prior correction used sample weighting rather than evaluating every non-splice position directly. This is a practical approximation. It is appropriate for this compute-constrained setting, but larger-scale experiments could directly evaluate full-genome calibration.

## Future work

Future work should evaluate:

- longer flanking-context models
- models trained for more epochs
- held-out chromosomes and genes under multiple annotation sources
- tissue-specific splice-site labels from RNA-seq junction data
- variant-effect calibration for donor/acceptor gain and loss predictions
- calibration under different downstream priors, such as candidate splice-site priors instead of all genomic positions

A particularly important direction is tissue-specific uncertainty-aware splicing prediction. In that setting, uncertainty would not only reflect whether a genomic position is a splice site, but also whether splice usage varies across tissues, individuals, or contexts.

## Conclusion

This work shows that in extremely imbalanced splice-site prediction, calibrated probability estimation and splice-site detection should be evaluated separately. The flank-80 focal-loss OpenSpliceAI-style model already ranks acceptor and donor sites well, but its probability scale depends on the calibration prior. Genome-prior weighted true-logit vector temperature scaling provides strong probability calibration under the natural genome-wide prior without sacrificing detection performance.