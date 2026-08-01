---
title: "Prior-aware calibration of OpenSpliceAI-style splice-site prediction under genome-level class imbalance"
author:
  - "[Author name]"
affiliation: "[Affiliation]"
lang: en
---

::: {.paper-abstract}
## Abstract

Deep learning splice-site predictors are usually evaluated as detection systems, although their output scores are often interpreted as probabilities. These objectives differ in per-nucleotide prediction, where acceptor and donor sites are extremely rare relative to non-splice positions. We evaluate prior-aware probability calibration for focal-loss OpenSpliceAI-style models trained on human GRCh38/MANE annotations with 80- and 400-nt sequence context. Using true pre-softmax logits, we compare uncalibrated predictions, fixed global temperature scaling, unweighted vector temperature scaling, genome-prior weighted vector temperature scaling, and a locally implemented OpenSpliceAI-style class-wise vector-temperature baseline. For the primary flank-400 checkpoint, genome-prior weighted vector scaling reduced weighted multiclass expected calibration error (ECE) from 0.001632 to 0.000029 and weighted negative log-likelihood (NLL) from 0.001838 to 0.000268. The OpenSpliceAI-style baseline achieved the lowest point estimates, with ECE 0.0000117 and NLL 0.0002578, while acceptor and donor AUPRC remained approximately 0.99946 and 0.99959. Exhaustive streaming over 52.88 million chr9 gene/transcript positions reproduced the same calibration pattern without negative sampling. Across both context settings, calibration depended strongly on the declared target prior while splice-site ranking changed minimally. Detection and calibrated probability estimation should therefore be evaluated separately, and splice-site probabilities should be interpreted only relative to an explicit target population.
:::

::: {.paper-body}
## Introduction

RNA splicing is a central mechanism by which eukaryotic cells generate transcript diversity from genomic sequence. Alternative splicing is regulated by combinations of cis-regulatory motifs, transcript structure, splice-site strength, and cellular context. Early computational work on the “splicing code” demonstrated that combinations of RNA features can predict tissue-dependent splicing patterns, establishing splicing prediction as a problem of learning regulatory sequence logic rather than detecting canonical splice motifs alone [@barash2010splicingcode].

Deep learning subsequently shifted splice prediction from hand-engineered regulatory features toward direct sequence-to-sequence prediction. SpliceAI introduced a residual convolutional architecture that predicts whether each position in a pre-mRNA sequence is an acceptor, donor, or neither [@jaganathan2019spliceai]. OpenSpliceAI later provided a modular PyTorch implementation supporting retraining, transfer learning, prediction, variant analysis, and post-hoc calibration [@chao2025openspliceai].

Most splice-site evaluations emphasize detection: whether true acceptor and donor sites receive higher scores than non-splice positions. Detection metrics are appropriate for annotation, variant prioritization, and candidate discovery. However, model scores are also commonly interpreted as confidence estimates. Such an interpretation requires calibration: among predictions assigned a particular probability, the empirical event frequency should be similar.

Calibration is especially sensitive to the evaluation population in per-nucleotide splice-site prediction. Nearly all evaluated positions are non-splice, whereas acceptor and donor sites are rare. Efficient evaluation caches often retain every splice-site positive but subsample non-splice positions, creating a splice-enriched distribution. Calibration fitted or evaluated on this sampled distribution estimates probabilities for that distribution, not automatically for the full target-position population.

This distinction separates two objectives. Detection asks whether true splice sites are ranked above non-splice positions. Calibration asks whether predicted probabilities match observed frequencies under a specified target population and class prior. A model may perform extremely well at detection while remaining unsuitable as a probability estimator under another prior.

We study this distinction using focal-loss OpenSpliceAI-style models with 80- and 400-nt context. Flank-400 is the primary experiment; flank-80 provides a controlled weaker-context comparison. We extract true pre-softmax logits and compare uncalibrated scores, fixed global temperature scaling, unweighted vector scaling, genome-prior weighted vector scaling, and a locally implemented OpenSpliceAI-style class-wise vector-temperature baseline. We evaluate probability quality using ECE, NLL, reliability diagrams, bootstrap resampling, prior-sensitivity analysis, and exhaustive chr9 gene/transcript-position streaming. Detection is evaluated separately using acceptor/donor AUPRC and threshold precision-recall.

Our contributions are:

1. We provide a prior-aware evaluation protocol for OpenSpliceAI-style per-position probabilities using true pre-softmax logits and an explicitly declared target population.
2. We show that calibration conclusions change when a splice-enriched cache is evaluated under its sampled prior versus a reconstructed target-position prior.
3. We compare five post-hoc calibration settings, including a locally implemented OpenSpliceAI-style class-wise vector-temperature baseline, without presenting temperature scaling itself as novel.
4. We demonstrate the calibration-detection separation across flank-80 and flank-400 checkpoints: probability quality changes substantially while acceptor and donor ranking remains nearly unchanged.
5. We confirm the main flank-400 result by streaming all 52.88 million chr9 gene/transcript positions represented in the test H5, eliminating negative subsampling for that chromosome-style robustness evaluation.

# Related Work

## Splicing codes and feature-based prediction

Early computational models of splicing framed alternative splicing as a regulatory code. These approaches combined cis-regulatory motifs, conservation, exon and intron structure, and tissue context to predict exon inclusion or tissue-dependent splicing changes. They established that splicing regulation depends on combinations of sequence and transcript features, but their scope was constrained by manually designed representations.

## Deep learning for splice-site prediction

SpliceAI formulated splice-site prediction as per-position three-class classification and showed that long primary-sequence context can support accurate splice-site and variant-effect prediction [@jaganathan2019spliceai]. This sequence-to-sequence target is also used here. Our objective is not to propose a new detector or claim state-of-the-art detection; it is to evaluate whether the detector's output scale can be interpreted probabilistically under a declared target distribution.

## OpenSpliceAI and calibration

OpenSpliceAI provides a trainable PyTorch implementation of the SpliceAI framework and includes class-wise temperature calibration [@chao2025openspliceai]. Its calibration analysis motivates a further question: calibration is defined relative to a dataset and is not guaranteed to transfer when the class prior, label definition, or downstream population changes. We therefore compare an OpenSpliceAI-style class-wise vector-temperature baseline with explicitly prior-weighted calibration and evaluate the resulting probabilities under stated target priors.

## Neural-network probability calibration under imbalance

High predictive accuracy does not guarantee calibrated probabilities. Temperature scaling rescales frozen model logits after training, while vector temperature scaling permits class-specific rescaling. Under extreme imbalance, both fitting and evaluation depend on the represented class prior. Retaining positives while subsampling negatives changes empirical frequencies, so unweighted calibration on a splice-enriched cache answers a different probability question from calibration under the full target-position prior.

## Gap addressed by this work

Prior splice-site prediction work primarily emphasizes whether models identify or rank splice sites accurately. We focus instead on how the target prior changes probability interpretation. Detection and calibration are reported separately, and no claim is made that the proposed evaluation framework improves the underlying detector.

# Methods

## Models and prediction task

We evaluated OpenSpliceAI-style per-position classifiers trained on human GRCh38 sequence and MANE canonical splice-site annotations. At each evaluated position, the model predicts one of three classes: non-splice, acceptor, or donor. Focal loss was used to address the extreme imbalance between non-splice and splice-site labels.

The primary model used 400-nt sequence context and the selected epoch-8 checkpoint:

`results/best_models/flank400_focal_epoch8_best.pt`

with stable alias:

`results/best_models/flank400_focal_best.pt`

The controlled context comparison used the flank-80 epoch-2 checkpoint:

`results/best_models/flank80_focal_epoch2_best.pt`

Because these checkpoints were selected at different epochs, comparisons are stated as differences between the selected flank-80 and flank-400 configurations, not as a causal estimate of context length alone.

## Evaluation data and sampled caches

The flank-400 model used:

- `data/processed_h5_flank400/dataset_validation.h5`
- `data/processed_h5_flank400/dataset_test.h5`

The flank-80 model used the corresponding files under `data/processed_h5/`. Both context settings represent the same underlying GRCh38/MANE gene and transcript records with different input context.

Across the complete test H5, 416,160,000 positions were evaluated: 106,088 splice-site positives and 416,053,912 non-splice positions. Repeated calibration analysis used caches containing every positive and a reservoir sample of 500,000 non-splice positions. Each test cache therefore contained 606,088 positions: 500,000 non-splice positions, 53,024 acceptors, and 53,064 donors.

The flank-400 caches were:

- `results/logit_cache_flank400_epoch8/validation_sampled_logits.npz`
- `results/logit_cache_flank400_epoch8/test_sampled_logits.npz`

The flank-80 caches were stored under `results/logit_cache_flank80_epoch2/`.

## True-logit extraction

Final analyses used true pre-softmax logits rather than probability-derived logit proxies. Setting:

`model.apply_softmax = False`

returned the final three-class logits. Softmax reconstruction agreed with the model's default probability output to a maximum absolute difference of approximately `1.19e-07` in the verified flank-80 extraction workflow. The same true-logit pathway was used for flank-400.

## Calibration methods

For logit vector:

`z = [z_nonsplice, z_acceptor, z_donor]`

and temperature vector:

`T = [T_nonsplice, T_acceptor, T_donor]`,

calibrated probabilities were computed as `softmax(z / T)`, with elementwise division. Model weights remained frozen.

We compared:

1. **Uncalibrated:** direct softmax probabilities.
2. **Fixed global T=1.1:** the same scalar applied to all three logits.
3. **Unweighted true-logit vector T:** three temperatures fitted on the splice-enriched sampled validation distribution.
4. **Genome-prior weighted true-logit vector T:** three temperatures fitted with non-splice importance weights that reconstruct the target-position prior.
5. **OpenSpliceAI-style vector T:** a locally implemented class-wise vector-temperature baseline fitted using the full validation distribution following the OpenSpliceAI-style objective.

For flank-400, the fitted vectors used in the final comparison were:

- unweighted vector: `[0.5890325, 0.31363136, 0.3490945]`;
- genome-weighted vector: `[0.47286093, 0.46725452, 0.49169588]`;
- OpenSpliceAI-style vector: `[0.3837826, 0.36834455, 0.38703716]`.

The OpenSpliceAI-style optimization converged near epoch 400 and was checked through epoch 1800; the recorded final vector came from the converged full-validation run.

## Target-prior weighting

Because the caches retained all splice sites but sampled non-splice positions, sampled non-splice examples received reconstruction weight:

`total non-splice positions / sampled non-splice positions`.

For the test set:

`416,053,912 / 500,000 = 832.107824`.

The corresponding validation non-splice weight was `181.749102`. These weights reconstruct the distribution over all positions represented in the evaluation H5 records. They should not be interpreted as the prior over every intergenic base in complete chromosomes.

## Calibration and detection metrics

Calibration was evaluated with multiclass ECE, multiclass NLL, class-wise acceptor and donor ECE, and reliability diagrams. Multiclass ECE bins predictions by maximum class probability and compares mean confidence with empirical accuracy. Class-wise reliability bins acceptor or donor probabilities directly and compares them with observed class frequencies. Lower ECE and NLL indicate better probability quality under the declared weighting scheme.

Detection was evaluated separately using acceptor and donor AUPRC, top-k summaries, and threshold-specific precision and recall. AUPRC was treated as the primary ranking metric because splice-site prediction is extremely imbalanced. Calibration may change probability values and threshold behavior without materially changing ranking.

## Bootstrap and prior-sensitivity analyses

We generated 200 nonparametric bootstrap replicates from each sampled test cache using random seed 17. Calibration and detection metrics were recomputed for each replicate, and 95% percentile intervals were obtained. For flank-400, the genome-weighted and OpenSpliceAI-style methods had overlapping bootstrap intervals, so their small point-estimate difference is not interpreted as evidence that one significantly outperforms the other.

Prior sensitivity was evaluated by varying the validation non-splice weight from the splice-enriched setting toward the target-position weight and refitting the temperature vector. We recorded learned temperatures, target-prior ECE and NLL, and argmax class counts. This analysis was completed for both context settings.

## Chr9 streaming robustness evaluation

To test whether weighted sampled-cache results were artifacts of negative sampling, we streamed all chr9 gene/transcript positions represented in `dataset_test.h5`. The flank-400 evaluation contained 748 genes, 10,769 sequence segments, and 52,880,000 post-clipping positions: 52,865,507 non-splice, 7,243 acceptor, and 7,250 donor positions. Neural-network inference was performed once, after which all five calibration settings were evaluated on the same logits.

This analysis is exhaustive for chr9 positions contained in the test H5 gene/transcript records. It is not full intergenic whole-chromosome inference.

## Implementation

All calibration fitting, cache extraction, bootstrap analysis, reliability plotting, context comparison, and chr9 streaming evaluation were implemented as versioned Python scripts in the project repository. The reproducibility package records the exact checkpoint, temperature vectors, random seeds, dataset paths, result-generation commands, and table/figure provenance for each reported experiment.

# Results

## Primary flank-400 result

The selected flank-400 configuration achieved very strong splice-site ranking but its uncalibrated probability scale remained prior dependent. Under test-cache weighting that reconstructs the target-position distribution, the uncalibrated checkpoint had multiclass ECE 0.00163185 and NLL 0.00183797. Genome-prior weighted vector scaling reduced these values to 0.00002899 and 0.00026765. The OpenSpliceAI-style baseline achieved the lowest point estimates, ECE 0.00001168 and NLL 0.00025782.

**Table 1. Primary flank-400 calibration and detection results under target-position weighting.**

| Method | Weighted ECE | Weighted NLL | Acceptor AUPRC | Donor AUPRC |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.00163185 | 0.00183797 | 0.999464 | 0.999590 |
| Genome-weighted vector T | 0.00002899 | 0.00026765 | 0.999463 | 0.999588 |
| OpenSpliceAI-style vector T | **0.00001168** | **0.00025782** | 0.999464 | 0.999589 |

Calibration substantially improved probability quality while AUPRC remained effectively unchanged. The OpenSpliceAI-style baseline had the best point estimates, but its bootstrap intervals overlapped those of the genome-weighted method. The supported conclusion is therefore that both prior-compatible vector-temperature procedures calibrated the flank-400 checkpoint well, not that one significantly outperformed the other.

## Comparison with flank-80

The same qualitative pattern appeared in the weaker-context flank-80 checkpoint. Longer-context and additional-training differences prevent attributing the performance gap solely to context length, but the selected flank-400 configuration had substantially stronger detection and lower uncalibrated NLL/ECE.

**Table 2. Flank-80 and flank-400 point-estimate comparison.**

| Configuration | Method | Weighted ECE | Weighted NLL | Acceptor AUPRC | Donor AUPRC |
|---|---|---:|---:|---:|---:|
| Flank-80, epoch 2 | Uncalibrated | 0.00560995 | 0.00623967 | 0.990909 | 0.994971 |
| Flank-80, epoch 2 | Genome-weighted vector T | 0.00004393 | 0.00081816 | 0.990906 | 0.994977 |
| Flank-80, epoch 2 | OpenSpliceAI-style vector T | **0.00001125** | **0.00080430** | 0.990884 | 0.994971 |
| Flank-400, epoch 8 | Uncalibrated | 0.00163185 | 0.00183797 | 0.999464 | 0.999590 |
| Flank-400, epoch 8 | Genome-weighted vector T | 0.00002899 | 0.00026765 | 0.999463 | 0.999588 |
| Flank-400, epoch 8 | OpenSpliceAI-style vector T | **0.00001168** | **0.00025782** | 0.999464 | 0.999589 |

Thus, stronger detection in the flank-400 configuration did not eliminate prior dependence. In both settings, calibration improved NLL and ECE far more than it changed AUPRC.

## Calibration under the wrong prior can be misleading

The flank-400 unweighted vector `[0.5890325, 0.31363136, 0.3490945]` fitted the splice-enriched validation cache. When evaluated under target-position weighting, it produced low multiclass ECE (`0.00009858`) but NLL `0.00228727`, worse than the uncalibrated NLL `0.00183797`. This disagreement illustrates why ECE alone is insufficient and why the target prior must be declared.

The fixed global `T=1.1` also worsened probability quality in both principal checkpoints. Because a positive scalar temperature preserves multiclass argmax ordering, this deterioration could not be detected from argmax counts alone.

## Exhaustive chr9 gene/transcript-position evaluation

The flank-400 chr9 streaming analysis reproduced the sampled-cache ordering without negative subsampling. Genome-weighted vector scaling reduced multiclass ECE from 0.00173240 to 0.00003704 and NLL from 0.00194420 to 0.00027973. The OpenSpliceAI-style vector achieved ECE 0.00000321 and NLL 0.00026712.

**Table 3. Flank-400 exhaustive chr9 gene/transcript-position calibration.**

| Method | Multiclass ECE | Multiclass NLL | Acceptor ECE | Donor ECE |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.00173240 | 0.00194420 | 0.00080175 | 0.00093204 |
| Fixed global T=1.1 | 0.00283828 | 0.00306006 | 0.00130887 | 0.00153062 |
| Unweighted vector T | 0.00008134 | 0.00244463 | 0.00066050 | 0.00060486 |
| Genome-weighted vector T | 0.00003704 | 0.00027973 | 0.00001859 | 0.00002030 |
| OpenSpliceAI-style vector T | **0.00000321** | **0.00026712** | **0.00000352** | **0.00000460** |

The close agreement between Table 1 and Table 3 shows that the central calibration result is not an artifact of reconstructing non-splice prevalence from a 500,000-negative cache. The experiment does not establish calibration over intergenic positions outside the H5 gene/transcript records.

## Argmax and threshold behavior

Unweighted vector scaling produced excessive chr9 splice argmax predictions: 31,270 acceptors and 29,132 donors, compared with 7,243 and 7,250 true sites. Genome-weighted and OpenSpliceAI-style scaling produced much more restrained counts.

**Table 4. Flank-400 chr9 predicted splice argmax counts.**

| Method | Predicted acceptors | Predicted donors |
|---|---:|---:|
| Uncalibrated | 6,035 | 6,850 |
| Fixed global T=1.1 | 6,035 | 6,850 |
| Unweighted vector T | 31,270 | 29,132 |
| Genome-weighted vector T | 6,193 | 6,275 |
| OpenSpliceAI-style vector T | 6,553 | 6,728 |
| True chr9 count | 7,243 | 7,250 |

Threshold behavior also changed substantially. At probability threshold 0.10, the uncalibrated acceptor predictions had precision 0.201 and recall 0.985; after genome-weighted calibration, precision was 0.567 and recall 0.929; after OpenSpliceAI-style calibration, precision was 0.609 and recall 0.918. Donor precision/recall changed from 0.192/0.990 uncalibrated to 0.580/0.935 genome weighted and 0.615/0.927 OpenSpliceAI style.

These values do not establish a universally preferred threshold. They show that thresholds are properties of the calibrated probability scale and target population and must be selected on validation data for the intended operating point.

## Bootstrap, reliability, and prior sensitivity

Bootstrap resampling supported the stability of the main calibration and detection results. For flank-400, the genome-weighted and OpenSpliceAI-style confidence intervals overlapped, despite the latter having slightly lower point-estimate ECE and NLL. Reliability diagrams showed that both methods aligned predicted confidence more closely with empirical frequencies than the uncalibrated model.

Prior-sensitivity analysis provided a mechanistic explanation. As the validation non-splice weight approached the target-position weight, target-prior NLL and ECE improved and excessive splice argmax counts decreased. The same qualitative relationship was observed in both context settings.

The retained flank-80 detection and prior-sensitivity figures provide a controlled visualization of this separation:

![Acceptor and donor AUPRC across flank-80 calibration methods.](figures/logit_based/logit_detection_auprc.png)

![Flank-80 prior sensitivity of learned temperatures.](figures/logit_based/logit_prior_sensitivity_temperatures.png)

![Flank-80 prior sensitivity of target-prior calibration metrics.](figures/logit_based/logit_prior_sensitivity_genome_metrics.png)

![Flank-80 prior sensitivity of argmax prediction counts.](figures/logit_based/logit_prior_sensitivity_argmax_counts.png)

## Summary of results

Across flank-80 and flank-400, vector-temperature calibration substantially improved probability quality under an explicit target-position prior while leaving splice-site ranking nearly unchanged. The OpenSpliceAI-style baseline achieved the lowest point-estimate ECE and NLL in both configurations, with overlapping bootstrap intervals relative to genome-prior weighted vector scaling. Unweighted fitting to a splice-enriched cache could produce poor target-prior NLL and excessive splice argmax predictions despite apparently small ECE. The exhaustive flank-400 chr9 evaluation confirmed that the main result was not created by negative subsampling.

# Discussion

## Detection and calibration answer different questions

The primary finding is not a new splice-site detector or a new temperature-scaling algorithm. It is that detection and calibrated probability estimation answer different questions under extreme genomic imbalance. The selected flank-400 checkpoint achieved acceptor and donor AUPRC near 0.9995, but its uncalibrated probabilities remained less suitable under the target-position prior than either prior-compatible vector calibration procedure.

Ranking can remain nearly unchanged while probability interpretation changes substantially. A low calibrated splice probability does not necessarily imply that a site is irrelevant for discovery; it may reflect the rarity of splice sites in the target position population. Candidate discovery should therefore use AUPRC, top-k summaries, or validated operating thresholds rather than assuming that multiclass argmax or an arbitrary probability cutoff is appropriate.

## Calibration is conditional on a target population

Calibration is not an intrinsic property of a model independent of data. A predictor may be calibrated for one annotation, prior, or evaluation population and miscalibrated for another. In this study, “target prior” refers to all positions represented in the GRCh38/MANE evaluation H5 gene/transcript records. It does not refer to every intergenic base in complete chromosomes or to noncanonical, cryptic, variant-altered, or tissue-specific splice sites.

This distinction explains why unweighted vector scaling can optimize the splice-enriched sampled task while performing poorly under target-position NLL. Importance weighting changes the calibration objective to match the intended evaluation population. The OpenSpliceAI-style full-validation baseline targets the same natural validation distribution without relying on the sampled calibration cache, explaining its strong performance.

## Interpretation of the OpenSpliceAI-style comparison

The local OpenSpliceAI-style vector-temperature baseline had the lowest point-estimate ECE and NLL for both context settings. This should not be described as our method beating OpenSpliceAI calibration, nor should the small difference between the two vector procedures be presented as statistically decisive. Their flank-400 bootstrap intervals overlapped. The supported contribution is the explicit prior-aware comparison and evaluation framework.

## Practical implications

Probability thresholds should be chosen only after calibration and under the intended target population. Fixed threshold transfer across uncalibrated, globally scaled, and vector-scaled outputs can materially change precision and recall. When the downstream task is candidate discovery, ranking metrics may remain the most useful summary. When probabilities are used for risk interpretation or decision support, the calibration population and label definition must be documented.

## Limitations

This study has several limitations. First, it evaluates one selected focal-loss checkpoint for each context setting. Bootstrap resampling quantifies uncertainty over cached positions but does not replace independent training seeds. Additional independently trained models are needed to establish training-run robustness.

Second, neither configuration is a full SpliceAI-10k ensemble or a claim of state-of-the-art detection. The flank-80 and flank-400 checkpoints also differ in selected training epoch, so their comparison does not isolate the causal effect of context length.

Third, the reconstructed target prior covers positions represented in the MANE gene/transcript evaluation H5. The chr9 streaming experiment removes negative subsampling for those records, but it is not full intergenic whole-chromosome inference.

Fourth, the positive label definition is based on MANE canonical annotations. Calibration is not demonstrated for all annotated isoforms, noncanonical or cryptic splice sites, variant-altered sequences, or tissue-specific splice usage. These are meaningful distribution shifts rather than interchangeable test sets.

Fifth, the OpenSpliceAI-style baseline was implemented locally to reproduce its class-wise vector-temperature behavior. We therefore compare calibration protocols and objectives rather than claiming a direct benchmark against every published OpenSpliceAI model.

Finally, post-hoc calibration estimates probability reliability under observed labels; it does not separately quantify epistemic and aleatoric uncertainty. Extending the project to tissue-specific or competitive splice usage would require RNA-seq/PSI labels and should be treated as a separate study.

## Conclusion

Across focal-loss OpenSpliceAI-style models with 80- and 400-nt context, strong splice-site detection did not guarantee calibrated target-prior probabilities. Genome-prior weighted vector scaling and an OpenSpliceAI-style full-validation vector-temperature baseline sharply improved ECE and NLL while preserving acceptor and donor ranking. Exhaustive streaming over chr9 gene/transcript positions reproduced the flank-400 calibration result without negative subsampling. Splice-site probability estimates should therefore be evaluated and interpreted relative to an explicit target population, separately from detection performance.

# References

::: {#refs}
:::

:::
