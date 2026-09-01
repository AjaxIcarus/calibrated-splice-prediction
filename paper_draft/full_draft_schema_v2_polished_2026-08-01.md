---
title: "Prior-aware calibration of OpenSpliceAI-style splice-site prediction under genome-level class imbalance"
author:
  - "Anonymous submission"
affiliation: "Affiliation withheld for review"
lang: en
bibliography: paper_draft/references.bib
link-citations: true
---

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


::: {.paper-abstract}
## Abstract

Deep learning splice-site predictors are usually evaluated as detection systems, although their output scores are often interpreted as probabilities. These objectives differ in per-nucleotide prediction, where acceptor and donor sites are extremely rare relative to non-splice positions. We evaluate prior-aware probability calibration for focal-loss OpenSpliceAI-style models trained on human GRCh38/MANE annotations. Using true pre-softmax logits, we compare uncalibrated predictions, fixed global temperature scaling, unweighted vector scaling, target-weighted vector scaling, and a locally implemented OpenSpliceAI-style full-validation vector-temperature baseline. In matched valid-only evaluations of a validation-selected seed-11/epoch-11 primary model and seed-23/epoch-14 replication, target-weighted vector scaling reduced weighted multiclass ECE by 98.87%/98.64%, NLL by 84.42%/85.44%, and Brier score by 36.31%/38.27%. The OpenSpliceAI-style baseline produced closely comparable reductions, and none of ten seed-by-metric paired 95% bootstrap intervals excluded zero. Target-weighted acceptor and donor AUPRC changed minimally. Detection and calibrated probability estimation should therefore be evaluated separately, and splice-site probabilities should be interpreted only relative to an explicit target population.
:::

::: {.paper-body}
## Introduction

RNA splicing is a central mechanism by which eukaryotic cells generate transcript diversity from genomic sequence. Alternative splicing is regulated by combinations of cis-regulatory motifs, transcript structure, splice-site strength, and cellular context. Early computational work on the “splicing code” demonstrated that combinations of RNA features can predict tissue-dependent splicing patterns, establishing splicing prediction as a problem of learning regulatory sequence logic rather than detecting canonical splice motifs alone [@barash2010splicingcode]. Competitive models later made the dependence of splice-site usage on neighboring sites explicit [@bretschneider2018cossmo].

Deep learning subsequently shifted splice prediction from hand-engineered regulatory features toward direct sequence-to-sequence prediction. SpliceAI introduced a residual convolutional architecture that predicts whether each position in a pre-mRNA sequence is an acceptor, donor, or neither [@jaganathan2019spliceai]. Pangolin extended sequence-based prediction toward tissue-aware splice-site strength [@zeng2022pangolin]. OpenSpliceAI later provided a modular PyTorch implementation supporting retraining, transfer learning, prediction, variant analysis, and post-hoc calibration [@chao2025openspliceai].

Most splice-site evaluations emphasize detection: whether true acceptor and donor sites receive higher scores than non-splice positions. Detection metrics are appropriate for annotation, variant prioritization, and candidate discovery. However, model scores are also commonly interpreted as confidence estimates. Such an interpretation requires calibration: among predictions assigned a particular probability, the empirical event frequency should be similar [@jaganathan2019spliceai; @zeng2022pangolin; @chao2025openspliceai].

Calibration is especially sensitive to the evaluation population in per-nucleotide splice-site prediction. Nearly all evaluated positions are non-splice, whereas acceptor and donor sites are rare. Efficient evaluation caches often retain every splice-site positive but subsample non-splice positions, creating a splice-enriched distribution. Calibration fitted or evaluated on this sampled distribution estimates probabilities for that distribution, not automatically for the full target-position population.

This distinction separates two objectives. Detection asks whether true splice sites are ranked above non-splice positions. Calibration asks whether predicted probabilities match observed frequencies under a specified target population and class prior. A model may perform extremely well at detection while remaining unsuitable as a probability estimator under another prior.

We study this distinction using two independently trained focal-loss OpenSpliceAI-style models with 400-nt context. We extract true pre-softmax logits and compare uncalibrated scores, fixed global temperature scaling, unweighted vector scaling, target-weighted vector scaling, and a locally implemented OpenSpliceAI-style full-validation class-wise vector-temperature baseline. The analysis uses schema-v2 caches that include every sequence, exclude all-zero padding, and reconstruct the valid-position target population. Probability quality is evaluated using ECE, NLL, Brier score, reliability diagrams, and paired bootstrap resampling; target-weighted AUPRC and operating-point analyses are reported separately. An earlier flank-80 configuration is retained only as developmental validation of the training and logit-extraction workflow, not as a quantitative context-length comparison.

Our contributions are:

1. We provide a prior-aware evaluation protocol for OpenSpliceAI-style per-position probabilities using true pre-softmax logits and an explicitly declared target population.
2. We show that calibration conclusions change when a splice-enriched cache is evaluated under its sampled prior versus a reconstructed target-position prior.
3. We compare five post-hoc calibration settings, including a locally implemented OpenSpliceAI-style class-wise vector-temperature baseline, without presenting temperature scaling itself as novel.
4. We demonstrate the calibration-detection separation in matched valid-position evaluations: probability quality changes substantially while target-weighted acceptor and donor AUPRC values change minimally.
5. We evaluate a validation-selected seed-11/epoch-11 primary model and an independently trained, validation-selected seed-23/epoch-14 replication using identical populations, metrics, calibration methods, and paired bootstrap design.

# Related Work

## Splicing codes and feature-based prediction

Early computational models of splicing framed alternative splicing as a regulatory code. These approaches combined cis-regulatory motifs, conservation, exon and intron structure, and tissue context to predict exon inclusion or tissue-dependent splicing changes. They established that splicing regulation depends on combinations of sequence and transcript features, but their scope was constrained by manually designed representations [@barash2010splicingcode].

## Deep learning for splice-site prediction

SpliceAI formulated splice-site prediction as per-position three-class classification and showed that long primary-sequence context can support accurate splice-site and variant-effect prediction [@jaganathan2019spliceai]. Related deep models address competitive splice-site selection [@bretschneider2018cossmo] and tissue-aware splice strength [@zeng2022pangolin]. This sequence-to-sequence target is also used here. Our objective is not to propose a new detector or claim state-of-the-art detection; it is to evaluate whether the detector's output scale can be interpreted probabilistically under a declared target distribution.

## OpenSpliceAI and calibration

OpenSpliceAI provides a trainable PyTorch implementation of the SpliceAI framework and includes class-wise temperature calibration [@chao2025openspliceai]. Its calibration analysis motivates a further question: calibration is defined relative to a dataset and is not guaranteed to transfer when the class prior, label definition, or downstream population changes. We therefore compare an OpenSpliceAI-style class-wise vector-temperature baseline with explicitly prior-weighted calibration and evaluate the resulting probabilities under stated target priors.

## Neural-network probability calibration under imbalance

High predictive accuracy does not guarantee calibrated probabilities [@guo2017calibration]. Temperature scaling rescales frozen model logits after training, while vector temperature scaling permits class-specific rescaling; more flexible multiclass maps, such as Dirichlet calibration, further illustrate that calibration is a separate post-hoc modeling problem [@kull2019dirichlet]. Under extreme imbalance, both fitting and evaluation depend on the represented class prior. Retaining positives while subsampling negatives changes empirical frequencies, so unweighted calibration on a splice-enriched cache answers a different probability question from calibration under the full target-position prior.

## Gap addressed by this work

Prior splice-site prediction work primarily emphasizes whether models identify or rank splice sites accurately. We focus instead on how the target prior changes probability interpretation. Detection and calibration are reported separately, and no claim is made that the proposed evaluation framework improves the underlying detector.

# Methods

## Models and prediction task

We evaluated OpenSpliceAI-style per-position classifiers trained on human GRCh38 sequence [@schneider2017grch38] and MANE canonical splice-site annotations [@morales2022mane]. At each evaluated position, the model predicts one of three classes: non-splice, acceptor, or donor. Focal loss was used to address the extreme imbalance between non-splice and splice-site labels [@lin2017focal].

The primary model was the validation-selected seed-11/epoch-11 flank-400 checkpoint. The independent replication was the validation-selected seed-23/epoch-14 flank-400 checkpoint. The models were trained independently, and checkpoint selection was performed within seed using validation performance.

## Evaluation data and valid-only sampled caches

Across all 84,258 test sequences, the evaluation dataset contained 421,290,000 raw fixed-length positions. We excluded 14,011,378 all-zero padding positions, leaving a target population of 407,278,622 valid positions: 407,170,634 non-splice positions, 53,994 acceptors, and 53,994 donors.

Each schema-v2 test cache retained all 107,988 valid splice-site positions and a reservoir sample of 500,000 valid non-splice positions, for 607,988 cached positions. The validation population similarly contained 89,016,242 valid positions after excluding 3,413,758 padding positions: 88,990,332 non-splice positions, 12,955 acceptors, and 12,955 donors.

Both seeds used the same complete sequence inventory and valid-position census. Earlier caches that truncated shard tails or treated all-zero padding as non-splice were excluded from the final comparison.

## True-logit extraction

Final analyses used true pre-softmax logits rather than probability-derived logit proxies. Disabling the model's final softmax layer by setting:

`model.apply_softmax = False`

returned the final three-class logits. Applying softmax to the extracted logits reproduced the model's default probabilities to a maximum absolute difference of approximately `1.19e-07`. The same verified extraction pathway was used for both flank-400 models.

## Developmental flank-80 workflow

An earlier seed-11/epoch-2 flank-80 configuration was used to establish the low-memory training and true-logit extraction workflow. Its validation and test outputs were subsequently re-extracted under schema v2, confirming complete sequence coverage, exclusion of all-zero padding, and the same valid-position label census used for flank-400. The earlier flank-80 calibration estimates predated schema v2, however, and the full five-method calibration and paired-bootstrap analysis was not repeated for that checkpoint. We therefore use flank-80 only as implementation validation and do not draw quantitative conclusions about context length from it.

## Calibration methods

For logit vector `z = [z_nonsplice, z_acceptor, z_donor]` and temperature vector `T = [T_nonsplice, T_acceptor, T_donor]`, calibrated probabilities were computed as `softmax(z / T)`, with elementwise division. Model weights remained frozen.

We compared:

1. **Uncalibrated:** direct softmax probabilities.
2. **Fixed global T=1.1:** the same scalar applied to all three logits.
3. **Unweighted true-logit vector T:** three temperatures fitted on the splice-enriched sampled validation distribution.
4. **Target-weighted true-logit vector T:** three temperatures fitted with valid non-splice importance weights that reconstruct the target-position prior.
5. **OpenSpliceAI-style full-validation vector T:** a locally implemented class-wise vector-temperature baseline fitted from all valid validation positions following the OpenSpliceAI-style objective.

Each model's vector temperatures were fitted independently using its own validation outputs. No temperature vector was transferred between training seeds. Seed-specific calibration artifacts and optimization provenance are included in the reproducibility archive.

## Target-prior weighting

Because each schema-v2 cache retained every valid splice site but sampled valid non-splice positions, each sampled test non-splice position received reconstruction weight

`407,170,634 / 500,000 = 814.341268`.

The corresponding validation non-splice weight was

`88,990,332 / 500,000 = 177.980664`.

Splice-site positions received unit weight. These weights reconstruct the distribution over valid positions represented in the evaluation H5 records. They do not represent every intergenic base in complete chromosomes.

## Calibration and detection metrics

Probability quality was evaluated with target-weighted multiclass ECE, NLL, and Brier score, together with class-wise acceptor and donor reliability diagrams. Multiclass ECE used 15 equal-width bins of maximum class probability. NLL and Brier are proper scoring rules and were included to prevent apparently small ECE from being interpreted in isolation [@gneiting2007proper]. Lower ECE, NLL, and Brier indicate better probability quality under the declared target population.

Detection was evaluated separately using acceptor and donor AUPRC, top-k summaries, and threshold-specific precision and recall. The main cross-seed table reports target-weighted AUPRC, computed under the same reconstructed valid-position population. Sampled-cache AUPRC describes the splice-enriched cache and is not interchangeable with target-weighted AUPRC. Precision-recall summaries are more informative than ROC summaries for highly imbalanced tasks [@saito2015precisionrecall].

## Bootstrap and prior-sensitivity analyses

For each trained model, we generated 200 paired position-level bootstrap replicates using random seed 17, stratified resampling by true class, and the same resampled positions across calibration methods. We obtained 95% percentile intervals for all 13 recorded metrics and for paired method contrasts. These intervals quantify evaluated-position uncertainty within each trained model; they do not estimate training-seed, gene, or chromosome variability.

Prior sensitivity was evaluated by varying the validation non-splice weight from the splice-enriched setting toward the valid-position target weight and refitting the temperature vector. Learned temperatures, target-prior ECE and NLL, and argmax class counts were recorded as mechanistic support for the effect of the declared prior.

## Matched cross-seed robustness design

The validation-selected seed-11/epoch-11 checkpoint was designated the primary model and the independently trained seed-23/epoch-14 checkpoint the replication. Both were evaluated using the identical valid-position population, five calibration methods, 13-metric schema, and paired 200-replicate bootstrap design. Agreement across two trained models is treated as descriptive robustness, not a population-level estimate of optimization variability.

## Implementation

All calibration fitting, cache extraction, bootstrap analysis, reliability plotting, and matched cross-seed schema-v2 evaluation were implemented as versioned Python scripts. The reproducibility archive records the checkpoint identifiers, temperature vectors, random seeds, data provenance, result-generation commands, and table/figure lineage for each reported experiment.

# Results

## Matched schema-v2 cross-seed probability quality

Target-weighted true-logit vector scaling reduced weighted multiclass ECE by 98.87% and 98.64%, NLL by 84.42% and 85.44%, and Brier score by 36.31% and 38.27% relative to the corresponding uncalibrated seed-11 and seed-23 models. OpenSpliceAI-style full-validation vector scaling produced closely comparable reductions in ECE (99.13%/98.92%), NLL (84.42%/85.39%), and Brier score (36.49%/37.77%). Target-weighted acceptor and donor AUPRC changed only minimally after either prior-aware calibration method.

In contrast, fixed global T=1.1 worsened all three probability-quality metrics in both models. Unweighted vector scaling reduced ECE but worsened NLL and Brier score, showing why ECE alone is insufficient under extreme class imbalance. None of the ten seed-by-metric paired 95% bootstrap intervals comparing target-weighted and OpenSpliceAI-style vector scaling excluded zero. The supported conclusion is therefore descriptive consistency of both prior-aware procedures, not superiority of either method.

:::

::: {.wide-table}
**Table 1. Matched flank-400 cross-seed robustness on the schema-v2 valid-position target population.**

Values are seed-11 epoch-11 / seed-23 epoch-14.

| Method | Weighted ECE | Weighted NLL | Weighted Brier | Target-weighted acceptor AUPRC | Target-weighted donor AUPRC |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 1.430e-03 / 1.480e-03 | 1.626e-03 / 1.671e-03 | 2.101e-04 / 2.095e-04 | 0.892620 / 0.893685 | 0.915186 / 0.922882 |
| Fixed global T=1.1 | 2.371e-03 / 2.465e-03 | 2.574e-03 / 2.663e-03 | 2.493e-04 / 2.491e-04 | 0.892617 / 0.893686 | 0.915174 / 0.922881 |
| Unweighted true-logit vector T | 1.252e-04 / 1.016e-04 | 2.316e-03 / 2.145e-03 | 1.261e-03 / 1.147e-03 | 0.892310 / 0.882316 | 0.901449 / 0.917532 |
| Target-weighted true-logit vector T | 1.615e-05 / 2.006e-05 | 2.532e-04 / 2.432e-04 | 1.338e-04 / 1.293e-04 | 0.892953 / 0.893145 | 0.914788 / 0.923178 |
| OpenSpliceAI-style full-validation vector T | 1.241e-05 / 1.602e-05 | 2.532e-04 / 2.441e-04 | 1.334e-04 / 1.304e-04 | 0.893008 / 0.893747 | 0.915055 / 0.922969 |

ECE, NLL, and Brier are lower-is-better; AUPRC is higher-is-better. All values use the same reconstructed valid-position target population. Paired prior-aware contrasts and their 95% bootstrap intervals are reported in Supplementary Table S1.
:::

::: {.paper-body}

:::

::: {.figure-grid}
![**Figure 1a.** Schema-v2 reliability diagnostics for the validation-selected seed-11/epoch-11 primary model. Curves connect nonempty 15-bin equal-width estimates. The ECE values in the legend are target-weighted multiclass ECE; they do not summarize the class-specific panel gaps.](figures/flank400_schema_v2_reliability_seed11.png)

![**Figure 1b.** Schema-v2 reliability diagnostics for the validation-selected seed-23/epoch-14 replication model. Curves connect nonempty 15-bin equal-width estimates. The ECE values in the legend are target-weighted multiclass ECE; they do not summarize the class-specific panel gaps.](figures/flank400_schema_v2_reliability_seed23.png)
:::

::: {.paper-body}

The class-wise curves are visually variable in rare high-probability bins because those bins contain little target-population mass. Their contribution to weighted ECE is correspondingly small. The unweighted-vector curves show a substantive failure that is also captured by their worse NLL and Brier score. Raw bin counts, reconstructed target weights, target-weight fractions, and per-bin ECE contributions are retained as machine-readable Supplementary Data rather than expanded into a large printed table.

## Calibration under the wrong prior can be misleading

Unweighted vector scaling lowered ECE from 1.430e-03 to 1.252e-04 in the primary model and from 1.480e-03 to 1.016e-04 in the replication. Nevertheless, NLL increased from 1.626e-03 to 2.316e-03 and from 1.671e-03 to 2.145e-03; Brier score also increased more than fivefold in each model. This disagreement shows why ECE alone is insufficient and why the target prior must be declared.

Fixed global T=1.1 also worsened ECE, NLL, and Brier score in both trained models (primary NLL 2.574e-03; replication NLL 2.663e-03). A positive scalar temperature preserves multiclass argmax ordering, so this deterioration cannot be detected from argmax counts alone.

## Argmax and threshold behavior

Calibration changed the probability scale even when ranking changed little. Consequently, thresholds were not transferred unchanged between uncalibrated, globally scaled, and vector-scaled outputs. Threshold-specific precision and recall were treated as operating-point analyses, and no universal threshold was inferred. The unweighted-vector result further shows why argmax counts and ECE cannot be interpreted alone: a method can appear favorable on one summary while assigning worse likelihood to the observed labels under the target population.

## Bootstrap, reliability, and prior sensitivity

Within-model paired bootstrap resampling and schema-v2 reliability diagrams supported the conclusion that both prior-aware vector-scaling procedures improved probability alignment relative to the uncalibrated models. None of the ten paired intervals contrasting target-weighted and OpenSpliceAI-style vector scaling excluded zero. Position-level intervals are not interpreted as training-seed uncertainty; the second independently trained checkpoint provides only a descriptive model-level replication.

Prior-sensitivity analysis provided a mechanistic explanation: as the validation non-splice weight approached the valid-position target weight, target-prior NLL and ECE improved and excessive splice argmax counts decreased.

## Summary of results

Across both validation-selected flank-400 models, target-weighted and OpenSpliceAI-style full-validation vector scaling sharply improved ECE, NLL, and Brier score under the valid-position target population while leaving target-weighted acceptor and donor AUPRC nearly unchanged. Fixed global scaling worsened all three probability-quality metrics, and unweighted vector fitting reduced ECE while worsening NLL and Brier. The qualitative result replicated across two independently trained models, but paired position-level intervals did not establish superiority of either prior-aware procedure.

# Discussion

## Detection and calibration answer different questions

The primary finding is not a new splice-site detector or a new temperature-scaling algorithm. It is that detection and calibrated probability estimation answer different questions under extreme genomic imbalance. In seed 11, uncalibrated target-weighted acceptor/donor AUPRC was 0.892620/0.915186; after target-weighted vector scaling it was 0.892953/0.914788, and after OpenSpliceAI-style scaling it was 0.893008/0.915055. Seed 23 showed the same stability, with values near 0.894 for acceptors and 0.923 for donors across the prior-aware methods, even as ECE, NLL, and Brier improved sharply.

Ranking can remain nearly unchanged while probability interpretation changes substantially. A low calibrated splice probability does not necessarily imply that a site is irrelevant for discovery; it may reflect the rarity of splice sites in the valid-position target population. Candidate discovery should therefore use target-appropriate AUPRC, top-k summaries, or validated operating thresholds rather than assuming that multiclass argmax or an arbitrary probability cutoff is appropriate.

## Calibration is conditional on a target population

Calibration is not an intrinsic property of a model independent of data. A predictor may be calibrated for one annotation, prior, or evaluation population and miscalibrated for another. In this study, “target prior” refers to all sequence positions across the GRCh38/MANE evaluation H5 gene/transcript records. It does not refer to every intergenic base in complete chromosomes or to noncanonical, cryptic, variant-altered, or tissue-specific splice sites.

This distinction explains why unweighted vector scaling can optimize the splice-enriched sampled task while performing poorly under target-position NLL. Importance weighting changes the calibration objective to match the intended evaluation population. The OpenSpliceAI-style full-validation baseline targets the same natural validation distribution without relying on the sampled calibration cache; this matching of fitting and evaluation priors is consistent with its strong performance.

## Interpretation of the OpenSpliceAI-style comparison

The local OpenSpliceAI-style full-validation vector-temperature baseline and target-weighted true-logit vector scaling produced closely comparable proper scoring rules in both independently trained flank-400 models. None of the ten seed-by-metric paired 95% bootstrap intervals excluded zero. These results should not be described as our method beating OpenSpliceAI calibration, nor as proof that the two procedures are equivalent. The supported contribution is the explicit prior-aware comparison and valid-position evaluation framework.

## Practical implications

Probability thresholds should be selected and validated for the specific score transformation and intended target population. Calibration is required when a threshold is intended to carry a probability interpretation, whereas ranking or operating-point thresholds can be selected directly on validation data. Fixed threshold transfer across uncalibrated, globally scaled, and vector-scaled outputs can materially change precision and recall. When the downstream task is candidate discovery, ranking metrics may remain the most useful summary. When probabilities are used for risk interpretation or decision support, the calibration population and label definition must be documented.

## Limitations

The main flank-400 calibration pattern replicated in a second independently trained model, reducing dependence on a single checkpoint. Nevertheless, two validation-selected training runs do not characterize the population distribution of optimization variability. The paired bootstrap unit is position and therefore does not quantify training-seed, gene, or chromosome uncertainty.

The target population covers valid sequence positions represented in the MANE gene/transcript evaluation H5 after all-zero padding is excluded. It is not complete intergenic whole-genome inference. Calibration is also not demonstrated for all annotated isoforms, noncanonical or cryptic splice sites, variant-altered sequences, or tissue-specific splice usage.

Neither flank-400 configuration is a full SpliceAI-10k ensemble or a claim of state-of-the-art detection. Context length was not evaluated as an isolated experimental factor: the developmental flank-80 checkpoint differed in training history and was not carried through the complete schema-v2 calibration analysis.

The OpenSpliceAI-style baseline was implemented locally to reproduce its class-wise vector-temperature behavior. We compare calibration protocols and objectives rather than every published OpenSpliceAI model. Finally, post-hoc calibration estimates reliability under observed labels; it does not separately quantify epistemic and aleatoric uncertainty.

## Conclusion

Across two independently trained focal-loss OpenSpliceAI-style flank-400 models, strong splice-site ranking did not guarantee calibrated valid-position probabilities. Target-weighted true-logit vector scaling and an OpenSpliceAI-style full-validation vector-temperature baseline sharply improved ECE, NLL, and Brier score while preserving target-weighted acceptor and donor AUPRC. Fixed global scaling worsened probability quality, and unweighted vector scaling reduced ECE while worsening proper scoring rules. Splice-site probabilities should therefore be evaluated and interpreted relative to an explicit target population, separately from detection performance. The two trained models support descriptive robustness but do not establish superiority of either prior-aware calibration procedure.

:::

::: {.references-section}
# References

::: {#refs}
:::
:::
