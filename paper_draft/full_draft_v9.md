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

Deep learning models for splice-site prediction are typically evaluated by detection performance, but their output scores are often interpreted as calibrated probabilities. This distinction is important in per-nucleotide splice-site prediction, where acceptor and donor sites are extremely rare relative to non-splice genomic positions. We evaluate probability calibration for a flank-80 OpenSpliceAI-style model trained on human GRCh38/MANE annotations using focal loss. Using true pre-softmax logits, we compare uncalibrated predictions, global temperature scaling, unweighted vector temperature scaling, and genome-prior weighted vector temperature scaling. Genome-weighted true-logit vector scaling reduced weighted multiclass expected calibration error from 0.005610 to 0.000044 and weighted negative log-likelihood from 0.006240 to 0.000818, while preserving strong splice-site detection performance with acceptor and donor AUPRC near 0.991 and 0.995. Bootstrap confidence intervals, reliability diagrams, and prior-sensitivity analysis confirmed that calibration depends strongly on the assumed class prior. These results show that calibrated probability estimation and splice-site detection are separate objectives under extreme class imbalance, and that splice-site probabilities should be interpreted only relative to an explicit evaluation prior.

## Introduction

RNA splicing is a central mechanism by which eukaryotic cells generate transcript diversity from genomic sequence. Alternative splicing affects most multi-exon human genes and is regulated by combinations of cis-regulatory motifs, transcript structure, splice-site strength, and cellular context. Early computational work on the “splicing code” showed that combinations of RNA features could predict tissue-dependent splicing patterns, establishing splicing prediction as a problem of learning regulatory sequence logic rather than detecting splice motifs alone [@barash2010splicingcode].

More recent deep learning models have shifted this problem from hand-engineered regulatory features toward sequence-to-sequence prediction directly from primary DNA sequence. SpliceAI introduced a deep residual convolutional model that predicts, for each position in a pre-mRNA sequence, whether that position is a splice acceptor, splice donor, or neither [@jaganathan2019spliceai]. This framing made splice-site prediction a per-nucleotide classification problem and enabled accurate prediction of canonical and cryptic splice sites from sequence context. OpenSpliceAI later provided a modular PyTorch implementation of this approach, enabling easier retraining, transfer learning, prediction, variant analysis, and model calibration [@chao2025openspliceai].

Most evaluations of splice-site prediction models emphasize detection: whether true acceptor and donor sites receive higher scores than non-splice positions. This is appropriate for annotation, variant prioritization, and candidate splice-site discovery. However, model outputs are also often interpreted as confidence estimates. A score near 0.9 may be read as high confidence, while a score near 0.01 may be read as low confidence. This interpretation assumes that the model’s probabilities are calibrated: predicted probabilities should match empirical event frequencies.

Calibration is especially challenging in per-nucleotide splice-site prediction because the class distribution is extremely imbalanced. Across the genome, nearly all positions are non-splice positions, while acceptor and donor splice sites are rare. A sampled benchmark that retains all splice sites but subsamples non-splice positions can be useful for efficient evaluation, but it changes the class prior. As a result, probabilities calibrated under a splice-enriched sampled distribution may not represent probabilities under the natural genome-wide distribution.

This creates a distinction between two objectives. Detection asks whether true splice sites are ranked above non-splice positions. Calibration asks whether predicted probabilities match empirical frequencies under a specified class prior. A model can perform well at detection while remaining miscalibrated as a probability estimator under the genome-wide prior.

In this work, we study this calibration-detection distinction using a flank-80 OpenSpliceAI-style model trained with focal loss on human GRCh38/MANE splice annotations. We extract true pre-softmax logits, fit temperature-scaling models, and compare unweighted calibration against genome-prior weighted calibration. We evaluate calibration using expected calibration error, negative log-likelihood, reliability diagrams, and bootstrap confidence intervals, while evaluating detection separately using acceptor/donor AUPRC and threshold precision-recall.

Our main result is that genome-prior weighted true-logit vector temperature scaling substantially improves probability calibration under the natural genome-position prior without sacrificing splice-site ranking performance. This shows that, in highly imbalanced splice-site prediction, calibrated probability estimation and splice-site detection should be evaluated separately, and calibrated probabilities should always be interpreted relative to an explicit class prior.

Our contributions are:

1. We evaluate splice-site prediction calibration using true pre-softmax logits rather than probability-derived logit proxies.
2. We show that calibration depends strongly on the assumed class prior in highly imbalanced per-nucleotide splice-site prediction.
3. We introduce genome-prior weighted vector temperature scaling for calibrated splice-site probability interpretation.
4. We show that genome-prior calibration improves ECE and NLL while preserving acceptor and donor AUPRC.
5. We separate calibration evaluation from splice-site detection evaluation using reliability diagrams, bootstrap confidence intervals, prior-sensitivity analysis, AUPRC, and threshold precision-recall.

# Related Work

## Splicing codes and feature-based prediction

Early computational models of splicing framed alternative splicing as a regulatory code. These approaches used combinations of cis-regulatory motifs, conservation, exon and intron structure, and tissue context to predict exon inclusion or tissue-dependent splicing changes. This line of work established that splicing prediction requires more than recognizing canonical splice dinucleotides: splice regulation depends on combinations of sequence features, transcript architecture, and regulatory context.

These models were important because they made splicing prediction interpretable and biologically grounded. However, they also depended heavily on manually designed feature sets. As a result, their performance and scope were limited by the quality and completeness of the selected regulatory features.

## Deep learning for splice-site prediction

Deep learning shifted splice prediction from hand-engineered feature extraction to direct learning from primary sequence. SpliceAI formulated splice-site prediction as a per-position classification task, where the model predicts whether each position in a pre-mRNA sequence is an acceptor site, donor site, or neither. By using deep residual convolutional networks and long sequence context, SpliceAI showed that primary sequence alone can support accurate splice-site prediction and variant-effect interpretation.

This sequence-to-sequence framing is central to our work. We use the same basic prediction target: a three-class probability distribution over non-splice, acceptor, and donor labels at each genomic position. However, while SpliceAI-style evaluations typically emphasize detection and variant prioritization, our focus is the calibration of the probabilities assigned to those classes.

## OpenSpliceAI and retrainable splice prediction

OpenSpliceAI provides a modular PyTorch implementation of the SpliceAI framework. It supports data preprocessing from genome FASTA and annotation files, model training, transfer learning, prediction, variant analysis, and calibration. This makes it easier to retrain SpliceAI-style models on different annotations, species, or training objectives.

Our model is OpenSpliceAI-style rather than a full-scale SpliceAI-10k ensemble. We use a flank-80 model trained under limited compute, which allows us to study calibration behavior in a controlled setting. This makes the analysis practical, but it also means that our goal is not to claim state-of-the-art splice-site detection. Instead, we use this model to investigate how probability calibration behaves under extreme class imbalance.

## Calibration of neural network probabilities

Neural network outputs are often interpreted as confidence scores, but high predictive accuracy does not guarantee calibrated probabilities. Calibration evaluates whether predicted probabilities match empirical frequencies. For example, among predictions assigned probability 0.8, roughly 80% should be correct if the model is calibrated.

Temperature scaling is a common post-hoc calibration method. It rescales logits after training while keeping model weights fixed. In multiclass settings, vector temperature scaling can assign a separate temperature to each class, allowing class-specific adjustment of predicted confidence.

In splice-site prediction, calibration is complicated by extreme imbalance. Across genomic positions, non-splice labels dominate, while acceptor and donor sites are rare. Evaluation sets often retain positives and subsample negatives, which changes the class prior. A model calibrated on such a splice-enriched sample may not produce probabilities that correspond to the natural genome-wide prior.

## Gap addressed by this work

Prior splice-site prediction work primarily evaluates whether models identify splice sites accurately. Less attention has been given to whether the resulting probabilities are calibrated under a specified prior. This distinction matters because a model can rank splice sites well while still assigning probabilities that are miscalibrated under the genome-wide distribution.

Our work addresses this gap by separating detection from calibration. We evaluate splice-site ranking using AUPRC and threshold precision-recall, while evaluating probability calibration using expected calibration error, negative log-likelihood, reliability diagrams, bootstrap confidence intervals, and prior-sensitivity analysis. This allows us to show that genome-prior weighted true-logit vector scaling improves probability calibration without substantially changing splice-site ranking.

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

# Results

## True-logit calibration setup

We evaluated calibration for a flank-80 OpenSpliceAI-style splice-site predictor trained with focal loss on GRCh38/MANE annotations. The model predicts three per-nucleotide classes: non-splice, acceptor, and donor. For the final calibration analysis, we used true pre-softmax logits rather than probability-derived logit proxies. Softmax reconstruction of the extracted logits matched the model’s original probability outputs with a maximum absolute difference of approximately `1.19e-07`.

The sampled test cache contained 606,088 positions: 500,000 non-splice positions, 53,024 acceptor sites, and 53,064 donor sites. Because this sampled cache retained all splice sites but only a subset of non-splice positions, it was strongly enriched for splice sites relative to the natural genome-wide distribution. We therefore evaluated calibration under the genome-position prior by weighting sampled non-splice positions according to their full-test-set abundance.

## Genome-prior weighted vector scaling improves calibration

[Table 1 here: Main calibration and detection results]

[Figure 1 here: Weighted multiclass ECE and NLL]

[Figure 3 here: Reliability diagrams]

Global scalar temperature scaling did not improve genome-prior calibration. Compared with the uncalibrated model, global `T = 1.1` increased weighted multiclass ECE from 0.005610 to 0.008538 and increased weighted multiclass NLL from 0.006240 to 0.009212.

True-logit vector temperature scaling performed better, but only when fit under the appropriate prior. The final genome-prior weighted vector temperature parameters were:

- `T_nonsplice = 0.435169`
- `T_acceptor = 0.416153`
- `T_donor = 0.427941`

This calibration reduced weighted multiclass ECE from 0.005610 to 0.000044 and reduced weighted multiclass NLL from 0.006240 to 0.000818. Bootstrap confidence intervals confirmed the stability of this result: weighted multiclass ECE was 0.000049 [0.000039, 0.000058], and weighted multiclass NLL was 0.000818 [0.000804, 0.000832].

Reliability diagrams showed the same pattern. The uncalibrated model was overconfident under genome-weighted evaluation, especially in low-probability splice-site bins. Genome-weighted vector scaling aligned predicted probabilities more closely with observed frequencies.

## Detection remains strong after calibration

[Figure 2 here: Acceptor and donor AUPRC across calibration methods]

Calibration changed probability interpretation but did not substantially change splice-site ranking. The uncalibrated model achieved acceptor AUPRC of 0.9909 and donor AUPRC of 0.9950 on the sampled test cache. After genome-weighted true-logit vector scaling, acceptor AUPRC remained 0.9909 and donor AUPRC remained 0.9950.

Bootstrap estimates supported this conclusion. Under genome-weighted vector scaling, acceptor AUPRC was 0.990906 [0.990350, 0.991494], and donor AUPRC was 0.994977 [0.994573, 0.995355].

These results show that calibration and detection measure different properties. Calibration evaluates whether predicted probabilities match empirical frequencies under a chosen prior. Detection evaluates whether true splice sites are ranked above non-splice positions. In this model, genome-prior calibration improved substantially while ranking performance remained high.

## Threshold behavior depends on calibration

[Table 2 here: Threshold precision/recall examples]

[Figure 4 here: Threshold precision-recall curves]

Although AUPRC was nearly unchanged, threshold-based precision and recall changed after calibration. With the uncalibrated model, a threshold of 0.5 produced very high precision but low recall: acceptor precision was 1.0000 with recall 0.0750, and donor precision was 0.9999 with recall 0.1432.

After genome-weighted vector calibration, lower thresholds provided more useful operating points. At threshold 0.01, acceptor precision was 0.9801 with recall 0.8954, and donor precision was 0.9852 with recall 0.9403.

Thus, calibration does not simply preserve all threshold behavior. It changes the probability scale. Thresholds should be selected according to the calibrated operating point rather than inherited from arbitrary defaults such as 0.5.

## Prior sensitivity explains the calibration behavior

[Figure 5 here: Prior sensitivity temperatures, genome-prior metrics, and argmax counts]

Prior-sensitivity analysis showed that learned temperature parameters depend on the assumed non-splice prior. As the validation negative weight increased toward the genome-wide prior, genome-prior ECE and NLL improved, while the number of argmax splice predictions decreased.

This confirms that temperature scaling is not prior-neutral in highly imbalanced per-position splice-site prediction. Calibration fit on a splice-enriched sampled distribution learns probabilities for that sampled task. Calibration fit with genome-prior weighting learns probabilities closer to the natural genome-position distribution.

## Argmax prediction is misleading under extreme class imbalance

Genome-prior weighted calibration produced the best probability calibration, but argmax predictions remained dominated by the non-splice class. This is expected because the natural class prior is overwhelmingly non-splice.

Therefore, multiclass argmax classification is not the right operating mode for splice-site discovery. A calibrated model can assign low absolute probabilities to splice sites while still ranking them highly relative to non-splice positions. Discovery should therefore rely on AUPRC, top-k recall, and threshold-specific precision/recall rather than argmax accuracy alone.

## Summary of results

In this flank-80 focal-loss model, true-logit genome-prior weighted vector temperature scaling gave the best calibrated probabilities under the natural genome-position prior. At the same time, acceptor and donor AUPRC remained near 0.991 and 0.995. These results support a calibration-detection separation: splice-site probability calibration depends on the assumed prior, while splice-site ranking remains strong.

# Discussion

This study shows that splice-site detection and probability calibration are distinct objectives in highly imbalanced per-nucleotide splice-site prediction. The flank-80 OpenSpliceAI-style model ranked acceptor and donor sites very well, with AUPRC near 0.991 and 0.995, but its probability scale depended strongly on the class prior used during calibration.

## Calibration depends on the target prior

The central result is that genome-prior weighted true-logit vector temperature scaling substantially improved calibration under the natural genome-position prior. Weighted multiclass ECE decreased from 0.005610 to 0.000044, and weighted NLL decreased from 0.006240 to 0.000818. These improvements were confirmed by bootstrap confidence intervals and reliability diagrams.

This result highlights a key issue in splice-site prediction benchmarks. To make evaluation computationally feasible, sampled datasets often retain positives while subsampling negatives. This changes the class prior. A calibration model fit to that sampled distribution learns probabilities for the sampled task, not for the genome-wide task. Therefore, calibrated splice-site probabilities are only meaningful when the calibration prior is stated explicitly.

## Detection can remain strong while probabilities shift

Genome-prior calibration changed the probability scale but did not substantially reduce ranking performance. Acceptor and donor AUPRC remained essentially unchanged after calibration. This means that the model can still rank true splice sites above non-splice positions even when its absolute probabilities are adjusted downward to reflect the rarity of splice sites across genomic positions.

This distinction matters for downstream interpretation. A low calibrated probability does not necessarily mean that a site is unimportant for discovery. It may reflect the extreme rarity of splice sites under the genome-wide prior. For candidate discovery, ranking metrics, top-k recall, AUPRC, and threshold-specific precision/recall are more appropriate than multiclass argmax accuracy.

## Thresholds should be selected after calibration

Calibration also changed threshold behavior. The common threshold of 0.5 produced very high precision but poor recall, while lower calibrated thresholds such as 0.01 provided much more useful recall with high precision. This suggests that thresholds should not be transferred blindly across calibration methods or class priors. Instead, thresholds should be selected according to the intended operating point.

## Relationship to prior splice prediction work

SpliceAI and OpenSpliceAI-style models are usually evaluated as splice-site detection systems. This work adds a probability-interpretation layer: it asks whether the scores produced by such a model can be interpreted as calibrated probabilities under a specified prior. The results show that strong splice-site detection does not automatically imply calibrated genome-wide probability estimates.

OpenSpliceAI includes temperature-scaling-based calibration, but our analysis emphasizes true-logit calibration, explicit genome-prior weighting, bootstrap uncertainty estimates, reliability diagrams, and prior-sensitivity analysis in an extremely imbalanced setting. The main contribution is therefore not a new splice-site detector, but an evaluation and calibration framework for interpreting splice-site probabilities.

## Limitations

This analysis used a flank-80 model trained for only two epochs under limited compute. Longer-context models may produce different calibration behavior because they can incorporate broader splice-regulatory context. Future work should test whether the same prior dependence holds for flank-400, flank-2000, and flank-10000 models.

The model was trained and evaluated using MANE annotation-derived labels. These labels capture annotated splice sites but do not fully represent tissue-specific splicing, cryptic splice usage, or RNA-seq-supported alternative junctions. Calibration against tissue-specific RNA-seq-derived labels may produce different probability interpretations.

The genome-prior analysis used sample weighting rather than repeated inference over every non-splice position. This was a practical approximation for compute-constrained evaluation. Larger-scale experiments could directly evaluate calibration across full chromosomes or genome-wide prediction outputs.

## Future work

Future work should extend this analysis to longer-context models, additional training checkpoints, and tissue-specific splice labels. A natural next step is uncertainty-aware tissue-specific splicing prediction, where uncertainty reflects not only whether a position is a splice site, but also whether splice usage varies across tissues, individuals, or biological contexts.

Variant-effect calibration is another important direction. SpliceAI-style models are often used to score donor or acceptor gain and loss events, but variant scores may have their own calibration behavior depending on the variant class, genomic distance from annotated splice sites, and tissue context.

## Conclusion

In highly imbalanced splice-site prediction, calibrated probability estimation and splice-site detection should be evaluated separately. The flank-80 focal-loss model ranks splice sites well, but its probability estimates depend on the assumed class prior. Genome-prior weighted true-logit vector scaling provides strong probability calibration under the natural genome-position prior while preserving acceptor and donor detection performance.
