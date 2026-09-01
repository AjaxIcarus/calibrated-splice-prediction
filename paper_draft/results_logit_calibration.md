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