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
