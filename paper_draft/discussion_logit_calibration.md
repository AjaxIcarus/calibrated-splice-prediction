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