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
