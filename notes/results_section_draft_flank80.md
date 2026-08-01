# Results Draft: Flank-80 Calibration and Detection Analysis

## Strong splice-site retrieval is preserved across calibration methods

We first evaluated the flank-80 focal-loss OpenSpliceAI model on a sampled held-out test set containing all annotated splice-site positives and a reservoir sample of non-splice positions. The model showed strong retrieval performance for both splice-site classes. The uncalibrated model achieved an AUPRC of 0.9909 for acceptor detection and 0.9950 for donor detection. Post-hoc calibration had almost no effect on ranking performance: global temperature scaling, unweighted vector temperature scaling, and genome-weighted vector temperature scaling all produced nearly identical AUPRC values.

This indicates that the model already ranks true splice sites above sampled non-splice positions effectively. Therefore, the main uncertainty problem is not splice-site retrieval, but the interpretation of the model's probability scale.

## Fixed probability thresholds are sensitive to calibration

Although ranking performance was high, fixed probability thresholds behaved very differently across calibration methods. For the uncalibrated model, a threshold of 0.5 yielded near-perfect precision but very low recall: 7.5% for acceptors and 14.3% for donors. Lower thresholds such as 0.05 or 0.1 gave much better recall while retaining high precision.

Genome-weighted vector calibration shifted splice-site probabilities downward, consistent with the extreme rarity of splice sites under the natural genome-wide class prior. Under this calibration, thresholds such as 0.01, 0.05, and 0.1 are more meaningful than 0.5. This shows that a universal threshold such as 0.5 is inappropriate for per-nucleotide splice-site prediction.

## Calibration depends on the assumed class prior

We next compared calibration under different post-hoc temperature scaling methods. Global scalar temperature scaling with T = 1.1 did not improve genome-weighted calibration. On the weighted test set, global temperature scaling increased multiclass ECE from 0.00561 to 0.00854 and increased NLL from 0.00624 to 0.00921.

Unweighted vector temperature scaling substantially improved calibration under the artificially splice-enriched sampled distribution, but it changed many argmax predictions and substantially increased apparent splice-site recall. This behavior is explained by the sampling design: the sampled test set contains all positives but only a subset of non-splice positions, making splice sites much more frequent than they are under the natural genome-wide distribution.

To correct for this, we fit a genome-weighted vector temperature model where sampled non-splice positions were weighted according to their original population frequency. This produced the best genome-prior calibration, reducing weighted multiclass ECE to 0.00034 and weighted NLL to 0.00105 on the held-out test sample.

## Argmax classification is not appropriate for calibrated splice-site discovery

Genome-weighted vector calibration produced well-calibrated probabilities under the natural genome-wide class distribution, but its argmax predictions were dominated by the non-splice class. This is expected because true splice sites are extremely rare at the per-nucleotide level. Under a calibrated posterior, a random genomic position is almost always most likely to be non-splice.

Therefore, argmax accuracy is not an appropriate primary metric for splice-site discovery. Ranking metrics, top-k retrieval, thresholded precision/recall, and probability calibration answer different questions and should be reported separately.

## Main finding

These results show that in extremely imbalanced per-nucleotide splice-site prediction, detection and calibration must be evaluated as separate objectives. The flank-80 focal-loss model achieves strong splice-site retrieval, while post-hoc calibration mainly changes the probability scale. Genome-weighted calibration gives the best posterior probability estimates under the natural class prior, but calibrated probabilities should not be interpreted through naive argmax classification.