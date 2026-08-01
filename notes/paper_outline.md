# Paper Outline

## Working Title

Calibration and Detection in Extremely Imbalanced Splice-Site Prediction

## Core Claim

Per-nucleotide splice-site prediction is an extreme class-imbalance problem where detection performance and probability calibration must be evaluated separately. A model can rank splice sites very well while still requiring prior-aware calibration before its probabilities can be interpreted.

---

# 1. Introduction

## Problem

RNA splicing is a central biological process, and mis-splicing is associated with disease. Deep learning models such as SpliceAI and OpenSpliceAI predict splice acceptor and donor sites from DNA sequence, but their probability scores are difficult to interpret under extreme class imbalance.

## Gap

Most splice-site prediction work emphasizes detection metrics such as AUPRC, top-k accuracy, precision, recall, or variant effect prediction. However, when model outputs are used as confidence estimates, uncertainty scores, or downstream decision probabilities, calibration becomes important.

## Key challenge

At the per-nucleotide level, true splice sites are extremely rare. Therefore:

- argmax classification is dominated by the non-splice class
- fixed thresholds such as 0.5 are inappropriate
- calibration depends strongly on the assumed class prior
- ranking performance and probability calibration answer different questions

## Contributions

1. We train a flank-80 OpenSpliceAI-style model with focal loss on human GRCh38 + MANE splice annotations.
2. We evaluate splice-site retrieval and probability calibration separately.
3. We compare scalar, unweighted vector, and genome-weighted vector temperature scaling.
4. We show that ranking performance remains strong across calibration methods, while calibrated probabilities change substantially depending on the assumed prior.
5. We argue that calibrated splice-site prediction should report both retrieval metrics and prior-aware calibration metrics.

---

# 2. Related Work

## Splicing codes and sequence-based splicing prediction

Discuss early splicing-code work such as Barash et al., which used combinations of RNA features to predict tissue-dependent alternative splicing.

## Deep learning for splice prediction

Discuss SpliceAI as a deep residual neural network that predicts acceptor, donor, or neither at each sequence position. Emphasize long-range sequence context and strong splice-site detection.

## OpenSpliceAI

Discuss OpenSpliceAI as a PyTorch implementation of SpliceAI that supports retraining, transfer learning, and calibration.

## Calibration and uncertainty

Discuss why calibrated probabilities matter. Explain temperature scaling, ECE, NLL, and the difference between ranking and calibration.

---

# 3. Methods

## Dataset

- Human GRCh38 primary assembly
- MANE v1.3 RefSeq genomic annotation
- OpenSpliceAI create-data pipeline
- protein-coding genes
- chromosome-based human split
- flank size 80
- labels: non-splice, acceptor, donor

## Model

- OpenSpliceAI-style architecture
- flank-80 context
- focal loss
- CPU training with low-memory patch
- best checkpoint: epoch 2

## Evaluation sample

- all positives retained
- reservoir-sampled non-splice positions
- test sample: 606,088 positions
- positives: 106,088
- sampled negatives: 500,000
- original test positions: 416,160,000
- original negatives: 416,053,912

## Calibration methods

- global scalar temperature scaling
- unweighted vector temperature scaling
- genome-weighted vector temperature scaling

## Metrics

Calibration:

- multiclass ECE
- multiclass NLL
- multiclass Brier
- acceptor/donor binary ECE
- acceptor/donor binary NLL
- acceptor/donor binary Brier

Detection:

- AUPRC
- top-k precision/recall
- thresholded precision/recall

---

# 4. Results

## Result 1: Strong splice-site retrieval

The uncalibrated model achieves high AUPRC:

- acceptor AUPRC: 0.9909
- donor AUPRC: 0.9950

Temperature scaling has almost no effect on AUPRC, showing that ranking is already strong.

## Result 2: Fixed thresholds are misleading

At threshold 0.5, the uncalibrated model has near-perfect precision but very low recall:

- acceptor recall: 0.075
- donor recall: 0.143

Lower thresholds give much better retrieval.

## Result 3: Calibration depends on prior

Global T = 1.1 does not improve genome-weighted calibration.

Unweighted vector calibration improves calibration on the splice-enriched sampled distribution but changes argmax predictions substantially.

Genome-weighted vector calibration gives the best weighted calibration:

- weighted multiclass ECE: 0.00034
- weighted multiclass NLL: 0.00105

## Result 4: Argmax is not a suitable splice-site detector

Genome-weighted calibration predicts almost all positions as non-splice under argmax. This is expected because non-splice positions dominate the natural genome-wide prior.

Therefore, splice-site discovery should use ranking or thresholded retrieval, not naive argmax classification.

---

# 5. Discussion

## Main interpretation

The model is good at ranking splice sites, but probability calibration depends on the target deployment distribution.

## Why this matters

If probabilities are used as confidence estimates, variant evidence, downstream model inputs, or uncertainty scores, calibration must be prior-aware.

## Detection vs calibration

Detection asks:

> Are true splice sites ranked above non-splice positions?

Calibration asks:

> Does a predicted probability correspond to the empirical probability of correctness under a particular distribution?

These are different objectives.

## Practical recommendation

Report both:

- ranking metrics such as AUPRC/top-k
- calibration metrics such as ECE/NLL under the intended prior

---

# 6. Limitations

1. The model uses flank size 80, not 400/2000/10000.
2. Training was CPU-limited and used a single focal-loss checkpoint.
3. Evaluation uses sampled negatives, requiring weighting to approximate genome-wide calibration.
4. Probability-space log scaling was used for calibration because cached true logits were not saved.
5. The study is sequence-only and does not include tissue-specific RNA-seq labels.
6. The current task predicts annotated splice sites, not tissue-specific alternative splice usage.

---

# 7. Future Work

1. Cache true logits and repeat vector temperature scaling directly on logits.
2. Evaluate larger flank sizes if compute becomes available.
3. Compare focal loss vs cross-entropy.
4. Add candidate-window evaluation around splice-like motifs.
5. Extend from sequence-only splice-site prediction to tissue-specific splicing or PSI prediction.
6. Evaluate calibration under held-out genes, held-out chromosomes, and possibly distribution shifts.

---

# 8. Main Figures

## Figure 1

Detection AUPRC across calibration methods.

File:

`figures/flank80_epoch2_detection_auprc.png`

## Figure 2

Weighted multiclass ECE across calibration methods.

File:

`figures/flank80_epoch2_multiclass_ece.png`

## Figure 3

Weighted multiclass NLL across calibration methods.

File:

`figures/flank80_epoch2_multiclass_nll.png`

## Figure 4

Threshold precision/recall behavior.

Files:

- `figures/flank80_epoch2_acceptor_threshold_precision_recall.png`
- `figures/flank80_epoch2_donor_threshold_precision_recall.png`

---

# 9. One-paragraph Abstract Draft

Deep learning models can predict splice acceptor and donor sites from DNA sequence with high accuracy, but their probability scores are difficult to interpret under the extreme class imbalance of per-nucleotide splice-site prediction. We trained a flank-80 OpenSpliceAI-style model with focal loss on human GRCh38 and MANE annotations and evaluated splice-site detection separately from probability calibration. The model achieved strong held-out retrieval performance, with AUPRC of 0.9909 for acceptors and 0.9950 for donors. However, post-hoc calibration behaved differently depending on the assumed class prior. Global scalar temperature scaling did not improve genome-weighted calibration, while genome-weighted vector temperature scaling substantially reduced weighted multiclass ECE and NLL. At the same time, calibrated argmax predictions were dominated by the non-splice class, showing that argmax classification is unsuitable for splice-site discovery under the natural genome-wide prior. These results demonstrate that splice-site prediction systems should report ranking-based detection and prior-aware probability calibration as separate evaluation objectives.