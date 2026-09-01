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

## Introduction

RNA splicing is a central mechanism by which eukaryotic cells generate transcript diversity from genomic sequence. Alternative splicing affects most multi-exon human genes and is regulated by combinations of cis-regulatory motifs, transcript structure, splice-site strength, and cellular context. Early computational work on the “splicing code” showed that combinations of RNA features could predict tissue-dependent splicing patterns, establishing splicing prediction as a problem of learning regulatory sequence logic rather than detecting splice motifs alone [Barash et al., 2010].

More recent deep learning models have shifted this problem from hand-engineered regulatory features toward sequence-to-sequence prediction directly from primary DNA sequence. SpliceAI introduced a deep residual convolutional model that predicts, for each position in a pre-mRNA sequence, whether that position is a splice acceptor, splice donor, or neither [Jaganathan et al., 2019]. This framing made splice-site prediction a per-nucleotide classification problem and enabled accurate prediction of canonical and cryptic splice sites from sequence context. OpenSpliceAI later provided a modular PyTorch implementation of this approach, enabling easier retraining, transfer learning, prediction, variant analysis, and model calibration [Chao et al., 2025].

Most evaluations of splice-site prediction models emphasize detection: whether true acceptor and donor sites receive higher scores than non-splice positions. This is appropriate for annotation, variant prioritization, and candidate splice-site discovery. However, model outputs are also often interpreted as confidence estimates. A score near 0.9 may be read as high confidence, while a score near 0.01 may be read as low confidence. This interpretation assumes that the model’s probabilities are calibrated: predicted probabilities should match empirical event frequencies.

Calibration is especially challenging in per-nucleotide splice-site prediction because the class distribution is extremely imbalanced. Across the genome, nearly all positions are non-splice positions, while acceptor and donor splice sites are rare. A sampled benchmark that retains all splice sites but subsamples non-splice positions can be useful for efficient evaluation, but it changes the class prior. As a result, probabilities calibrated under a splice-enriched sampled distribution may not represent probabilities under the natural genome-wide distribution.

This creates a distinction between two objectives. Detection asks whether true splice sites are ranked above non-splice positions. Calibration asks whether predicted probabilities match empirical frequencies under a specified class prior. A model can perform well at detection while remaining miscalibrated as a probability estimator under the genome-wide prior.

In this work, we study this calibration-detection distinction using a flank-80 OpenSpliceAI-style model trained with focal loss on human GRCh38/MANE splice annotations. We extract true pre-softmax logits, fit temperature-scaling models, and compare unweighted calibration against genome-prior weighted calibration. We evaluate calibration using expected calibration error, negative log-likelihood, reliability diagrams, and bootstrap confidence intervals, while evaluating detection separately using acceptor/donor AUPRC and threshold precision-recall.

Our main result is that genome-prior weighted true-logit vector temperature scaling substantially improves probability calibration under the natural genome-position prior without sacrificing splice-site ranking performance. This shows that, in highly imbalanced splice-site prediction, calibrated probability estimation and splice-site detection should be evaluated separately, and calibrated probabilities should always be interpreted relative to an explicit class prior.
