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
