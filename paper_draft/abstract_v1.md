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

## Abstract

Deep learning models for splice-site prediction are typically evaluated by detection performance, but their output scores are often interpreted as calibrated probabilities. This distinction is important in per-nucleotide splice-site prediction, where acceptor and donor sites are extremely rare relative to non-splice genomic positions. We evaluate probability calibration for a flank-80 OpenSpliceAI-style model trained on human GRCh38/MANE annotations using focal loss. Using true pre-softmax logits, we compare uncalibrated predictions, global temperature scaling, unweighted vector temperature scaling, and genome-prior weighted vector temperature scaling. Genome-weighted true-logit vector scaling reduced weighted multiclass expected calibration error from 0.005610 to 0.000044 and weighted negative log-likelihood from 0.006240 to 0.000818, while preserving strong splice-site detection performance with acceptor and donor AUPRC near 0.991 and 0.995. Bootstrap confidence intervals, reliability diagrams, and prior-sensitivity analysis confirmed that calibration depends strongly on the assumed class prior. These results show that calibrated probability estimation and splice-site detection are separate objectives under extreme class imbalance, and that splice-site probabilities should be interpreted only relative to an explicit evaluation prior.
