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

Our contributions are:

1. We evaluate splice-site prediction calibration using true pre-softmax logits rather than probability-derived logit proxies.
2. We show that calibration depends strongly on the assumed class prior in highly imbalanced per-nucleotide splice-site prediction.
3. We introduce genome-prior weighted vector temperature scaling for calibrated splice-site probability interpretation.
4. We show that genome-prior calibration improves ECE and NLL while preserving acceptor and donor AUPRC.
5. We separate calibration evaluation from splice-site detection evaluation using reliability diagrams, bootstrap confidence intervals, prior-sensitivity analysis, AUPRC, and threshold precision-recall.
