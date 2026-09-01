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

# Figure and Table Captions

## Figures

**Figure 1. Genome-prior calibration improves after true-logit vector temperature scaling.** Weighted multiclass expected calibration error and negative log-likelihood are shown for the uncalibrated model, global scalar temperature scaling, unweighted vector temperature scaling, and genome-prior weighted vector temperature scaling. Genome-prior weighted vector scaling gives the best calibration under the natural genome-position prior.

**Figure 2. Splice-site ranking remains strong after calibration.** Acceptor and donor AUPRC remain high across calibration methods, indicating that calibration changes probability interpretation without substantially degrading splice-site ranking.

**Figure 3. Reliability diagrams before and after genome-prior weighted calibration.** Reliability curves compare predicted probabilities with empirical frequencies for multiclass, acceptor, and donor predictions. Genome-prior weighted true-logit vector scaling improves agreement between predicted confidence and observed frequency under the genome-position prior.

**Figure 4. Calibration changes threshold precision-recall behavior.** Precision and recall are shown across probability thresholds for acceptor and donor detection. Although ranking performance remains stable, calibrated probabilities require threshold selection based on the desired operating point.

**Figure 5. Temperature scaling is sensitive to the assumed class prior.** As the validation non-splice weight increases toward the genome-wide prior, learned temperature parameters and argmax prediction counts change, while genome-prior calibration improves. This shows that calibrated probabilities must be interpreted relative to an explicit prior.

## Tables

**Table 1. Main calibration and detection results for the flank-80 epoch-2 model.** Weighted multiclass ECE and NLL evaluate genome-prior calibration; acceptor and donor AUPRC evaluate splice-site detection.

**Table 2. Threshold precision and recall for acceptor and donor detection.** Threshold-specific metrics show how calibration changes the probability scale used for candidate splice-site discovery.

## Supplementary Tables

**Supplementary Table S1. Bootstrap confidence intervals for calibration and detection metrics.** Bootstrap intervals confirm that genome-prior weighted vector scaling substantially improves calibration while preserving acceptor and donor AUPRC.

**Supplementary Table S2. Prior-sensitivity analysis.** Learned temperature parameters, genome-prior calibration metrics, and argmax prediction counts vary as the assumed validation non-splice weight changes.

**Supplementary Table S3. Reliability bin outputs.** Bin-level predicted confidence and empirical frequency values used to generate reliability diagrams.
