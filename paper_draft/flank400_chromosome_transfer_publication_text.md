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

# Flank-400 cross-seed publication text

Generated from the frozen 2026-07-24 chromosome-transfer milestone.
Primary: seed 11, epoch 11. Replication: seed 23, epoch 14.

## Suggested Results subsection

### Calibration transfers across chromosomes and training runs

Across five evaluated test chromosomes (chr1, chr3, chr5, chr7, and
chr9), the qualitative calibration result replicated in the
validation-selected seed-11/epoch-11 primary model and the independently
trained, validation-selected seed-23/epoch-14 replication model. Each
model was evaluated over
416,140,000 positions represented in the held-out
gene/transcript records. In the primary model, pooled multiclass ECE
and NLL were 1.507e-03
and 1.698e-03
before calibration. OpenSpliceAI-style vector temperature scaling
reduced them to
4.554e-06
and 2.442e-04,
respectively. The replication yielded the same ordering, with
uncalibrated ECE/NLL of
1.545e-03/
1.726e-03
and OpenSpliceAI-style values of
3.085e-06/
2.305e-04.
Relative to the uncalibrated predictions, this corresponds to ECE
reductions of 99.70% and 99.80% and NLL reductions of 85.62% and
86.64% in the primary and replication models, respectively.

Genome-prior-weighted vector scaling also improved both probability
metrics in each model, yielding ECE/NLL of
2.087e-05/
2.522e-04
in the primary model and
2.739e-05/
2.404e-04
in the replication. By contrast, unweighted vector scaling lowered ECE
but increased NLL relative to the uncalibrated model in both runs, and
fixed global T=1.1 worsened both metrics. Thus, the qualitative
conclusion was stable across the two training runs: prior-aware,
class-specific scaling improved both probability metrics, whereas a
small ECE alone did not imply a better likelihood.

These comparisons are descriptive replications across two trained
models and five selected test chromosomes. They are not a
random-effects test over chromosomes and do not represent complete
intergenic whole-genome inference.

## Figure caption

**Figure X. Five-chromosome calibration transfer across independently
trained flank-400 models.** Pooled multiclass ECE (a) and NLL (b) are
shown for the validation-selected seed-11/epoch-11 primary model and
seed-23/epoch-14 replication over chr1, chr3, chr5, chr7, and chr9
(416,140,000 evaluated positions per model). Gray segments
pair the two training runs within each calibration method. Both axes
use logarithmic scales; lower values indicate better calibration.
OpenSpliceAI-style vector temperature scaling achieved the lowest
pooled ECE and NLL in both runs, while genome-prior-weighted vector
scaling also improved both metrics.

## Table caption

**Table X. Cross-seed robustness of five-chromosome flank-400
calibration transfer.** Pooled multiclass ECE and NLL are reported for
the seed-11/epoch-11 primary model and seed-23/epoch-14 replication.
Each model covers 416,140,000 positions across chr1, chr3,
chr5, chr7, and chr9. Lower is better. Results are descriptive across
two training runs.

## Suggested abstract sentence

Across two independently trained flank-400 models, streaming
evaluation over five test chromosomes (416,140,000
positions per model) reproduced the calibration ordering:
OpenSpliceAI-style vector temperature scaling had the lowest pooled
ECE and NLL, while genome-prior-weighted vector scaling also improved
both metrics.

## Suggested limitation replacement

The main flank-400 calibration pattern replicated in a second
independently trained model, reducing dependence on a single
checkpoint. Nevertheless, two validation-selected training runs do
not fully characterize optimization variability, and the five
evaluated chromosomes should not be treated as randomly sampled
replicate units.
