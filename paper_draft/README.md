# Paper Draft Archive README

> **ARCHIVE STATUS — NOT CURRENT TCBB SUBMISSION AUTHORITY**
>
> This entire `paper_draft/` directory is retained for development history.
> Historical filenames such as `current`, `final`, `v10`, or `schema_v2`
> do **not** indicate current submission authority. The current corrected
> quantitative checkpoints are seed 11/epoch 12 and seed 23/epoch 13.
> See the repository root `README.md`, `CURRENT_SUBMISSION_AUTHORITY.md`,
> and `reproducibility/tcbb_v9/README.md`.


## Project

**Prior-dependent calibration for OpenSpliceAI-style splice-site prediction under extreme class imbalance**

This directory is a historical development archive containing manuscript drafts, supplementary-material drafts, bibliography work, figures, and planning notes from multiple stages of the project. It is preserved for traceability and is not the current TCBB submission authority.

---

## Main manuscript draft

Historical draft formerly treated as current at this development stage:

- `paper_draft/full_draft_v10.md`

Important section drafts:

- `paper_draft/abstract_v1.md`
- `paper_draft/introduction_v1.md`
- `paper_draft/related_work_v1.md`
- `paper_draft/methods_calibration_eval_v1.md`
- `paper_draft/results_logit_calibration_v1.md`
- `paper_draft/discussion_logit_calibration_v1.md`
- `paper_draft/figure_captions_v1.md`

---

## Supplementary materials

Historical supplementary skeleton from this development stage:

- `paper_draft/supplementary_materials_v1.md`

This includes notes on:

- true-logit extraction
- sampled-cache construction
- genome-prior weighting
- bootstrap confidence intervals
- prior-sensitivity analysis
- reliability diagrams
- implementation details

---

## Bibliography

Starter bibliography:

- `paper_draft/references.bib`

Reference planning file:

- `paper_draft/references_todo.md`

Core citation keys:

- `barash2010splicingcode`
- `jaganathan2019spliceai`
- `chao2025openspliceai`
- `guo2017calibration`
- `naeini2015calibration`
- `lin2017focalloss`

---

## Main result tables

Main tables:

- `tables/logit_based/logit_main_results_table.md`
- `tables/logit_based/logit_threshold_results_table.md`

Supplementary tables:

- `tables/logit_based/logit_bootstrap_summary.md`
- `tables/logit_based/logit_prior_sensitivity_summary.md`
- `tables/logit_based/logit_reliability_bins.csv`

---

## Main figures

Calibration figures:

- `figures/logit_based/logit_multiclass_ece.png`
- `figures/logit_based/logit_multiclass_nll.png`
- `figures/logit_based/logit_acceptor_ece.png`
- `figures/logit_based/logit_donor_ece.png`

Detection figure:

- `figures/logit_based/logit_detection_auprc.png`

Threshold figures:

- `figures/logit_based/logit_acceptor_threshold_precision_recall.png`
- `figures/logit_based/logit_donor_threshold_precision_recall.png`

Reliability figures:

- `figures/logit_based/reliability_multiclass_uncalibrated.png`
- `figures/logit_based/reliability_multiclass_logit_weighted_vector.png`
- `figures/logit_based/reliability_acceptor_uncalibrated_vs_weighted.png`
- `figures/logit_based/reliability_donor_uncalibrated_vs_weighted.png`

Prior-sensitivity figures:

- `figures/logit_based/logit_prior_sensitivity_temperatures.png`
- `figures/logit_based/logit_prior_sensitivity_genome_metrics.png`
- `figures/logit_based/logit_prior_sensitivity_argmax_counts.png`

---

## Historical analysis scripts for this development stage

True-logit calibration pipeline:

- `scripts/extract_logits_cache.py`
- `scripts/vector_temperature_from_logits_cache.py`
- `scripts/detection_metrics_from_logits_cache.py`
- `scripts/make_logit_results_tables_and_figures.py`
- `scripts/bootstrap_logit_metrics.py`
- `scripts/make_logit_reliability_diagrams.py`
- `scripts/prior_sensitivity_logit_calibration.py`

---

## Historical flank-80 empirical claim — superseded as current quantitative evidence

In extremely imbalanced per-nucleotide splice-site prediction, probability calibration and splice-site detection are separate objectives.

The flank-80 focal-loss model ranks acceptor and donor sites well, with acceptor and donor AUPRC near 0.991 and 0.995. However, calibrated probabilities depend strongly on the assumed class prior.

Genome-prior weighted true-logit vector temperature scaling reduced weighted multiclass ECE from 0.005610 to 0.000044 and weighted NLL from 0.006240 to 0.000818 while preserving splice-site ranking performance.

---

## Historical next-writing tasks

1. Convert the Markdown manuscript into a cleaner paper format.
2. Decide whether the target is workshop-style short paper, full paper, or technical report.
3. Add formal citations throughout the manuscript.
4. Insert actual tables and figures in place of callout placeholders.
5. Tighten the title, abstract, and contribution statement.
6. Add a reproducibility section or code/data availability statement.
