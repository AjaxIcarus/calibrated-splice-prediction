# Project Status: Paper Draft Phase

## Current status

The coding and analysis phase for the true-logit calibration experiment is frozen.

The current manuscript draft is:

- `paper_draft/full_draft_v11.md`

The current supplementary materials file is:

- `paper_draft/supplementary_materials_v1.md`

The current bibliography file is:

- `paper_draft/references.bib`

## Final empirical result

The final analysis uses true pre-softmax logits from the flank-80 focal-loss OpenSpliceAI-style model.

Main checkpoint:

- `results/best_models/flank80_focal_epoch2_best.pt`

Main result:

Genome-prior weighted true-logit vector temperature scaling reduced weighted multiclass ECE from 0.005610 to 0.000044 and weighted multiclass NLL from 0.006240 to 0.000818, while preserving acceptor and donor AUPRC near 0.991 and 0.995.

## Frozen analysis outputs

Main result directories:

- `results/logit_cache_flank80_epoch2/`
- `results/logit_vector_temperature_flank80_epoch2/`
- `results/logit_detection_metrics_flank80_epoch2/`
- `results/logit_bootstrap_flank80_epoch2/`
- `results/logit_prior_sensitivity_flank80_epoch2/`

Main figures:

- `figures/logit_based/`

Main tables:

- `tables/logit_based/`

## Final scripts

The final analysis scripts are:

- `scripts/extract_logits_cache.py`
- `scripts/vector_temperature_from_logits_cache.py`
- `scripts/detection_metrics_from_logits_cache.py`
- `scripts/make_logit_results_tables_and_figures.py`
- `scripts/bootstrap_logit_metrics.py`
- `scripts/make_logit_reliability_diagrams.py`
- `scripts/prior_sensitivity_logit_calibration.py`

## Do not rerun unless necessary

Do not regenerate datasets, retrain the model, or rerun all calibration experiments unless there is a specific error to fix.

The next phase is manuscript polishing:

1. clean title and abstract
2. insert tables and figures
3. clean citations
4. decide paper format
5. prepare final PDF or technical report
