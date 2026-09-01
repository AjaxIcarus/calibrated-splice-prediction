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

# Figure and Table Plan

## Authoritative evidence

- Primary flank-400 model: seed 11, epoch 11.
- Independent replication: seed 23, epoch 14.
- Schema-v2 test population: 84,258 sequences and 421,290,000 raw positions.
- Valid target population: 407,278,622 positions after excluding 14,011,378 all-zero padding positions.
- Valid classes: 407,170,634 non-splice, 53,994 acceptor, and 53,994 donor positions.
- Each sampled test cache: all 107,988 positives plus 500,000 valid non-splice positions.
- Test/validation non-splice weights: 814.341268 / 177.980664.
- Bootstrap: 200 paired position-level replicates, stratified by true class, seed 17.
- Epoch-8, pre-schema-v2, single-chr9, and five-chromosome raw-position outputs are historical only.
- Flank-80 is developmental workflow validation only. Its pre-schema-v2 calibration numbers are not final-paper evidence.

## Main text figure

### Figure 1: Matched schema-v2 reliability diagnostics

- `figures/flank400_schema_v2_reliability_seed11.png`
- `figures/flank400_schema_v2_reliability_seed23.png`

Use both accepted combined panels. State that curves connect nonempty 15-bin equal-width estimates and that the legend reports multiclass ECE, not class-specific ECE. Explain that jagged rare-class bins carry little target-population mass and that proper scoring rules prevent ECE-only interpretation.

## Main text table

### Table 1: Matched schema-v2 cross-seed probability quality

Source: `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_core_point_estimates.csv`.

Report all five methods and five core metrics as seed-11/seed-23 pairs. Label AUPRC explicitly as target-weighted. Do not insert sampled-cache AUPRC near 0.999 into this table.

## Printed supplementary tables

### Table S1: Paired prior-aware comparisons

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_prior_aware_paired_comparison.csv`

Print the ten seed-by-metric OpenSpliceAI-style-versus-target-weighted contrasts, including the paired 95% interval and the direction convention. State that none excludes zero and that position-level intervals do not estimate training-seed variability.

### Table S2: Population and analysis provenance

- Valid and padding-excluded position census.
- Sampled-cache class counts and reconstruction weights.
- Bootstrap unit, stratification, replicate count, and seed.
- Point-estimate and cache hashes from `cross_seed_metadata.json`.

## Machine-readable supplementary data

### Supplementary Data S1: Full 13-metric point estimates

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_point_estimates_all_metrics.csv`

Do not print this very wide table in the PDF; provide the CSV with metric definitions.

### Supplementary Data S2: Changes relative to uncalibrated

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_change_vs_uncalibrated.csv`

Retain absolute and percentage changes with explicit lower-is-better/higher-is-better directions.

### Supplementary Data S3: Reliability bins and provenance

- Seed-11 schema-v2 raw reliability-bin table and metadata.
- Seed-23 schema-v2 raw reliability-bin table and metadata.
- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_metadata.json`

Preserve, for every method, reliability type, class, and bin: raw count, reconstructed target weight, target-weight fraction, mean predicted probability, observed frequency or accuracy, absolute gap, and ECE contribution. Keep these as machine-readable CSV/JSON rather than a hundreds-of-rows printed table.

## Developmental flank-80 placement

- Mention in Methods that flank-80 established low-memory training and true-logit extraction.
- State that schema-v2 re-extraction confirmed complete sequence coverage, padding exclusion, and paired labels.
- Do not report its earlier point estimates, bootstrap intervals, reliability plots, or flank-80/flank-400 performance table unless the complete five-method schema-v2 evaluation is rerun.
- Do not imply a causal context-length comparison.

## Operating-point material

Include only validation-selected operating points generated from valid-only schema-v2 caches. Keep sampled and target-weighted AUPRC explicitly separated.

## Retired from the main manuscript

- `figures/flank400_chromosome_transfer_cross_seed.pdf`
- `figures/flank400_primary_calibration.png`
- `figures/flank400_chr9_calibration.png`
- all epoch-8 flank-400 tables and figures
- pre-schema-v2 reliability figures and bootstrap summaries
- the five-chromosome 416,140,000 raw-position table
- single-chr9 calibration and argmax tables
- legacy sampled-cache AUPRC values when the table is labeled target-population or target-weighted
