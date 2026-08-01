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

## Main text figure

### Figure 1: Matched schema-v2 reliability diagnostics

- `figures/flank400_schema_v2_reliability_seed11.png`
- `figures/flank400_schema_v2_reliability_seed23.png`

Use both accepted combined panels. Explain that jagged rare-class bins carry little target-population mass and that proper scoring rules prevent ECE-only interpretation.

## Main text table

### Table 1: Matched schema-v2 cross-seed probability quality

Source: `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_core_point_estimates.csv`.

Report all five methods and five core metrics as seed-11/seed-23 pairs. Label AUPRC explicitly as target-weighted. Do not insert sampled-cache AUPRC near 0.999 into this table.

## Supplementary tables

### Table S1: Full 13-metric point estimates

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_point_estimates_all_metrics.csv`

### Table S2: Changes relative to uncalibrated

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_change_vs_uncalibrated.csv`

### Table S3: Paired prior-aware comparisons

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_prior_aware_paired_comparison.csv`

State that none of ten intervals exclude zero and that position-level intervals do not estimate training-seed variability.

### Table S4: Reliability bins and provenance

- Seed-11 schema-v2 reliability output directory.
- Seed-23 schema-v2 reliability output directory.
- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_metadata.json`

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
