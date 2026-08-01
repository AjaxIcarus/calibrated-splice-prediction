# Schema-v2 manuscript polish decisions

## Main-paper decisions

1. Keep the matched flank-400 schema-v2 analysis as the sole quantitative experiment.
2. Remove checkpoint, dataset, and cache paths from the manuscript body. Preserve them only in the reproducibility materials.
3. Keep both accepted reliability panels in the main paper because replication is central to the argument.
4. Clarify that reliability curves connect nonempty equal-width bins and that legend ECE values are multiclass, not class-specific.
5. Keep Table 1 as point estimates. Put the ten paired prior-aware bootstrap contrasts and intervals in Supplementary Table S1.

## Flank-80 decision

Flank-80 remains useful as developmental validation of low-memory training, true-logit extraction, complete sequence coverage, and padding exclusion. Its earlier calibration numbers predate schema v2, and no complete five-method schema-v2 recalibration with paired bootstrap was run. It is therefore described in Methods and the supplement, not presented as a quantitative result or context-length comparison.

## Supplementary placement

- Print the compact ten-row paired-contrast table and the population/provenance table.
- Release the full 13-metric estimates and baseline-relative changes as CSV.
- Release raw reliability bins and metadata as CSV/JSON. Do not print hundreds of bin rows.
- Preserve sampled-cache and target-weighted AUPRC as explicitly different metrics.

## Claims retained

- Both prior-aware procedures sharply improve target-weighted ECE, NLL, and Brier relative to uncalibrated predictions in both trained models.
- Target-weighted AUPRC changes minimally under prior-aware calibration.
- Global T=1.1 worsens all three probability-quality metrics.
- Unweighted vector scaling improves ECE while worsening NLL and Brier, demonstrating that ECE alone is insufficient.
- None of the ten paired prior-aware intervals excludes zero; no superiority or equivalence claim is supported.
- Two trained seeds provide descriptive robustness, not a population-level estimate of training-seed variability.
