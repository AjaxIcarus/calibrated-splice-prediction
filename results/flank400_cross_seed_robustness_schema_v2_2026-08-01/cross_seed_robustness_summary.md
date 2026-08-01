# Flank-400 cross-seed robustness interpretation

## Proper scoring rules versus uncalibrated

### Global T=1.1
- Weighted multiclass ECE: -65.84% / -66.55% improvement.
- Weighted multiclass NLL: -58.34% / -59.42% improvement.
- Weighted multiclass Brier: -18.65% / -18.90% improvement.

### Unweighted true-logit vector-T
- Weighted multiclass ECE: +91.25% / +93.14% improvement.
- Weighted multiclass NLL: -42.45% / -28.38% improvement.
- Weighted multiclass Brier: -499.98% / -447.61% improvement.

### Target-weighted true-logit vector-T
- Weighted multiclass ECE: +98.87% / +98.64% improvement.
- Weighted multiclass NLL: +84.42% / +85.44% improvement.
- Weighted multiclass Brier: +36.31% / +38.27% improvement.

### OpenSpliceAI-style full-validation vector-T
- Weighted multiclass ECE: +99.13% / +98.92% improvement.
- Weighted multiclass NLL: +84.42% / +85.39% improvement.
- Weighted multiclass Brier: +36.49% / +37.77% improvement.

## Prior-aware paired comparison

- 0 of 10 seed-by-metric paired 95% bootstrap intervals exclude zero.
- Treat seed-level consistency as descriptive robustness: two trained seeds do not support a population-level variance estimate.
- The bootstrap unit is position and intervals quantify evaluated-position uncertainty, not training-seed, gene, or chromosome uncertainty.
- Target-weighted AUPRC values must not be mixed with legacy sample-based AUPRC values.
