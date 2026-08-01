# True-logit Calibration Results: Flank-80 Epoch-2 Model

## Model and data

Checkpoint:

`results/best_models/flank80_focal_epoch2_best.pt`

Dataset:

- Validation positions seen: 90,900,000
- Test positions seen: 416,160,000
- Test sampled positions: 606,088
- Test sampled non-splice: 500,000
- Test acceptor positives: 53,024
- Test donor positives: 53,064
- Test negative weight: 832.107824

## Final temperatures

Unweighted true-logit vector temperature:

- T_nonsplice = 0.811902
- T_acceptor = 0.248638
- T_donor = 0.345119

Genome-weighted true-logit vector temperature:

- T_nonsplice = 0.435169
- T_acceptor = 0.416153
- T_donor = 0.427941

## Main calibration results

| Method | Weighted multiclass ECE | Weighted multiclass NLL | Acceptor AUPRC | Donor AUPRC |
|---|---:|---:|---:|---:|
| Uncalibrated | 0.005610 | 0.006240 | 0.9909 | 0.9950 |
| Global T=1.1 | 0.008538 | 0.009212 | 0.9909 | 0.9950 |
| Logit unweighted vector T | 0.000445 | 0.010417 | 0.9881 | 0.9932 |
| Logit genome-weighted vector T | 0.000044 | 0.000818 | 0.9909 | 0.9950 |

## Bootstrap confidence intervals

Genome-weighted true-logit vector T:

- Weighted multiclass ECE = 0.000049 [0.000039, 0.000058]
- Weighted multiclass NLL = 0.000818 [0.000804, 0.000832]
- Acceptor AUPRC = 0.990906 [0.990350, 0.991494]
- Donor AUPRC = 0.994977 [0.994573, 0.995355]

## Main interpretation

Genome-weighted true-logit vector temperature scaling gives the best genome-prior calibration, reducing weighted multiclass ECE from 0.005610 to 0.000044 and weighted multiclass NLL from 0.006240 to 0.000818.

Detection remains very strong after calibration. Acceptor and donor AUPRC remain near 0.991 and 0.995 respectively.

The results show that calibration is prior-dependent. Unweighted calibration fits the splice-enriched sampled distribution, while genome-weighted calibration fits the natural genome-wide class prior.

Argmax prediction is not the right operating mode for splice-site discovery under extreme class imbalance. Calibration should be evaluated under an explicit prior, while discovery should be evaluated using ranking, threshold, and AUPRC metrics.

## Official output files

Tables:

- `tables/logit_based/logit_main_results_table.md`
- `tables/logit_based/logit_threshold_results_table.md`
- `tables/logit_based/logit_bootstrap_summary.md`
- `tables/logit_based/logit_prior_sensitivity_summary.md`
- `tables/logit_based/logit_reliability_bins.csv`

Figures:

- `figures/logit_based/logit_multiclass_ece.png`
- `figures/logit_based/logit_multiclass_nll.png`
- `figures/logit_based/logit_detection_auprc.png`
- `figures/logit_based/reliability_multiclass_uncalibrated.png`
- `figures/logit_based/reliability_multiclass_logit_weighted_vector.png`
- `figures/logit_based/reliability_acceptor_uncalibrated_vs_weighted.png`
- `figures/logit_based/reliability_donor_uncalibrated_vs_weighted.png`
- `figures/logit_based/logit_prior_sensitivity_temperatures.png`
- `figures/logit_based/logit_prior_sensitivity_genome_metrics.png`
- `figures/logit_based/logit_prior_sensitivity_argmax_counts.png`