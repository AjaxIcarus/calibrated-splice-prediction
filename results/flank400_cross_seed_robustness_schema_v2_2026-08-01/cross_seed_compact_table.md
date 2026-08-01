# Flank-400 matched cross-seed robustness (schema v2)

Values are seed-11 epoch-11 / seed-23 epoch-14.

| Method | Weighted ECE | Weighted NLL | Weighted Brier | Acceptor AUPRC | Donor AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated | 1.430e-03 / 1.480e-03 | 1.626e-03 / 1.671e-03 | 2.101e-04 / 2.095e-04 | 0.892620 / 0.893685 | 0.915186 / 0.922882 |
| Global T=1.1 | 2.371e-03 / 2.465e-03 | 2.574e-03 / 2.663e-03 | 2.493e-04 / 2.491e-04 | 0.892617 / 0.893686 | 0.915174 / 0.922881 |
| Unweighted true-logit vector-T | 1.252e-04 / 1.016e-04 | 2.316e-03 / 2.145e-03 | 1.261e-03 / 1.147e-03 | 0.892310 / 0.882316 | 0.901449 / 0.917532 |
| Target-weighted true-logit vector-T | 1.615e-05 / 2.006e-05 | 2.532e-04 / 2.432e-04 | 1.338e-04 / 1.293e-04 | 0.892953 / 0.893145 | 0.914788 / 0.923178 |
| OpenSpliceAI-style full-validation vector-T | 1.241e-05 / 1.602e-05 | 2.532e-04 / 2.441e-04 | 1.334e-04 / 1.304e-04 | 0.893008 / 0.893747 | 0.915055 / 0.922969 |

ECE, NLL, and Brier are lower-is-better; AUPRC is higher-is-better.
All values use the matched schema-v2 valid-position target population.
