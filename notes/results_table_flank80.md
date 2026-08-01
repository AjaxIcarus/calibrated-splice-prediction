# Results Table: Flank-80 OpenSpliceAI Experiments

## Best model

`results/best_models/flank80_focal_epoch2_best.pt`

## Training / validation results

| Run | Dataset | Loss | Epoch / continuation | Acceptor top-k | Donor top-k | Acceptor AUPRC | Donor AUPRC | Acceptor F1 | Donor F1 | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| medium debug | 20x256 train, 8x256 val | cross_entropy_loss | 3 | 0.0000 | 0.0000 | ~0.0008 | ~0.0011 | 0.0000 | 0.0000 | collapsed to non-splice |
| medium debug | 20x256 train, 8x256 val | focal_loss | 3 | 0.0489 | 0.1300 | 0.0159 | 0.0334 | 0.0000 | 0.0000 | ranking improved |
| large debug | 60x512 train, 14x512 val | focal_loss | 3 | 0.2122 | 0.2897 | 0.1254 | 0.2260 | 0.0282 | 0.1535 | clear splice signal |
| full lowmem | full train, validation | focal_loss | epoch 1 | 0.2439 | 0.3472 | 0.1837 | 0.3042 | 0.1304 | 0.2454 | first full checkpoint |
| full lowmem | full train, validation | focal_loss | epoch 2 | 0.4018 | 0.4746 | 0.3729 | 0.4624 | 0.1745 | 0.3024 | best model |
| full lowmem | full train, validation | focal_loss | epoch 3 | 0.3981 | 0.4329 | 0.3435 | 0.4079 | TBD | TBD | worse than epoch 2 |

## Approximate chunked test results for epoch-2 model

| Metric | Approximate chunk average |
|---|---:|
| Acceptor top-k | ~0.3445 |
| Donor top-k | ~0.3975 |
| Acceptor AUPRC | ~0.3008 |
| Donor AUPRC | ~0.3650 |
| Acceptor F1 | ~0.1360 |
| Donor F1 | ~0.2144 |

Note: chunk averages are not exact global test metrics.

## Sampled test calibration, epoch-2 model

Sampling:
- all positive positions
- 500,000 reservoir-sampled non-splice positions
- total test positions seen: 416,160,000
- positive positions: 106,088
- total sampled positions: 606,088

| Metric | Uncalibrated | Temperature-scaled T=1.1 | Delta |
|---|---:|---:|---:|
| Multiclass ECE | 0.1016 | 0.1038 | +0.0023 |
| Multiclass NLL | 0.2374 | 0.2278 | -0.0095 |
| Multiclass Brier | 0.2031 | 0.1956 | -0.0075 |
| Acceptor ECE | 0.0605 | 0.0610 | +0.0004 |
| Acceptor NLL | 0.1262 | 0.1206 | -0.0056 |
| Acceptor Brier | 0.0470 | 0.0451 | -0.0019 |
| Donor ECE | 0.0579 | 0.0590 | +0.0011 |
| Donor NLL | 0.1112 | 0.1073 | -0.0039 |
| Donor Brier | 0.0416 | 0.0400 | -0.0016 |

## Interpretation

Global temperature scaling improves NLL and Brier score on held-out sampled test data, but slightly worsens ECE. This suggests global temperature scaling only partially calibrates splice-site probabilities.