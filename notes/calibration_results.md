# Calibration Results: Flank-80 Focal Epoch-2 Model

Best model:

`results/best_models/flank80_focal_epoch2_best.pt`

## Sampled held-out test calibration

Dataset:

`data/processed_h5/dataset_test.h5`

Sampling:

- All positive splice positions
- 500,000 uniformly reservoir-sampled non-splice positions
- Total positions seen: 416,160,000
- Positive positions: 106,088
- Sampled negatives: 500,000
- Total sampled positions: 606,088

Validation-fitted global temperature:

`T = 1.1`

## Uncalibrated sampled test metrics

- multiclass_ece: 0.10159759748643442
- multiclass_nll: 0.2373567819595337
- multiclass_brier: 0.20313487946987152
- acceptor_ece: 0.06053022614263218
- acceptor_nll: 0.12624812126159668
- acceptor_brier: 0.04700326547026634
- donor_ece: 0.057912050332420674
- donor_nll: 0.1111941710114479
- donor_brier: 0.0415811613202095

## Temperature-scaled sampled test metrics, T = 1.1

- multiclass_ece: 0.10384912216169903
- multiclass_nll: 0.22783315181732178
- multiclass_brier: 0.19564110040664673
- acceptor_ece: 0.06096683511752642
- acceptor_nll: 0.12063432484865189
- acceptor_brier: 0.04509793967008591
- donor_ece: 0.05898027280400718
- donor_nll: 0.10733403265476227
- donor_brier: 0.040018804371356964

## Interpretation

Global temperature scaling improves NLL and Brier score on held-out sampled test data, but slightly worsens ECE. This suggests that a single global temperature is not sufficient for splice-site calibration. Next step: class-specific calibration for acceptor and donor.