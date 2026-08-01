# Results Summary: flank80 focal-loss epoch-2 model

## Model

Checkpoint:

`results/best_models/flank80_focal_epoch2_best.pt`

Dataset:

- `data/processed_h5/dataset_validation.h5`
- `data/processed_h5/dataset_test.h5`

Prediction task:

Per-position 3-class splice-site prediction:

- non-splice
- acceptor
- donor

## Sample sizes

Validation sampled cache:

- total sampled positions: 525,449

Test sampled cache:

- total sampled positions: 606,088
- sampled negatives: 500,000
- positives: 106,088
  - acceptors: 53,024
  - donors: 53,064

Original test distribution:

- total positions seen: 416,160,000
- positive positions: 106,088
- negative positions: 416,053,912

Therefore, the sampled test set is splice-enriched relative to the true genome-position distribution.

## Calibration experiments

### 1. Global scalar temperature

Global temperature:

`T = 1.1`

On sampled test:

- improved NLL/Brier slightly
- did not improve ECE
- preserved argmax predictions exactly

### 2. Unweighted vector temperature

Fitted on sampled validation distribution:

- `T_nonsplice = 0.0591`
- `T_acceptor = 0.8414`
- `T_donor = 0.7686`

On sampled test:

- strongly improved unweighted ECE/NLL
- changed argmax predictions substantially
- improved apparent splice recall because sampled validation/test were splice-enriched

Argmax change from uncalibrated:

- 83,242 / 606,088 positions changed
- most changes were true acceptor/donor positions previously classified as non-splice

### 3. Genome-weighted vector temperature

Fitted with negative positions weighted according to genome prior:

- validation negative weight: 181.749102
- test negative weight: 832.107824

Fitted temperatures:

- `T_nonsplice = 3.9132`
- `T_acceptor = 0.5750`
- `T_donor = 0.5771`

On weighted test:

- weighted multiclass ECE:
  - uncalibrated: 0.00561
  - global T: 0.00854
  - unweighted vector T: 0.00157
  - weighted vector T: 0.00034

- weighted multiclass NLL:
  - uncalibrated: 0.00624
  - global T: 0.00921
  - unweighted vector T: 0.00917
  - weighted vector T: 0.00105

Interpretation:

Weighted vector calibration gives the best genome-prior probability calibration, but argmax predictions become almost entirely non-splice. This is expected under extreme class imbalance and shows that argmax is not an appropriate splice-site detection rule.

## Detection / ranking metrics on sampled test

AUPRC:

| Method | Acceptor AUPRC | Donor AUPRC |
|---|---:|---:|
| Uncalibrated | 0.99091 | 0.99497 |
| Global T=1.1 | 0.99091 | 0.99497 |
| Unweighted vector T | 0.99084 | 0.99493 |
| Weighted vector T | 0.99091 | 0.99497 |

Top-k retrieval:

Uncalibrated acceptor top 1x positives:

- precision: 0.9608
- recall: 0.9608

Uncalibrated donor top 1x positives:

- precision: 0.9747
- recall: 0.9747

Weighted vector acceptor top 1x positives:

- precision: 0.9608
- recall: 0.9608

Weighted vector donor top 1x positives:

- precision: 0.9747
- recall: 0.9747

Interpretation:

Temperature scaling barely changes ranking. The model is already strong at retrieval; calibration mainly changes probability scale.

## Main empirical claim

In extremely imbalanced per-nucleotide splice-site prediction, calibration and detection must be evaluated separately.

- Ranking-based splice-site retrieval is strong and stable across calibration methods.
- Probability calibration depends strongly on the assumed class prior.
- Genome-weighted calibration gives the best calibrated posterior under the natural genome-wide class distribution.
- Argmax classification is not suitable for splice-site discovery because the calibrated posterior is dominated by the non-splice prior.
