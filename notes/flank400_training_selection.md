# Flank-400 training selection

Selected checkpoint:
`results/best_models/flank400_focal_epoch8_best.pt`

Alias:
`results/best_models/flank400_focal_best.pt`

Training setup:
- OpenSpliceAI-style model
- flanking size: 400
- loss: focal_loss
- random seed: 11
- trained incrementally from epoch 1 through epoch 9
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Epoch comparison around stopping point:

| Metric | Epoch 8 | Epoch 9 |
|---|---:|---:|
| Validation loss | 0.000079836 | 0.000081109 |
| Acceptor precision | 0.904688 | 0.933665 |
| Acceptor recall | 0.762846 | 0.741765 |
| Acceptor F1 | 0.827734 | 0.826725 |
| Donor precision | 0.844755 | 0.898734 |
| Donor recall | 0.807487 | 0.759358 |
| Donor F1 | 0.825701 | 0.823188 |

Decision:
Epoch 8 is selected because it has lower validation loss, higher acceptor F1, higher donor F1, and higher splice-site recall than epoch 9. Epoch 9 increased precision but reduced recall and slightly worsened validation loss and F1.

Interpretation:
Training plateaued around epoch 8. We stop training here and move to flank-400 logit extraction and calibration analysis.
