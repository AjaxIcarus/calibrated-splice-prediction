# Flank-400 focal model: epoch 6

Model:
`results/best_models/flank400_focal_epoch6_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch5_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~16521 seconds

Validation summary:
- Acceptor line: 0.9763 0.8103 0.9315 0.9776 0.8875 0.7088 0.4318 0.2243 0.1325 759
- Donor line: 0.9626 0.8128 0.9519 0.9866 0.8886 0.6998 0.4570 0.2446 0.1376 748
- Overall accuracy: 0.840342
- Non-splice F1: 0.999947
- Acceptor precision: 0.882171
- Acceptor recall: 0.749671
- Acceptor F1: 0.810541
- Donor precision: 0.858631
- Donor recall: 0.771390
- Donor F1: 0.812676
- Training loss: 0.000088476
- Validation loss: 0.000086045

Interpretation:
Epoch 6 improved over epoch 5 in validation loss, acceptor recall/F1, and donor recall/F1. Precision decreased somewhat, but F1 and loss improved, so epoch 6 is currently the best flank-400 checkpoint.
