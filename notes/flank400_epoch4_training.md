# Flank-400 focal model: epoch 4

Model:
`results/best_models/flank400_focal_epoch4_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch3_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~16143 seconds

Validation summary:
- Acceptor line: 0.9551 0.7615 0.9157 0.9618 0.8407 0.5868 0.3544 0.2071 0.1277 759
- Donor line: 0.9385 0.7861 0.9198 0.9840 0.8472 0.5963 0.3728 0.2129 0.1269 748
- Overall accuracy: 0.736373
- Non-splice F1: 0.999931
- Acceptor precision: 0.911469
- Acceptor recall: 0.596838
- Acceptor F1: 0.721338
- Donor precision: 0.905138
- Donor recall: 0.612299
- Donor F1: 0.730463
- Training loss: 0.000114948
- Validation loss: 0.000112304

Interpretation:
Epoch 4 improved substantially over epoch 3 in validation loss, acceptor recall/F1, and donor recall/F1. It is currently the best flank-400 checkpoint.
