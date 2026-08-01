# Flank-400 focal model: epoch 5

Model:
`results/best_models/flank400_focal_epoch5_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch4_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~15975 seconds

Validation summary:
- Acceptor line: 0.9736 0.7997 0.9262 0.9776 0.8707 0.6283 0.3753 0.2058 0.1230 759
- Donor line: 0.9465 0.8061 0.9412 0.9893 0.8746 0.6531 0.4084 0.2276 0.1306 748
- Overall accuracy: 0.790810
- Non-splice F1: 0.999941
- Acceptor precision: 0.915771
- Acceptor recall: 0.673254
- Acceptor F1: 0.776006
- Donor precision: 0.886441
- Donor recall: 0.699198
- Donor F1: 0.781764
- Training loss: 0.000098162
- Validation loss: 0.000095630

Interpretation:
Epoch 5 improved over epoch 4 in validation loss, acceptor recall/F1, and donor recall/F1. It is currently the best flank-400 checkpoint.
