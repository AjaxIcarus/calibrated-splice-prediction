# Flank-400 focal model: epoch 2

Model:
`results/best_models/flank400_focal_epoch2_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch1_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~15533 seconds

Validation summary:
- Acceptor line: 0.7916 0.6034 0.7747 0.9012 0.6502 0.5436 0.4169 0.3105 0.2178 759
- Donor line: 0.8422 0.6564 0.8436 0.9318 0.7133 0.5546 0.4112 0.2889 0.1932 748
- Overall accuracy: 0.654128
- Non-splice F1: 0.999898
- Acceptor precision: 0.747346
- Acceptor recall: 0.463768
- Acceptor F1: 0.572358
- Donor precision: 0.759674
- Donor recall: 0.498663
- Donor F1: 0.602098
- Training loss: 0.000202515
- Validation loss: 0.000163655

Interpretation:
Flank-400 epoch 2 is substantially stronger than epoch 1 and is the current main checkpoint for the flank-400 extension.
