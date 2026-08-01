# Flank-400 focal model: epoch 3

Model:
`results/best_models/flank400_focal_epoch3_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch2_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~15822 seconds

Validation summary:
- Acceptor line: 0.9129 0.7075 0.8603 0.9513 0.7767 0.5237 0.3387 0.2173 0.1385 759
- Donor line: 0.9037 0.7313 0.8971 0.9586 0.7940 0.5393 0.3563 0.2192 0.1339 748
- Overall accuracy: 0.664717
- Non-splice F1: 0.999915
- Acceptor precision: 0.911330
- Acceptor recall: 0.487484
- Acceptor F1: 0.635193
- Donor precision: 0.881395
- Donor recall: 0.506684
- Donor F1: 0.643463
- Training loss: 0.000144776
- Validation loss: 0.000138662

Interpretation:
Epoch 3 improved over epoch 2 in validation loss, acceptor F1, donor F1, and precision. It is currently the best flank-400 checkpoint.
