# Flank-400 focal model: epoch 8

Model:
`results/best_models/flank400_focal_epoch8_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch7_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~15970 seconds

Validation summary:
- Acceptor line: 0.9815 0.8221 0.9460 0.9816 0.9009 0.7111 0.4107 0.1987 0.1113 759
- Donor line: 0.9679 0.8262 0.9626 0.9933 0.9049 0.7373 0.4689 0.2269 0.1209 748
- Overall accuracy: 0.856766
- Non-splice F1: 0.999950
- Acceptor precision: 0.904688
- Acceptor recall: 0.762846
- Acceptor F1: 0.827734
- Donor precision: 0.844755
- Donor recall: 0.807487
- Donor F1: 0.825701
- Training loss: 0.000076411
- Validation loss: 0.000079836

Interpretation:
Epoch 8 improved validation loss and acceptor F1 relative to epoch 7. Donor recall improved, but donor F1 was essentially flat/slightly lower because donor precision decreased. The model is still improving by validation loss, but training is beginning to plateau.
