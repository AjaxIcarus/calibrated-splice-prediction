# Flank-400 focal model: epoch 7

Model:
`results/best_models/flank400_focal_epoch7_best.pt`

Initialized from:
`results/best_models/flank400_focal_epoch6_best.pt`

Training setup:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation dataset: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~15866 seconds

Validation summary:
- Acceptor line: 0.9868 0.8274 0.9394 0.9776 0.8985 0.6900 0.3994 0.1985 0.1132 759
- Donor line: 0.9652 0.8209 0.9572 0.9920 0.8995 0.7119 0.4574 0.2338 0.1268 748
- Overall accuracy: 0.844394
- Non-splice F1: 0.999951
- Acceptor precision: 0.922951
- Acceptor recall: 0.741765
- Acceptor F1: 0.822498
- Donor precision: 0.864234
- Donor recall: 0.791444
- Donor F1: 0.826239
- Training loss: 0.000081232
- Validation loss: 0.000082190

Interpretation:
Epoch 7 improved over epoch 6 in validation loss, acceptor F1, donor recall, and donor F1. Acceptor recall decreased slightly, but precision increased enough that acceptor F1 improved. Epoch 7 is currently the best flank-400 checkpoint.
