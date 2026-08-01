# Flank-400 focal model: epoch 1

Model:
`results/best_models/flank400_focal_epoch1_best.pt`

Training command:
- OpenSpliceAI train
- flanking size: 400
- loss: focal_loss
- random seed: 11
- train dataset: `data/processed_h5_flank400/dataset_train.h5`
- validation/test dataset during training: `data/processed_h5_flank400/dataset_validation.h5`

Training time:
~15613 seconds

Validation summary from training log:
- Acceptor top-k line: 0.4354 0.3267 0.4651 0.6074 0.2776 0.4015 0.3446 0.2840 0.2312 759
- Donor top-k line: 0.5241 0.3984 0.5668 0.7246 0.3577 0.4369 0.3628 0.2930 0.2304 748
- Overall accuracy: 0.416233
- Non-splice: precision 0.999736, recall 0.999978, F1 0.999857
- Acceptor: precision 0.6875, recall 0.08696, F1 0.15439
- Donor: precision 0.6050, recall 0.16176, F1 0.25527
- Training loss: 0.001146
- Validation loss: 0.000307966

Interpretation:
Flank-400 epoch 1 completed successfully and saved a best checkpoint. The model has learned splice-site signal, but recall is still low at default argmax/threshold behavior. Next step is to evaluate validation/test ranking metrics and then continue to epoch 2.
