## 2026-06-11: OpenSpliceAI flank-80 debug training

Completed final HDF5 dataset generation for flank size 80.

Final dataset files:
- data/processed_h5/dataset_train.h5
- data/processed_h5/dataset_validation.h5
- data/processed_h5/dataset_test.h5

Inspected HDF5 format:
- X datasets: (N, 15000, 4), int8
- Y datasets: (1, N, 5000, 3), int8
- Labels use 3 classes: non-splice, acceptor, donor

Created tiny debug HDF5 files:
- data/debug_h5/tiny_train.h5
- data/debug_h5/tiny_validation.h5

First 16-segment tiny run failed because OpenSpliceAI dataloader produced 0 iterations per shard.

Recreated tiny files with 128 segments per group:
- train: 4 groups × 128 segments
- validation/test: 2 groups × 128 segments

Ran first successful OpenSpliceAI debug training:

openspliceai train \
  --epochs 1 \
  --scheduler MultiStepLR \
  --output-dir results/debug_train_flank80/ \
  --project-name flank80_tiny_debug \
  --exp-num 0 \
  --flanking-size 80 \
  --random-seed 10 \
  --train-dataset data/debug_h5/tiny_train.h5 \
  --test-dataset data/debug_h5/tiny_validation.h5 \
  --loss cross_entropy_loss

Result:
- Device: CPU
- Training completed successfully
- Validation completed successfully
- Testing completed successfully
- Best model saved
- Runtime: ~31.6 seconds
## 2026-06-12: Full 124-shard flank-80 focal-loss training succeeded

Applied low-memory patch to OpenSpliceAI train_base/utils.py:
- skipped full train prediction/label accumulation
- saved model immediately after train epoch as model_0_after_train.pt
- skipped duplicate test pass by using validation loss as test loss during training

Ran full 1-epoch flank-80 focal-loss training:

openspliceai train \
  --epochs 1 \
  --scheduler MultiStepLR \
  --output-dir results/full_train_flank80_focal_1epoch_lowmem/ \
  --project-name flank80_full_focal_1epoch_lowmem \
  --exp-num 0 \
  --flanking-size 80 \
  --random-seed 10 \
  --train-dataset data/processed_h5/dataset_train.h5 \
  --test-dataset data/processed_h5/dataset_validation.h5 \
  --loss focal_loss

Result:
- Full train shards completed: 124/124
- Validation shards completed: 14/14
- Runtime: 7159.21 seconds (~1h 59m)
- Saved:
  - model_0_after_train.pt
  - model_0.pt
  - model_best.pt

Validation metrics:
- acceptor top-k: 0.2439335887611438
- donor top-k: 0.347169811320711
- acceptor AUPRC: 0.18370371712162956
- donor AUPRC: 0.30418797492939453

Hard classification:
- acceptor precision: 0.4836
- acceptor recall: 0.0754
- acceptor F1: 0.1304
- donor precision: 0.5161
- donor recall: 0.1610
- donor F1: 0.2454

Conclusion:
Full flank-80 focal-loss training is now feasible on CPU with the low-memory patch.

## 2026-06-12: Full flank-80 epoch 2 continuation succeeded

Added resume/checkpoint-loading patch to:
.venv_wsl/lib/python3.12/site-packages/openspliceai/train/train.py

Patch behavior:
- If --model points to a .pt file, load it using torch.load(..., map_location=device)
- Load weights with model.load_state_dict(...)
- Continue training from previous checkpoint
- This resumes model weights only, not optimizer/scheduler state

Continued training from epoch-1 model:
results/full_train_flank80_focal_1epoch_lowmem/SpliceAI_flank80_full_focal_1epoch_lowmem_80_0_rs10/0/models/model_best.pt

Epoch 2 command used:
openspliceai train \
  --epochs 1 \
  --scheduler MultiStepLR \
  --output-dir results/full_train_flank80_focal_epoch2_lowmem/ \
  --project-name flank80_full_focal_epoch2_lowmem \
  --exp-num 0 \
  --flanking-size 80 \
  --random-seed 11 \
  --train-dataset data/processed_h5/dataset_train.h5 \
  --test-dataset data/processed_h5/dataset_validation.h5 \
  --loss focal_loss \
  --model results/full_train_flank80_focal_1epoch_lowmem/SpliceAI_flank80_full_focal_1epoch_lowmem_80_0_rs10/0/models/model_best.pt

Runtime:
- 7230.69 seconds (~2h 0m)

Saved:
- model_0_after_train.pt
- model_0.pt
- model_best.pt

Validation metrics after epoch 2:
- acceptor top-k: 0.4018445322792619
- donor top-k: 0.4745989304812199
- acceptor AUPRC: 0.3729463235137322
- donor AUPRC: 0.4623805480003607

Hard classification:
- acceptor precision: 0.8315
- acceptor recall: 0.0975
- acceptor F1: 0.1745
- donor precision: 0.7435
- donor recall: 0.1898
- donor F1: 0.3024

Comparison to epoch 1:
- acceptor AUPRC improved from 0.1837 to 0.3729
- donor AUPRC improved from 0.3042 to 0.4624
- acceptor top-k improved from 0.2439 to 0.4018
- donor top-k improved from 0.3472 to 0.4746

Conclusion:
Resume patch worked. Full 124-shard CPU training can now be continued epoch-by-epoch using model_best.pt.

## 2026-06-12: Full flank-80 epoch 3 continuation completed but validation decreased

Continued from epoch-2 best checkpoint and trained one more full 124-shard epoch with random seed 12.

Epoch 3 saved:
- model_0_after_train.pt
- model_0.pt
- model_best.pt

Epoch 3 validation metrics:
- acceptor top-k: 0.39805825242712
- donor top-k: 0.4328593996839758
- acceptor AUPRC: 0.34354087551572676
- donor AUPRC: 0.4078568032175007

Comparison:
- Epoch 2 acceptor AUPRC: 0.3729463235137322
- Epoch 3 acceptor AUPRC: 0.34354087551572676
- Epoch 2 donor AUPRC: 0.4623805480003607
- Epoch 3 donor AUPRC: 0.4078568032175007

Conclusion:
Epoch 3 completed successfully, but validation metrics decreased. Current best flank-80 model is epoch 2:
results/full_train_flank80_focal_epoch2_lowmem/SpliceAI_flank80_full_focal_epoch2_lowmem_80_0_rs11/0/models/model_best.pt

## 2026-06-12: Chunked held-out test evaluation completed

Full test evaluation on dataset_test.h5 failed after processing all 56 shards but before metric computation, likely because valid_epoch accumulates all predictions/labels in RAM.

Workaround:
Evaluated test set in 4 chunks:
- chunk_00_13
- chunk_14_27
- chunk_28_41
- chunk_42_55

Chunk metrics:

chunk_00_13:
- acceptor top-k: 0.3881499395404609
- donor top-k: 0.4424460431654145
- acceptor AUPRC: 0.34618372853698753
- donor AUPRC: 0.4135965843790328
- acceptor F1: 0.13186813186813187
- donor F1: 0.22606924643584522

chunk_14_27:
- acceptor top-k: 0.32642487046626484
- donor top-k: 0.38568935427567436
- acceptor AUPRC: 0.29070865055532086
- donor AUPRC: 0.3576360582077262
- acceptor F1: 0.14618973561430793
- donor F1: 0.22058823529411764

chunk_28_41:
- acceptor top-k: 0.2916666666666059
- donor top-k: 0.3304535637148314
- acceptor AUPRC: 0.2440122944148951
- donor AUPRC: 0.286785723236821
- acceptor F1: 0.12686567164179105
- donor F1: 0.18772563176895307

chunk_42_55:
- acceptor top-k: 0.37165354330702804
- donor top-k: 0.43122102009266905
- acceptor AUPRC: 0.32217469690918815
- donor AUPRC: 0.40176104662217454
- acceptor F1: 0.13900709219858157
- donor F1: 0.2230971128608924

Approximate simple average across chunks:
- acceptor top-k: ~0.3445
- donor top-k: ~0.3975
- acceptor AUPRC: ~0.3008
- donor AUPRC: ~0.3650
- acceptor F1: ~0.1360
- donor F1: ~0.2144

Important caveat:
These are chunk-wise estimates. AUPRC/top-k are ranking metrics, so final paper-quality test metrics should use a memory-safe global evaluator.

## 2026-06-13: Sampled temperature scaling on validation

Full validation temperature scaling was too heavy because validation has ~90.9M nucleotide positions. Switched to sampled calibration:
- all positive splice positions
- 500,000 sampled non-splice positions

Script:
scripts/fit_temperature_validation_sampled.py

Result:
- positive positions: 25,449
- sampled negative positions: 500,000
- best temperature: 1.1

Uncalibrated sampled metrics:
- multiclass ECE: 0.030012634115184438
- multiclass NLL: 0.06873532384634018
- multiclass Brier: 0.08215265721082687
- acceptor ECE: 0.017561487041769207
- acceptor NLL: 0.03653913363814354
- acceptor Brier: 0.012926815077662468
- donor ECE: 0.017266199027220826
- donor NLL: 0.03221934288740158
- donor Brier: 0.011280752718448639

Calibrated sampled metrics with T=1.1:
- multiclass ECE: 0.032134682365449134
- multiclass NLL: 0.06842301926175325
- multiclass Brier: 0.07980645036806669
- acceptor ECE: 0.018681691504121016
- acceptor NLL: 0.036078258630557546
- acceptor Brier: 0.012449095467144235
- donor ECE: 0.018484819224614268
- donor NLL: 0.03237587290092694
- donor Brier: 0.010906966684273111

Interpretation:
Global temperature scaling with T=1.1 slightly improves NLL and Brier score, but worsens ECE. The model is only mildly overconfident under this sampled calibration setup, and global temperature scaling is not sufficient to fully calibrate acceptor/donor probabilities.
