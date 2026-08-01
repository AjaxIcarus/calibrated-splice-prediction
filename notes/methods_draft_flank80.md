# Methods Draft: Flank-80 OpenSpliceAI Calibration Study

## Dataset construction

We constructed a human splice-site prediction dataset using GRCh38 primary assembly sequence and MANE v1.3 RefSeq genomic annotations. The prediction task was formulated as per-nucleotide three-class classification, where each genomic position was labeled as one of three classes: non-splice, acceptor, or donor.

Datasets were generated using the OpenSpliceAI `create-data` pipeline with protein-coding genes, canonical transcript parsing, chromosome-based human train/test splitting, and a flanking sequence size of 80 nucleotides. The resulting HDF5 datasets were:

- `dataset_train.h5`
- `dataset_validation.h5`
- `dataset_test.h5`

The held-out test set contained 416,160,000 evaluated nucleotide positions. Among these, 106,088 were annotated splice-site positives, consisting of 53,024 acceptor positions and 53,064 donor positions. The remaining 416,053,912 positions were non-splice positions.

## Model training

We trained an OpenSpliceAI-style splice-site prediction model using a flank size of 80 nucleotides. The model was trained with focal loss to address the extreme imbalance between non-splice and splice-site positions. Training was performed on CPU using a low-memory OpenSpliceAI patch. The best checkpoint was selected from the epoch-2 model:

`results/best_models/flank80_focal_epoch2_best.pt`

This checkpoint was used for all calibration and detection analyses.

## Sampled evaluation set

Because the full test set contains hundreds of millions of positions, we used a sampled evaluation strategy. All splice-site positives were retained, and non-splice positions were reservoir-sampled.

For the sampled test set:

- total sampled positions: 606,088
- acceptor positives: 53,024
- donor positives: 53,064
- total positives: 106,088
- sampled non-splice negatives: 500,000

This sampled set is intentionally enriched for splice sites relative to the true genome-wide distribution. Therefore, we evaluated both unweighted sampled metrics and genome-weighted metrics.

## Calibration methods

We evaluated three post-hoc calibration approaches.

### Global scalar temperature scaling

First, we applied a single scalar temperature to all three classes. The scalar temperature was fitted on validation data and applied to the test set. The selected scalar temperature was:

`T = 1.1`

This method preserves argmax predictions because all class logits are scaled equally.

### Unweighted vector temperature scaling

Second, we fit a vector temperature model with one temperature per class:

`T = [T_nonsplice, T_acceptor, T_donor]`

The unweighted vector temperature was optimized on the sampled validation distribution by minimizing multiclass negative log-likelihood. This produced:

- `T_nonsplice = 0.0591`
- `T_acceptor = 0.8414`
- `T_donor = 0.7686`

Because the sampled validation set was splice-enriched, this calibration reflected the sampled class prior rather than the natural genome-wide prior.

### Genome-weighted vector temperature scaling

Third, we fit a genome-weighted vector temperature model. In this setting, each sampled non-splice position was weighted according to the number of original non-splice positions it represented. Positive splice-site positions retained weight 1.

For the test set, the non-splice weight was:

`416,053,912 / 500,000 = 832.107824`

For the validation set, the corresponding non-splice weight was:

`181.749102`

The genome-weighted vector temperature model produced:

- `T_nonsplice = 3.9132`
- `T_acceptor = 0.5750`
- `T_donor = 0.5771`

This calibration estimates probabilities under the natural genome-wide class prior.

## Calibration metrics

Calibration was evaluated using:

- multiclass expected calibration error
- multiclass negative log-likelihood
- multiclass Brier score
- binary acceptor ECE, NLL, and Brier score
- binary donor ECE, NLL, and Brier score

Expected calibration error was computed by binning predictions by confidence and comparing average predicted confidence to empirical accuracy or empirical positive frequency within each bin.

For genome-weighted calibration metrics, each sampled non-splice position was weighted according to the number of original non-splice positions represented by that sample.

## Detection and ranking metrics

Detection was evaluated separately from calibration. For acceptor and donor classes, we computed:

- area under the precision-recall curve
- top-k retrieval precision and recall
- thresholded precision and recall

Top-k retrieval was evaluated at multiples of the number of true positives: 1x, 2x, 5x, and 10x positives.

This separation is important because calibrated posterior probabilities and splice-site discovery are different objectives. Under the natural genome-wide prior, a calibrated argmax classifier is expected to predict nearly all random positions as non-splice. Therefore, argmax accuracy is not an appropriate primary metric for splice-site discovery.

## Reproducibility

Prediction caches were saved for validation and test samples so that calibration methods could be compared without rerunning model inference. The primary cached files were:

- `results/classwise_temperature_flank80_epoch2/validation_sampled_predictions.npz`
- `results/classwise_temperature_flank80_epoch2/test_sampled_predictions.npz`

Primary result summaries were saved in:

- `results/weighted_vector_temperature_flank80_epoch2/weighted_vector_temperature_summary.csv`
- `results/detection_metrics_flank80_epoch2/detection_auprc_summary.csv`
- `results/detection_metrics_flank80_epoch2/detection_threshold_topk_summary.csv`

Figures were generated in:

- `figures/flank80_epoch2_multiclass_ece.png`
- `figures/flank80_epoch2_multiclass_nll.png`
- `figures/flank80_epoch2_detection_auprc.png`
- `figures/flank80_epoch2_acceptor_threshold_precision_recall.png`
- `figures/flank80_epoch2_donor_threshold_precision_recall.png`