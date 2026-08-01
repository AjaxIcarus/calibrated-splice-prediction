# chr9 held-out transcript-position evaluation: flank-80 epoch-2

## Purpose

Evaluate calibration on all chr9 gene/transcript positions present in the OpenSpliceAI test H5, instead of using a sampled-negative cache.

## Scope

This is not full intergenic whole-chromosome inference. It is streaming evaluation over all chr9 gene/transcript segments in `dataset_test.h5`.

## Dry run / coverage

```text
Chromosome: chr9
Genes: 748
Segments: 10769
Approx positions before clipping: 53845000
Final evaluated positions: 52880000
Class counts [non, acceptor, donor]: [52865507, 7243, 7250]
Shards touched: [47, 48, 49, 50, 51, 52, 53, 54, 55]
Main metrics
Uncalibrated:
  multiclass_ece = 0.0056889447
  multiclass_nll = 0.0063592399
  acceptor_ece = 0.0027592989
  donor_ece = 0.0029330155

OpenSpliceAI-style vector T:
  multiclass_ece = 0.0000065416
  multiclass_nll = 0.0008498552
  acceptor_ece = 0.0000075313
  donor_ece = 0.0000102451
Threshold behavior
OpenSpliceAI-style vector T, acceptor:
  @0.01 precision = 0.0690, recall = 0.8709
  @0.05 precision = 0.1668, recall = 0.6569
  @0.10 precision = 0.2462, recall = 0.5201
  @0.50 precision = 0.6082, recall = 0.1455

OpenSpliceAI-style vector T, donor:
  @0.01 precision = 0.0878, recall = 0.9381
  @0.05 precision = 0.2016, recall = 0.7794
  @0.10 precision = 0.2915, recall = 0.6625
  @0.50 precision = 0.6606, recall = 0.2537
Interpretation

The chr9 streaming evaluation supports the main calibration claim. OpenSpliceAI-style vector-temperature calibration reduced multiclass ECE from 0.005689 to 0.00000654 and NLL from 0.006359 to 0.000850 on all chr9 transcript/gene positions. This confirms that the strong calibration result is not only an artifact of the sampled-negative cache.

Threshold precision is much lower than in the sampled-cache evaluation because chr9 contains an extremely small splice-site fraction: 14,493 splice sites among 52,880,000 evaluated positions. This reinforces the paper's claim that splice-site detection, calibrated probability estimation, and threshold-based deployment are distinct under genome-level class imbalance.
