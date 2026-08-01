# Larger negative sample robustness check: flank-80 epoch-2

## Purpose

We tested whether the OpenSpliceAI-style class-wise vector-temperature calibration result remains stable when the sampled test cache is expanded from 500,000 negatives to 1,000,000 negatives.

## Cache construction

Two independent 500k-negative reservoirs were generated with different random seeds and combined into a 1M-negative cache.

Combined class counts:

```text
[non-splice, acceptor, donor] = [1000000, 53024, 53064]
total sampled positions = 1106088
effective negative weight = 416.053912
softmax(logits) vs stored probabilities max difference = 1.19e-07
probability row-sum max error = 1.79e-07
Evaluation

Method:

openspliceai_style_unweighted_vectorT_long_1Mneg_combined

Temperatures:

T_nonsplice = 0.3932362
T_acceptor = 0.35275677
T_donor = 0.36842206

Main metrics:

weighted_multiclass_ece = 0.0000122322
weighted_nll = 0.0008169837
acceptor_ece = 0.0000111581
donor_ece = 0.0000146450
acceptor_auprc = 0.9834087058
donor_auprc = 0.9897882386

Threshold metrics:

acceptor @ 0.01: precision = 0.9681, recall = 0.8684
acceptor @ 0.05: precision = 0.9880, recall = 0.6491
acceptor @ 0.10: precision = 0.9926, recall = 0.5104
acceptor @ 0.50: precision = 0.9990, recall = 0.1370

donor @ 0.01: precision = 0.9747, recall = 0.9236
donor @ 0.05: precision = 0.9901, recall = 0.7465
donor @ 0.10: precision = 0.9935, recall = 0.6182
donor @ 0.50: precision = 0.9987, recall = 0.2092
Interpretation

The larger-negative cache gives nearly identical weighted calibration results to the 500k-negative cache. Weighted ECE remains around 1e-5 and weighted NLL remains around 8e-4. AUPRC decreases modestly because PR-AUC depends on the positive-negative prevalence of the sampled evaluation set; doubling sampled negatives while keeping positives fixed makes the ranking task more stringent. Overall, the larger-negative experiment supports the stability of the main calibration result.
