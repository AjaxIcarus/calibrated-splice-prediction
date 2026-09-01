> **ARCHIVED / SUPERSEDED RESEARCH DRAFT — NOT CURRENT TCBB SUBMISSION AUTHORITY**
>
> This file is preserved unchanged below for development traceability.
> It may contain historical flank-80 results, earlier flank-400 checkpoints,
> earlier population definitions, or manuscript wording that was superseded
> during reproducibility closure and TCBB revision.
>
> **Do not use this file as current quantitative or submission authority.**
> Current repository quantitative authority is seed 11/epoch 12 and
> seed 23/epoch 13. See the root `README.md`,
> `CURRENT_SUBMISSION_AUTHORITY.md`, and
> `reproducibility/tcbb_v9/README.md`.

# Supplementary Materials

## Supplementary Note S1: True-logit extraction

The final calibration analysis used true pre-softmax logits rather than probability-derived logit proxies. The OpenSpliceAI-style model normally returns softmax probabilities, but setting:

`model.apply_softmax = False`

returns the final logits before softmax. We verified that applying softmax to these logits reconstructed the model’s probability outputs with maximum absolute difference approximately `1.19e-07` for both validation and test caches.

The resulting caches were:

- `results/logit_cache_flank80_epoch2/validation_sampled_logits.npz`
- `results/logit_cache_flank80_epoch2/test_sampled_logits.npz`

Each cache contains sampled logits, probabilities, and one-hot labels.

---

## Supplementary Note S2: Sampled-cache construction

The validation and test caches retained all positive splice-site positions and reservoir-sampled 500,000 non-splice positions. This made repeated calibration, bootstrap analysis, and threshold evaluation computationally feasible.

The sampled test cache contained:

| Class | Count |
|---|---:|
| Non-splice | 500,000 |
| Acceptor | 53,024 |
| Donor | 53,064 |
| Total | 606,088 |

The full test set contained 416,160,000 positions, including 106,088 splice-site positives and 416,053,912 non-splice positions.

---

## Supplementary Note S3: Genome-prior weighting

Because the sampled cache was splice-enriched, genome-prior calibration used non-splice sample weights to approximate the full genome-position distribution.

The test negative weight was:

`416,053,912 / 500,000 = 832.107824`

The validation negative weight used for fitting genome-prior weighted vector temperature scaling was:

`181.749102`

---

## Supplementary Note S4: Bootstrap confidence intervals

Bootstrap confidence intervals were estimated using 200 nonparametric bootstrap replicates over the sampled test cache with random seed 17. For each replicate, sampled positions were resampled with replacement, and calibration and detection metrics were recomputed.

Files:

- `results/logit_bootstrap_flank80_epoch2/point_estimates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_replicates.csv`
- `results/logit_bootstrap_flank80_epoch2/bootstrap_summary.csv`
- `tables/logit_based/logit_bootstrap_summary.md`

Main result:

Genome-prior weighted vector scaling achieved weighted multiclass ECE of 0.000049 [0.000039, 0.000058] and weighted multiclass NLL of 0.000818 [0.000804, 0.000832].

---

## Supplementary Note S5: Prior-sensitivity analysis

Prior-sensitivity analysis refit vector temperature scaling while varying the validation non-splice weight across:

`1, 2, 5, 10, 25, 50, 100, 181.749`

For each setting, we recorded learned temperature parameters, genome-prior ECE, genome-prior NLL, and argmax prediction counts on the test cache.

Files:

- `results/logit_prior_sensitivity_flank80_epoch2/prior_sensitivity_summary.csv`
- `tables/logit_based/logit_prior_sensitivity_summary.md`
- `figures/logit_based/logit_prior_sensitivity_temperatures.png`
- `figures/logit_based/logit_prior_sensitivity_genome_metrics.png`
- `figures/logit_based/logit_prior_sensitivity_argmax_counts.png`

Main result:

As the non-splice weight increased toward the genome-wide prior, genome-prior calibration improved and argmax splice predictions decreased. This supports the conclusion that temperature scaling is prior-sensitive under extreme class imbalance.

---

## Supplementary Note S6: Reliability diagrams

Reliability diagrams were generated for multiclass confidence and for class-specific acceptor and donor probabilities.

Files:

- `figures/logit_based/reliability_multiclass_uncalibrated.png`
- `figures/logit_based/reliability_multiclass_logit_weighted_vector.png`
- `figures/logit_based/reliability_acceptor_uncalibrated_vs_weighted.png`
- `figures/logit_based/reliability_donor_uncalibrated_vs_weighted.png`
- `tables/logit_based/logit_reliability_bins.csv`

Main result:

The uncalibrated model was overconfident under genome-prior evaluation, especially in low-probability splice-site bins. Genome-prior weighted vector scaling improved agreement between predicted probability and observed frequency.

---

## Supplementary Note S7: Implementation details

Final scripts used for the true-logit calibration analysis:

- `scripts/extract_logits_cache.py`
- `scripts/vector_temperature_from_logits_cache.py`
- `scripts/detection_metrics_from_logits_cache.py`
- `scripts/make_logit_results_tables_and_figures.py`
- `scripts/bootstrap_logit_metrics.py`
- `scripts/make_logit_reliability_diagrams.py`
- `scripts/prior_sensitivity_logit_calibration.py`

Final main manuscript draft:

- `paper_draft/full_draft_v10.md`

Final bibliography starter:

- `paper_draft/references.bib`
