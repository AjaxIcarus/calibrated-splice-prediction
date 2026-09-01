> **ARCHIVED / SUPERSEDED RESEARCH DRAFT — NOT CURRENT TCBB SUBMISSION AUTHORITY**
>
> This file is retained as historical project-development material.
> Statements below using terms such as `current`, `final`, `primary`,
> `replication`, or older checkpoint identities describe the project state
> at the time this file was created; they do not define the current TCBB
> submission or corrected quantitative authority.
>
> **Current quantitative authority is seed 11/epoch 12 and seed 23/epoch 13.**
> See the repository root `README.md`,
> `CURRENT_SUBMISSION_AUTHORITY.md`, and
> `reproducibility/tcbb_v9/README.md`.
>
> The historical content below is intentionally preserved rather than
> rewritten so that project evolution remains auditable.


# Reproducibility



## Computing environment



All analyses were run in a WSL Ubuntu environment using Python 3.12.3 and OpenSpliceAI 0.0.5. The project environment was activated with:



```bash

source .venv_wsl/bin/activate



Data



The model was trained and evaluated using OpenSpliceAI-formatted HDF5 datasets derived from human GRCh38 sequence and MANE v1.3 splice annotations.



Final processed datasets:



data/processed_h5/dataset_train.h5

data/processed_h5/dataset_validation.h5

data/processed_h5/dataset_test.h5



The validation and test datasets used in the final calibration analysis were:



data/processed_h5/dataset_validation.h5

data/processed_h5/dataset_test.h5



The full test set contained 416,160,000 genomic positions, including 106,088 splice-site positives and 416,053,912 non-splice positions.



Model checkpoint



All final analyses used the flank-80 focal-loss epoch-2 checkpoint:



results/best_models/flank80_focal_epoch2_best.pt



The model performs per-position three-class classification over:



non-splice

acceptor

donor

True-logit caches



True pre-softmax logits were extracted by setting:



model.apply_softmax = False



The final sampled true-logit caches were generated with:



python scripts/extract_logits_cache.py \

  --model results/best_models/flank80_focal_epoch2_best.pt \

  --validation-dataset data/processed_h5/dataset_validation.h5 \

  --test-dataset data/processed_h5/dataset_test.h5 \

  --out-dir results/logit_cache_flank80_epoch2 \

  --flanking-size 80 \

  --max-negatives 500000 \

  --random-seed 11



Final cache files:



results/logit_cache_flank80_epoch2/validation_sampled_logits.npz

results/logit_cache_flank80_epoch2/test_sampled_logits.npz



Each cache contains sampled logits, probabilities, and one-hot labels. All positive splice-site positions were retained, and 500,000 non-splice positions were reservoir-sampled.



Temperature scaling



True-logit vector temperature scaling was run with:



python scripts/vector_temperature_from_logits_cache.py \

  --validation-cache results/logit_cache_flank80_epoch2/validation_sampled_logits.npz \

  --test-cache results/logit_cache_flank80_epoch2/test_sampled_logits.npz \

  --out-dir results/logit_vector_temperature_flank80_epoch2 \

  --global-temperature 1.1 \

  --n-bins 15 \

  --lr 0.01 \

  --steps 3000



Final genome-prior weighted vector temperatures:



T_nonsplice = 0.435169

T_acceptor = 0.416153

T_donor = 0.427941



Main output files:



results/logit_vector_temperature_flank80_epoch2/logit_vector_temperature_summary.csv

results/logit_vector_temperature_flank80_epoch2/logit_vector_argmax_summary.txt

Detection metrics



Detection metrics were computed with:



python scripts/detection_metrics_from_logits_cache.py \

  --cache results/logit_cache_flank80_epoch2/test_sampled_logits.npz \

  --out-dir results/logit_detection_metrics_flank80_epoch2 \

  --global-temperature 1.1 \

  --unweighted-t-nonsplice 0.81190228 \

  --unweighted-t-acceptor 0.24863805 \

  --unweighted-t-donor 0.34511948 \

  --weighted-t-nonsplice 0.43516919 \

  --weighted-t-acceptor 0.41615313 \

  --weighted-t-donor 0.42794070



Main output files:



results/logit_detection_metrics_flank80_epoch2/logit_detection_auprc_summary.csv

results/logit_detection_metrics_flank80_epoch2/logit_detection_threshold_topk_summary.csv

Tables and figures



Final tables and figures were generated with:



python scripts/make_logit_results_tables_and_figures.py



Main table outputs:



tables/logit_based/logit_main_results_table.md

tables/logit_based/logit_threshold_results_table.md



Main figure outputs:



figures/logit_based/logit_multiclass_ece.png

figures/logit_based/logit_multiclass_nll.png

figures/logit_based/logit_detection_auprc.png

figures/logit_based/logit_acceptor_threshold_precision_recall.png

figures/logit_based/logit_donor_threshold_precision_recall.png

Bootstrap confidence intervals



Bootstrap confidence intervals were computed with:



python scripts/bootstrap_logit_metrics.py \

  --cache results/logit_cache_flank80_epoch2/test_sampled_logits.npz \

  --out-dir results/logit_bootstrap_flank80_epoch2 \

  --n-bootstrap 200 \

  --random-seed 17 \

  --global-temperature 1.1 \

  --unweighted-t-nonsplice 0.81190228 \

  --unweighted-t-acceptor 0.24863805 \

  --unweighted-t-donor 0.34511948 \

  --weighted-t-nonsplice 0.43516919 \

  --weighted-t-acceptor 0.41615313 \

  --weighted-t-donor 0.42794070



Main output files:



results/logit_bootstrap_flank80_epoch2/point_estimates.csv

results/logit_bootstrap_flank80_epoch2/bootstrap_replicates.csv

results/logit_bootstrap_flank80_epoch2/bootstrap_summary.csv

tables/logit_based/logit_bootstrap_summary.md

Reliability diagrams



Reliability diagrams were generated with:



python scripts/make_logit_reliability_diagrams.py



Main output files:



figures/logit_based/reliability_multiclass_uncalibrated.png

figures/logit_based/reliability_multiclass_logit_weighted_vector.png

figures/logit_based/reliability_acceptor_uncalibrated_vs_weighted.png

figures/logit_based/reliability_donor_uncalibrated_vs_weighted.png

tables/logit_based/logit_reliability_bins.csv

Prior-sensitivity analysis



Prior-sensitivity analysis was run with:



python scripts/prior_sensitivity_logit_calibration.py \

  --validation-cache results/logit_cache_flank80_epoch2/validation_sampled_logits.npz \

  --test-cache results/logit_cache_flank80_epoch2/test_sampled_logits.npz \

  --out-dir results/logit_prior_sensitivity_flank80_epoch2 \

  --steps 3000 \

  --lr 0.01 \

  --n-bins 15



Main output files:



results/logit_prior_sensitivity_flank80_epoch2/prior_sensitivity_summary.csv

tables/logit_based/logit_prior_sensitivity_summary.md

figures/logit_based/logit_prior_sensitivity_temperatures.png

figures/logit_based/logit_prior_sensitivity_genome_metrics.png

figures/logit_based/logit_prior_sensitivity_argmax_counts.png

Final scripts



The final analysis depends on the following scripts:



scripts/extract_logits_cache.py

scripts/vector_temperature_from_logits_cache.py

scripts/detection_metrics_from_logits_cache.py

scripts/make_logit_results_tables_and_figures.py

scripts/bootstrap_logit_metrics.py

scripts/make_logit_reliability_diagrams.py

scripts/prior_sensitivity_logit_calibration.py

Reproducibility note



The large HDF5 datasets, model checkpoint, and .npz logit caches may be too large to store directly in a standard Git repository. The manuscript therefore tracks scripts, tables, figures, and result summaries, while large intermediate files should be stored separately or regenerated from the commands above.

