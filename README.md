# Prior-aware calibration of OpenSpliceAI-style splice-site prediction

Research repository for **Prior-aware calibration of OpenSpliceAI-style splice-site prediction under genome-level class imbalance**.

This project studies probability calibration for position-level splice-site prediction under severe class imbalance (non-splice, acceptor, donor), while treating detection/ranking and calibration as distinct questions.

## V9 scientific authority

Exact frozen Git state:

- commit `a959efa517ff11a59b8f13e80ff880242a9441b2`
- tag `tcbb-v9-scientific-freeze-2026-08-31`

Later commits on `main` do not replace that frozen reference.

The first post-freeze archival closure is commit `b342abf6128c4a16e7eabe99c0521f64cfb8c33b`. See `reproducibility/tcbb_v9/README.md` for the archival boundary and exact artifact identities.

## Corrected V9 models

Current quantitative authority uses:

- seed 11, epoch 12
- seed 23, epoch 13

Archived checkpoints:

- `results/best_models/flank400_focal_epoch12_best.pt`
- `results/best_models/flank400_seed23_focal_epoch13_best.pt`

Older epoch-11/epoch-14, flank-80, and related files are retained as historical development material.

## Reproducibility and data

Corrected-V9 scripts, provenance, checksums, and closure records are under `reproducibility/tcbb_v9/`.

Verify archived payloads with:

`sha256sum -c reproducibility/tcbb_v9/TCBB_V9_REPOSITORY_CLOSURE_SHA256SUMS.txt`

The evaluation workflow uses the GRCh38 primary assembly and MANE v1.3 RefSeq genomic annotations. Large HDF5 datasets, logit caches, genome files, and machine-local intermediates are generally kept outside ordinary Git tracking.

The two corrected model runs provide descriptive robustness evidence, not independent dataset replications or a population-level estimate of training-seed uncertainty.

This canonical research repository intentionally preserves historical material for traceability.
