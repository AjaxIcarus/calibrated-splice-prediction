# TCBB V9 reproducibility closure

This directory archives reproducibility artifacts associated with the
corrected V9 analysis for:

**Prior-aware calibration of OpenSpliceAI-style splice-site prediction under
genome-level class imbalance**

## Scientific Git freeze

The exact Git repository state frozen for the V9 submission-preparation scientific authority is:

- commit: `a959efa517ff11a59b8f13e80ff880242a9441b2`
- tag: `tcbb-v9-scientific-freeze-2026-08-31`

That tag is the immutable reference for the repository state that existed at
the scientific freeze.

## Why this directory was added after the freeze

Several corrected-V9 analysis scripts and provenance records were created in
the external V9 work area during the final audit/revision cycle and were
identified by explicit SHA-256 hashes in the V9 provenance records.

They were not tracked as files in commit `a959efa517ff11a59b8f13e80ff880242a9441b2`.

This post-freeze repository closure archives byte-identical copies of those
already-existing artifacts. It does **not** claim that they were present in
the frozen commit, and it does not revise the historical Git record.

No model was retrained, no calibration was refit, no bootstrap analysis was
rerun, and no quantitative result was changed as part of this archival step.

## Corrected V9 model authority

The corrected quantitative V9 analysis uses:

- seed 11, epoch 12:
  `results/best_models/flank400_focal_epoch12_best.pt`
- seed 23, epoch 13:
  `results/best_models/flank400_seed23_focal_epoch13_best.pt`

The historical seed-11/epoch-11 and seed-23/epoch-14 material remains in the
repository as development/history and is not the current V9 quantitative
authority.

## Archived exact work scripts

`pinned_work_scripts/` contains byte-identical archival copies of the six
externally pinned corrected-V9 work scripts.

These files are preserved exactly as executed/audited. Machine-local paths or
historical workspace assumptions inside them have intentionally not been
rewritten, because doing so would change their audited identities.

## Archived provenance records

`provenance/` contains byte-identical copies of the V9 script provenance,
environment provenance, and authoritative-artifact manifest.

## Integrity

`TCBB_V9_REPOSITORY_CLOSURE_SHA256SUMS.txt` records the SHA-256 identities of
the archived work scripts, provenance records, and corrected checkpoints.

This directory is an archival/reproducibility closure layer. It does not
supersede the scientific-freeze tag.
