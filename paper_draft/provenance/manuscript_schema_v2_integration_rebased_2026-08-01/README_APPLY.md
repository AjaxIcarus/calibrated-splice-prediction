# Flank-400 schema-v2 manuscript integration (rebased)

This replacement package promotes the accepted valid-only, matched cross-seed results into the exact current manuscript and figure/table plan uploaded on August 1, while preserving both current files as backups. It replaces the earlier package that correctly stopped on an input-checksum mismatch.

## What changes

- Replaces the pre-schema-v2 census with 84,258 test sequences, 421,290,000 raw positions, 14,011,378 excluded padding positions, and 407,278,622 valid target positions.
- Makes seed-11/epoch-11 and seed-23/epoch-14 the matched primary and replication models.
- Adds ECE, NLL, Brier score, and target-weighted acceptor/donor AUPRC for all five methods.
- Adds the paired-bootstrap interpretation: none of ten target-weighted versus OpenSpliceAI-style intervals excludes zero.
- Installs the accepted schema-v2 reliability figures for both models.
- Revises the limitations to distinguish evaluated-position uncertainty from training-seed variability.
- Retires the pre-schema-v2 five-chromosome raw-position table, epoch-8 material, old single-chr9 material, and sampled AUPRC values from the main manuscript. These historical artifacts are not deleted.

## Apply

Extract the ZIP into the repository root. From `~/projects/calibrated-splice-prediction`, run:

```bash
bash manuscript_schema_v2_integration_rebased_2026-08-01/apply_schema_v2_manuscript_update.sh
```

The script is guarded by SHA-256 checks and refuses to overwrite an existing backup, versioned draft, figure, or generator script. It creates:

- `paper_draft/full_draft_current.md.bak_pre_schema_v2_20260801`
- `paper_draft/figure_table_plan.md.bak_pre_schema_v2_20260801`
- `paper_draft/full_draft_schema_v2_2026-08-01.md`
- `paper_draft/figure_table_plan_schema_v2_2026-08-01.md`
- `figures/flank400_schema_v2_reliability_seed11.png`
- `figures/flank400_schema_v2_reliability_seed23.png`
- `scripts/make_schema_v2_manuscript.py`

It also promotes the versioned draft and plan to the existing `full_draft_current.md` and `figure_table_plan.md` paths.

The installer expects these exact current inputs:

- `paper_draft/full_draft_current.md`: `1e4301fed2a2366e6a10f6c66307f3ca6ef8a6f105f5f1153fabf3041d03bd4f`
- `paper_draft/figure_table_plan.md`: `457b244e15f06c799a5a9619beaa4ebf9cf571bc987d9a9232a5101c5ebe4910`

If either hash differs, stop rather than forcing the update; the repository manuscript has changed again and needs a fresh exact rebase.

## Evidence included

`source_tables/` contains the eight authoritative schema-v2 cross-seed files used for the manuscript values and interpretation. The generator validates their fixed hashes, population census, row order, method set, and paired-interval result before writing.

## After applying

Run the repository's existing manuscript build command, inspect Table 1 and both reliability panels, and verify that no old five-chromosome or near-0.999 AUPRC claim remains in the rendered main paper.
