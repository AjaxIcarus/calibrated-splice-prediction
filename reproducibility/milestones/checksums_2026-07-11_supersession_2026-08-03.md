# Supersession of `checksums_2026-07-11.sha256`

Status: superseded for verification; retained unchanged as a historical record.

Audit date: 2026-08-03.

## Scope

This correction concerns artifact identity and provenance only. It does not change the selected models, calibration results, robustness results, manuscript contents, scientific conclusions, accepted milestone tag, or historical checksum file.

## Audited repository identity

- Base commit: `083b26ca3175379ba055c879db991800e2a01d75`
- Base tree: `b0ec5c813a3a8a8ce4975fedec1ec91c1d3a4a5d`
- Accepted tag object: `f25e59d17e84652caf023561ca2a4b7bffcb31aa`
- Accepted tag target: `71ce901068cc1dae0e258f04e5e83056a3d3f2cf`
- Historical manifest source commit: `3403371790a89bd35aacbfbe5a781f9793ed02f1`

## Why the July 11 record is not usable for verification

| Path | July 11 declared SHA-256 | Audited SHA-256 | Finding |
|---|---|---|---|
| `results/best_models/flank400_focal_best.pt` | `637e330fe78230072c24667b180484c79f6aee2bc343219868fb410ef463ada4` | `77fdf04e4817e9b667b985bef4e34d595cc50161256178f90ea47c37a982bfc5` | The path is ignored, absent from the index and every Git ref, and therefore not recoverable or verifiable from Git. |
| `paper_draft/full_draft_current.pdf` | `b71d352c50649b2edd000ba2ae705347247d21e2fb6311522c90d7a494a3d38b` | `b872a9935d92ce2046850469963e6f86badba8878a65e351de3db03c6999d568` | The tracked PDF already had the audited hash when the historical manifest first entered Git, so the committed manifest and committed PDF were inconsistent from that point. |

The historical file `reproducibility/milestones/checksums_2026-07-11.sha256` must not be used to verify the current or accepted milestone. It remains unchanged so the provenance record is not rewritten.

## Authoritative current artifact identities

- Primary model role: seed 11, epoch 11.
  - Selection record: `results/best_models/flank400_seed11_selection.txt`
  - Selection-record SHA-256: `dc84ec4b472f443794e62a3afcca3a90cf2b7793846611e0275928619519b0c1`
  - Checkpoint: `results/best_models/flank400_focal_epoch11_best.pt`
  - Checkpoint SHA-256: `77fdf04e4817e9b667b985bef4e34d595cc50161256178f90ea47c37a982bfc5`
- Secondary robustness role: seed 23, epoch 14.
  - Selection record: `results/best_models/flank400_seed23_selection.txt`
  - Selection-record SHA-256: `f7d70996443da8164c75c887ee8242fd9d660e40ebeb326fb0dbe7b1a3ac812e`
  - Checkpoint: `results/best_models/flank400_seed23_focal_epoch14_best.pt`
  - Checkpoint SHA-256: `93945abda428fdbfe7996f0639264f2ee9ebdf909d445886e7f843f6b5a923a0`
- Current manuscript PDF: `paper_draft/full_draft_current.pdf`
  - PDF SHA-256: `b872a9935d92ce2046850469963e6f86badba8878a65e351de3db03c6999d568`

At audit time, the ignored legacy alias `results/best_models/flank400_focal_best.pt` was a separate regular-file copy, not a symbolic link or hard link. Its bytes matched the authoritative primary checkpoint. The alias is a convenience path only and must not be used as the authoritative model identity.

## Replacement verification record

The replacement is `reproducibility/milestones/flank400_schema_v2_authoritative_artifacts_2026-08-03.sha256`. It covers both selection records, both epoch-specific checkpoints, and the current manuscript PDF.

Replacement-manifest SHA-256: `2b2eb1538ec48ff97a9a27fbfed9a0a9f2fdf50834e7063301a785e29e937244`.

From the repository root, verify it with:

```bash
sha256sum -c reproducibility/milestones/flank400_schema_v2_authoritative_artifacts_2026-08-03.sha256
```

All five entries must report `OK`.
