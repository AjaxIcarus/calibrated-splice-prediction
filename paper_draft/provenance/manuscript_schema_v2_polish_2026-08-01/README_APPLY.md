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

# Apply the schema-v2 manuscript polish

This package applies the publication-polish pass on top of the already promoted schema-v2 manuscript. It does not modify Git and does not overwrite an unexpected manuscript state.

From the repository root, run:

```bash
bash manuscript_schema_v2_polish_2026-08-01/apply_schema_v2_polish.sh
```

The installer validates its own files and the exact pre-polish manuscript hashes, creates backups, installs versioned polished files, promotes the polished manuscript and figure/table plan, and adds the supplementary manuscript plus machine-readable source tables.

If any expected target already exists or the current manuscript has changed, the installer stops before copying files.
