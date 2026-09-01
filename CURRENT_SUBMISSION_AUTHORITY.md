# Current TCBB Submission Authority

**Status date: 2026-09-01**

This file identifies the current submission-package authority for the
calibrated-splice-prediction project.

## Important repository boundary

The Markdown manuscripts and supplementary-material drafts under
`paper_draft/` are retained as **historical development records**.

They are not the current IEEE TCBB portal manuscript or supplementary
materials, even when a historical filename contains terms such as
`current`, `final`, `v10`, or `schema_v2`.

For repository-side quantitative and reproducibility authority, use:

- the root `README.md`;
- `reproducibility/tcbb_v9/README.md`;
- the corrected repository closure records under `reproducibility/tcbb_v9/`.

## Current corrected checkpoint authority

The corrected quantitative authority uses:

- seed 11, epoch 12;
- seed 23, epoch 13.

Corresponding repository checkpoints are:

- `results/best_models/flank400_focal_epoch12_best.pt`
- `results/best_models/flank400_seed23_focal_epoch13_best.pt`

Earlier seed-11/epoch-11, seed-23/epoch-14, flank-80, epoch-8, and related
analyses remain preserved for historical traceability but are not the current
quantitative authority.

## Git chronology

The historical scientific-freeze commit is:

`a959efa517ff11a59b8f13e80ff880242a9441b2`

The first corrected post-freeze reproducibility-closure commit is:

`b342abf6128c4a16e7eabe99c0521f64cfb8c33b`

The pre-hygiene documentation HEAD was:

`d91858eaaec59855045ae53bdaa885f44d7c109e`

Later documentation-only hygiene commits do not rewrite or supersede the
historical scientific-freeze commit.

## Frozen V10.1 submission-package artifacts

The final V10.1 four-file submission package frozen on 2026-08-31 contains:

| Portal file | SHA-256 |
|---|---|
| `TCBB_MAIN_MANUSCRIPT.pdf` | `7f0ca7481b87d0d7c30ab8d9bf45aefcc4d2cda26eea17b33bf7cbc40c42dcda` |
| `TCBB_SUPPLEMENTARY_MATERIALS.pdf` | `8b2d6fc39a2221e493aeba41cc2a8c3033567d6d571acc270ec470cd9c0bf519` |
| `TCBB_SUPPLEMENTARY_DATA.zip` | `aea4fffb4bfb68c0d83c50c147449603bd0225669acf1231032f901e2ef8f954` |
| `TCBB_LATEX_SOURCE.zip` | `3a90d4f38109eaad092adfa64e623ec977c529cbdbe04415461a68bd957e6a70` |

The V10.1 update was a human-readable supplementary-material wording polish.
Relative to the frozen V10 package:

- the main manuscript is unchanged;
- the machine-readable supplementary data are unchanged;
- the main LaTeX source package is unchanged;
- the scientific results are unchanged.

The submitted/frozen portal artifacts above, rather than historical
`paper_draft/` Markdown filenames, define the current submission package.
