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

# References TODO List

## Core splice prediction papers

### Barash et al. — Splicing code / tissue-dependent splicing

Use for:

- early framing of splicing as a regulatory code
- feature-based prediction of tissue-dependent splicing
- motivation that splicing depends on combinations of sequence/regulatory features

Citation placeholder:

`[Barash et al., 2010]`

Where used:

- Introduction
- Related Work: Splicing codes and feature-based prediction

---

### Jaganathan et al. — SpliceAI

Use for:

- deep learning from primary sequence
- per-position acceptor/donor/neither prediction
- long-range sequence context
- splice-altering variant prediction

Citation placeholder:

`[Jaganathan et al., 2019]`

Where used:

- Introduction
- Related Work: Deep learning for splice-site prediction

---

### Chao et al. — OpenSpliceAI

Use for:

- PyTorch reimplementation of SpliceAI-style models
- retraining and transfer learning
- preprocessing from FASTA/GFF/GTF
- calibration support
- model prediction and variant analysis

Citation placeholder:

`[Chao et al., 2025]`

Where used:

- Introduction
- Related Work: OpenSpliceAI and retrainable splice prediction
- Methods: Model and prediction task

---

## Calibration papers

### Guo et al. — Temperature scaling / neural network calibration

Use for:

- neural networks can be accurate but miscalibrated
- temperature scaling as post-hoc calibration
- ECE/NLL calibration evaluation framing

Citation placeholder:

`[Guo et al., 2017]`

Where used:

- Related Work: Calibration of neural network probabilities
- Methods: Temperature scaling
- Discussion: calibration interpretation

---

### Naeini et al. or calibration/ECE source

Use for:

- expected calibration error
- reliability diagrams
- confidence vs empirical accuracy

Citation placeholder:

`[Naeini et al., 2015]`

Where used:

- Methods: Calibration metrics

---

## Imbalance / loss function papers

### Lin et al. — Focal loss

Use for:

- focal loss for class imbalance
- motivation for training under rare-positive setting

Citation placeholder:

`[Lin et al., 2017]`

Where used:

- Methods: Model and prediction task
- Possibly Introduction if mentioning focal-loss model

---

## Optional splice-related background papers

### Bretschneider et al. — COSSMO

Use for:

- alternative splice-site choice
- competitive splice-site selection
- PSI / splice-site usage framing

Citation placeholder:

`[Bretschneider et al.]`

Where used:

- Optional Related Work paragraph if expanding beyond canonical splice-site detection

---

### Zeng RNA splicing review/background

Use for:

- general RNA splicing biological background
- importance of alternative splicing
- splicing regulation overview

Citation placeholder:

`[Zeng et al.]`

Where used:

- First paragraph of Introduction, if needed

---

## Optional future-work references

### GTEx Consortium

Use for:

- tissue-specific expression/splicing data
- future work on tissue-specific calibration

Citation placeholder:

`[GTEx Consortium]`

Where used:

- Discussion: Future work

---

### Variant interpretation / clinical splicing references

Use for:

- variant-effect calibration motivation
- donor/acceptor gain-loss interpretation

Citation placeholder:

`[variant-splicing reference]`

Where used:

- Discussion: Future work

---

# Immediate citation tasks

1. Replace informal placeholders such as `[Barash et al., 2010]` with real citation keys.
2. Create a BibTeX file later, likely:

   `paper_draft/references.bib`

3. Use consistent citation keys, for example:

   - `barash2010splicingcode`
   - `jaganathan2019spliceai`
   - `chao2025openspliceai`
   - `guo2017calibration`
   - `naeini2015calibration`
   - `lin2017focalloss`

4. Later convert manuscript citations from plain text placeholders to citation-key format:

   - `[Barash et al., 2010]` → `[@barash2010splicingcode]`
   - `[Jaganathan et al., 2019]` → `[@jaganathan2019spliceai]`
   - `[Chao et al., 2025]` → `[@chao2025openspliceai]`
   - `[Guo et al., 2017]` → `[@guo2017calibration]`
   - `[Naeini et al., 2015]` → `[@naeini2015calibration]`
   - `[Lin et al., 2017]` → `[@lin2017focalloss]`
