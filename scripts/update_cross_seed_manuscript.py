#!/usr/bin/env python3
from pathlib import Path
import csv
import shutil

ROOT = Path(".")
DRAFT = ROOT / "paper_draft/full_draft_current.md"
PLAN = ROOT / "paper_draft/figure_table_plan.md"
PUB = ROOT / "paper_draft/flank400_chromosome_transfer_publication_text.md"
TABLE = ROOT / "results/flank400_chromosome_transfer_cross_seed_manuscript_table.csv"


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, found {n}")
    return text.replace(old, new, 1)


def section(text, start, end, body, label):
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{label}: section markers are not unique")
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    return left + body.rstrip() + "\n\n" + end + right


def between(text, start, end, label):
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{label}: publication-text markers are not unique")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


for path in [DRAFT, PLAN, PUB, TABLE]:
    if not path.is_file():
        raise SystemExit(f"Missing required file: {path}")

primary_candidates = [
    Path("results/best_models/flank400_focal_epoch11_best.pt"),
    Path("results/best_models/flank400_seed11_focal_epoch11_best.pt"),
    Path("results/best_models/flank400_seed11_epoch11_best.pt"),
]
primary_matches = [p for p in primary_candidates if p.is_file()]
if len(primary_matches) != 1:
    raise SystemExit(
        "Could not resolve one primary checkpoint. Existing candidates: "
        + repr([str(p) for p in primary_matches])
    )
primary_checkpoint = str(primary_matches[0])

replication_checkpoint = Path(
    "results/best_models/flank400_seed23_focal_epoch14_best.pt"
)
if not replication_checkpoint.is_file():
    raise SystemExit(
        f"Missing replication checkpoint: {replication_checkpoint}"
    )

publication = PUB.read_text()
suggested_results = between(
    publication,
    "### Calibration transfers across chromosomes and training runs",
    "## Figure caption",
    "suggested Results",
)
suggested_results = suggested_results.replace(
    "416,140,000 positions represented in the held-out\n"
    "gene/transcript records",
    "416,140,000 sequence positions across the held-out\n"
    "gene/transcript records",
)
suggested_results = suggested_results.replace(
    "fixed global T=1.1",
    "fixed global temperature scaling (T=1.1)",
)
figure_caption = between(
    publication,
    "## Figure caption",
    "## Table caption",
    "figure caption",
).replace("Figure X.", "Figure 1.")
abstract_sentence = between(
    publication,
    "## Suggested abstract sentence",
    "## Suggested limitation replacement",
    "abstract sentence",
)

with TABLE.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 5:
    raise SystemExit(f"Expected five table rows, found {len(rows)}")

table_lines = [
    "| Method | Primary ECE | Replication ECE | Primary NLL | Replication NLL |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    values = [
        f"{float(row['primary_ece']):.3e}",
        f"{float(row['replication_ece']):.3e}",
        f"{float(row['primary_nll']):.3e}",
        f"{float(row['replication_nll']):.3e}",
    ]
    if row["method"] == "OpenSpliceAI-style vector-T":
        values = [f"**{value}**" for value in values]
    table_lines.append(
        f"| {row['method'].replace('-T', ' T')} | "
        + " | ".join(values)
        + " |"
    )
table_md = "\n".join(table_lines)

draft = DRAFT.read_text()
old_abstract = draft.split("## Abstract\n\n", 1)[1].split("\n:::", 1)[0]
new_abstract = (
    "Deep learning splice-site predictors are usually evaluated as detection "
    "systems, although their output scores are often interpreted as "
    "probabilities. These objectives differ in per-nucleotide prediction, "
    "where acceptor and donor sites are extremely rare relative to non-splice "
    "positions. We evaluate prior-aware probability calibration for focal-loss "
    "OpenSpliceAI-style models trained on human GRCh38/MANE annotations with "
    "80- and 400-nt sequence context. Using true pre-softmax logits, we compare "
    "uncalibrated predictions, fixed global temperature scaling, unweighted "
    "vector temperature scaling, genome-prior-weighted vector temperature "
    "scaling, and a locally implemented OpenSpliceAI-style class-wise "
    "vector-temperature baseline. "
    + " ".join(abstract_sentence.split())
    + " Detection and calibrated probability estimation should therefore be "
    "evaluated separately, and splice-site probabilities should be interpreted "
    "only relative to an explicit target population."
)
draft = once(draft, old_abstract, new_abstract, "abstract")

draft = once(
    draft,
    "and exhaustive chr9 gene/transcript-position streaming.",
    (
        "and exhaustive streaming across five test chromosomes in two "
        "independently trained flank-400 models."
    ),
    "Introduction analysis scope",
)
draft = once(
    draft,
    (
        "5. We confirm the main flank-400 result by streaming all 52.88 "
        "million chr9 gene/transcript positions represented in the test H5, "
        "eliminating negative subsampling for that chromosome-style robustness "
        "evaluation."
    ),
    (
        "5. We test chromosome and training-run robustness by streaming "
        "416,140,000 sequence positions across chr1, chr3, chr5, chr7, and "
        "chr9 in a validation-selected seed-11/epoch-11 primary model and an "
        "independently trained, validation-selected seed-23/epoch-14 "
        "replication model."
    ),
    "Contribution 5",
)

models = f"""## Models and prediction task

We evaluated OpenSpliceAI-style per-position classifiers trained on human GRCh38 sequence [@schneider2017grch38] and MANE canonical splice-site annotations [@morales2022mane]. At each evaluated position, the model predicts one of three classes: non-splice, acceptor, or donor. Focal loss was used to address the extreme imbalance between non-splice and splice-site labels [@lin2017focal].

The primary flank-400 model was the validation-selected seed-11/epoch-11 checkpoint:

`{primary_checkpoint}`

The independent replication was the validation-selected seed-23/epoch-14 checkpoint:

`{replication_checkpoint}`

The controlled context comparison used `results/best_models/flank80_focal_epoch2_best.pt`. The two flank-400 models were trained independently and selected within seed using validation performance. The flank-80/flank-400 comparison is descriptive because context, seed, and selected epoch differ."""
draft = section(
    draft,
    "## Models and prediction task",
    "## Evaluation data and sampled caches",
    models,
    "Models",
)

draft = once(
    draft,
    (
        "The flank-400 caches were:\n\n"
        "- `results/logit_cache_flank400_epoch8/validation_sampled_logits.npz`\n"
        "- `results/logit_cache_flank400_epoch8/test_sampled_logits.npz`\n\n"
        "The flank-80 caches were stored under "
        "`results/logit_cache_flank80_epoch2/`."
    ),
    (
        "The authoritative flank-400 sampled caches were:\n\n"
        "- `results/logit_cache_flank400_seed11_epoch11/"
        "validation_sampled_logits.npz`\n"
        "- `results/logit_cache_flank400_seed11_epoch11/"
        "test_sampled_logits.npz`\n"
        "- `results/logit_cache_flank400_seed23_epoch14/"
        "validation_sampled_logits.npz`\n"
        "- `results/logit_cache_flank400_seed23_epoch14/"
        "test_sampled_logits.npz`\n\n"
        "The flank-80 caches were stored under "
        "`results/logit_cache_flank80_epoch2/`. Epoch-8 flank-400 outputs "
        "are historical and excluded from the final comparison."
    ),
    "Cache paths",
)

draft = section(
    draft,
    "For flank-400, the fitted vectors used in the final comparison were:",
    "## Target-prior weighting",
    (
        "For each flank-400 model, the unweighted, genome-prior-weighted, and "
        "OpenSpliceAI-style temperature vectors were fitted independently from "
        "that model's validation outputs. Seed-specific calibration artifacts "
        "and optimization provenance are recorded in the project results; no "
        "temperature vector was transferred between training seeds."
    ),
    "Temperature provenance",
)

streaming = """## Five-chromosome streaming robustness evaluation

To test whether weighted sampled-cache results were artifacts of negative sampling and whether the conclusion transferred across training runs, we streamed all sequence positions from chr1, chr3, chr5, chr7, and chr9 represented in `dataset_test.h5`. The validation-selected seed-11/epoch-11 primary model and seed-23/epoch-14 replication model were each evaluated over 416,140,000 positions. Neural-network inference was performed once per model and chromosome, after which all five calibration settings were evaluated on the same logits.

This evaluation is exhaustive for the evaluated H5 gene/transcript records, not complete intergenic whole-genome inference. Chromosome comparisons are descriptive because the five chromosomes were selected evaluation partitions rather than randomly sampled replicate units."""
draft = section(
    draft,
    "## Chr9 streaming robustness evaluation",
    "## Implementation",
    streaming,
    "Streaming Methods",
)
draft = once(
    draft,
    "and chr9 streaming evaluation",
    "and five-chromosome streaming evaluation",
    "Implementation scope",
)

results_block = f"""## Calibration transfers across chromosomes and training runs

{suggested_results}

:::

::: {{.wide-table}}
**Table 1. Cross-seed robustness of five-chromosome flank-400 calibration transfer.**

{table_md}

Each model covers 416,140,000 positions across chr1, chr3, chr5, chr7, and chr9. Lower is better. Results are descriptive across two training runs.
:::

::: {{.paper-body}}

:::

::: {{.wide-figure}}
![{figure_caption}](figures/flank400_chromosome_transfer_cross_seed.pdf)
:::

::: {{.paper-body}}

## Comparison with flank-80

The weaker-context flank-80 checkpoint showed the same qualitative separation between detection and calibration. Because the flank-80 and flank-400 models differ in context, training seed, and selected epoch, this comparison is supporting evidence rather than a causal estimate of context length."""
draft = section(
    draft,
    "## Primary flank-400 result",
    "## Calibration under the wrong prior can be misleading",
    results_block,
    "Primary Results",
)

wrong_prior = """## Calibration under the wrong prior can be misleading

In the five-chromosome primary analysis, unweighted vector scaling reduced ECE from 1.507e-03 to 4.692e-05 but increased NLL from 1.698e-03 to 2.062e-03. The replication showed the same disagreement: ECE fell from 1.545e-03 to 5.049e-05 while NLL increased from 1.726e-03 to 1.975e-03. Thus, ECE alone is insufficient and the target prior must be declared.

Fixed global temperature scaling (T=1.1) also worsened both metrics in both trained models. A positive scalar temperature preserves multiclass argmax ordering, so this deterioration cannot be detected from argmax counts alone."""
draft = section(
    draft,
    "## Calibration under the wrong prior can be misleading",
    "## Argmax and threshold behavior",
    wrong_prior,
    "Wrong-prior Results",
)

operating = """## Argmax and threshold behavior

Calibration changed the probability scale even when ranking changed little. Consequently, thresholds were not transferred unchanged between uncalibrated, globally scaled, and vector-scaled outputs. Threshold-specific precision and recall were treated as operating-point analyses, and no universal threshold was inferred. The unweighted-vector result further shows why argmax counts and ECE cannot be interpreted alone: a method can appear favorable on one summary while assigning worse likelihood to the observed labels under the target population."""
draft = section(
    draft,
    "## Argmax and threshold behavior",
    "## Bootstrap, reliability, and prior sensitivity",
    operating,
    "Operating-point Results",
)

support = """## Bootstrap, reliability, and prior sensitivity

Within-model bootstrap resampling and reliability diagrams supported the conclusion that prior-compatible vector scaling improved probability alignment relative to the uncalibrated model. The independent seed-23 model and five-chromosome evaluation provide the separate training-run and transfer check; position-level bootstrap intervals are not interpreted as training-seed uncertainty.

Prior-sensitivity analysis provided a mechanistic explanation: as the validation non-splice weight approached the target-position weight, target-prior NLL and ECE improved and excessive splice argmax counts decreased. The flank-80 analyses are retained as a context comparison rather than as the primary flank-400 evidence.

:::

::: {.figure-grid}
![Multiclass reliability for seed-11/epoch-11.](figures/logit_based_flank400_seed11_epoch11/reliability_multiclass_flank400_methods.png)

![Acceptor reliability for seed-11/epoch-11.](figures/logit_based_flank400_seed11_epoch11/reliability_acceptor_flank400_methods.png)

![Donor reliability for seed-11/epoch-11.](figures/logit_based_flank400_seed11_epoch11/reliability_donor_flank400_methods.png)
:::

::: {.paper-body}"""
draft = section(
    draft,
    "## Bootstrap, reliability, and prior sensitivity",
    "## Summary of results",
    support,
    "Supporting Results",
)

draft = section(
    draft,
    "## Summary of results",
    "# Discussion",
    """## Summary of results

Across both validation-selected flank-400 models, OpenSpliceAI-style vector temperature scaling achieved the lowest pooled ECE and NLL, while genome-prior-weighted vector scaling also improved both metrics. Unweighted fitting produced low ECE but worse NLL, and fixed global scaling worsened both metrics. Agreement across five test chromosomes and two independently trained models shows that the qualitative result was not created by negative subsampling or a single training run.""",
    "Results summary",
)

draft = once(
    draft,
    (
        "The selected flank-400 checkpoint achieved acceptor and donor AUPRC "
        "near 0.9995, but its uncalibrated probabilities remained less "
        "suitable under the target-position prior than either prior-compatible "
        "vector calibration procedure."
    ),
    (
        "The validation-selected flank-400 models achieved very strong "
        "splice-site ranking on sampled-cache analyses, while their "
        "uncalibrated probabilities remained less suitable under the "
        "target-position population than either prior-compatible vector "
        "calibration procedure."
    ),
    "Discussion detection",
)
draft = once(
    draft,
    "all positions represented in the GRCh38/MANE evaluation H5",
    "all sequence positions across the GRCh38/MANE evaluation H5",
    "Target population wording",
)

draft = section(
    draft,
    "## Interpretation of the OpenSpliceAI-style comparison",
    "## Practical implications",
    """## Interpretation of the OpenSpliceAI-style comparison

The local OpenSpliceAI-style vector-temperature baseline had the lowest pooled ECE and NLL in both independently trained flank-400 models. This should not be described as our method beating OpenSpliceAI calibration or as a benchmark against every published OpenSpliceAI model. The supported contribution is the explicit prior-aware comparison and evaluation framework. The five-chromosome comparisons are descriptive replications, not an inferential test of superiority over chromosomes.""",
    "OpenSpliceAI interpretation",
)

limitations = """## Limitations

The main flank-400 calibration pattern replicated in a second independently trained model, reducing dependence on a single checkpoint. Nevertheless, two validation-selected training runs do not fully characterize optimization variability, and the five evaluated chromosomes should not be treated as randomly sampled replicate units.

Neither configuration is a full SpliceAI-10k ensemble or a claim of state-of-the-art detection. The flank-80/flank-400 comparison also does not isolate the causal effect of context length.

The target population covers sequence positions represented in the MANE gene/transcript evaluation H5. Five-chromosome streaming removes negative subsampling for those records, but it is not complete intergenic whole-genome inference. Calibration is also not demonstrated for all isoforms, noncanonical or cryptic sites, variant-altered sequences, or tissue-specific splice usage.

The OpenSpliceAI-style baseline was implemented locally to reproduce its class-wise vector-temperature behavior. We compare calibration protocols and objectives rather than every published OpenSpliceAI model. Finally, post-hoc calibration does not separately quantify epistemic and aleatoric uncertainty."""
draft = section(
    draft,
    "## Limitations",
    "## Conclusion",
    limitations,
    "Limitations",
)

conclusion = """## Conclusion

Across two independently trained focal-loss OpenSpliceAI-style flank-400 models, strong splice-site detection did not guarantee calibrated target-prior probabilities. Streaming 416,140,000 sequence positions across five test chromosomes per model reproduced the calibration ordering: OpenSpliceAI-style vector temperature scaling achieved the lowest pooled ECE and NLL, while genome-prior-weighted vector scaling also sharply improved both metrics. Unweighted vector scaling reduced ECE but worsened NLL, demonstrating that a small ECE alone does not establish better probability quality. Splice-site probabilities should therefore be evaluated and interpreted relative to an explicit target population, separately from detection performance.

:::"""
draft = section(
    draft,
    "## Conclusion",
    "::: {.references-section}",
    conclusion,
    "Conclusion",
)

plan = f"""# Figure and Table Plan

## Authoritative evidence

- Primary: seed 11, epoch 11.
- Replication: seed 23, epoch 14.
- Transfer scope: chr1, chr3, chr5, chr7, and chr9.
- Evaluated positions: 416,140,000 per model.
- Epoch-8 and single-chr9 flank-400 outputs are historical only.

## Main text figures

### Figure 1: Cross-seed chromosome transfer

- Final: `figures/flank400_chromosome_transfer_cross_seed.pdf`
- Preview: `figures/flank400_chromosome_transfer_cross_seed.png`

{figure_caption}

### Figure 2: Reliability diagrams

- `figures/logit_based_flank400_seed11_epoch11/reliability_multiclass_flank400_methods.png`
- `figures/logit_based_flank400_seed11_epoch11/reliability_acceptor_flank400_methods.png`
- `figures/logit_based_flank400_seed11_epoch11/reliability_donor_flank400_methods.png`

Use as supporting probability-alignment evidence for the primary model.

### Figure 3: Detection and operating points

Generate later from:

- `results/detection_metrics_flank400_seed11_epoch11/`
- `results/detection_metrics_flank400_seed23_epoch14/`

Do not reuse the epoch-8 detection figure.

## Main text tables

### Table 1: Cross-seed five-chromosome calibration

- `results/flank400_chromosome_transfer_cross_seed_manuscript_table.csv`
- `results/flank400_chromosome_transfer_cross_seed_manuscript_table.tex`

Report all five methods for both trained models. Lower ECE/NLL is better; comparisons are descriptive.

### Table 2: Detection and threshold summary

Pending exact cross-seed collation from the two detection directories. Include acceptor/donor AUPRC and only validation-selected operating points; do not include epoch-8 values.

## Supplementary tables

### Table S1: Per-chromosome metrics

- `results/flank400_chromosome_transfer_cross_seed_paired_by_chromosome.csv`
- `results/flank400_chromosome_transfer_cross_seed_paired_summary.csv`

### Table S2: Cross-seed contrasts

- `results/flank400_chromosome_transfer_cross_seed_pooled_comparison.csv`
- `results/flank400_chromosome_transfer_cross_seed_key_contrasts.csv`

### Table S3: Bootstrap summaries

- `results/logit_bootstrap_flank400_seed11_epoch11/bootstrap_summary.csv`
- `results/logit_bootstrap_flank400_seed23_epoch14/bootstrap_summary.csv`
- `results/bootstrap_openspliceai_style_vectorT_flank400_seed11_epoch11/bootstrap_summary.csv`
- `results/bootstrap_openspliceai_style_vectorT_flank400_seed23_epoch14/bootstrap_summary.csv`

Bootstrap intervals describe sampled-position variability within models, not training-seed uncertainty.

### Table S4: Reliability bins

- `results/flank400_chromosome_transfer_seed11_epoch11_recomputed_pooled_reliability_bins.csv`
- `results/flank400_chromosome_transfer_seed23_epoch14_pooled_reliability_bins.csv`

## Retired from the main manuscript

- `figures/flank400_primary_calibration.png`
- `figures/flank400_chr9_calibration.png`
- any table or figure labeled `flank400_epoch8`
- the old single-chr9 calibration and argmax tables
"""

for path in [DRAFT, PLAN]:
    backup = path.with_suffix(path.suffix + ".bak_pre_cross_seed_20260724")
    if backup.exists():
        raise SystemExit(f"Refusing to overwrite backup: {backup}")
    shutil.copy2(path, backup)

DRAFT.write_text(draft)
PLAN.write_text(plan)

for obsolete in [
    "flank400_focal_epoch8_best.pt",
    "52.88 million chr9",
    "flank400_primary_calibration.png",
    "Exhaustive flank-400 chr9",
]:
    if obsolete in draft:
        raise SystemExit(f"Obsolete primary material remains: {obsolete}")

print(f"PASS: updated {DRAFT}")
print(f"PASS: updated {PLAN}")
print("PASS: backups created; Git untouched")
