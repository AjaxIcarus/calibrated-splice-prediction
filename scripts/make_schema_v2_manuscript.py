#!/usr/bin/env python3
"""Generate the valid-only schema-v2 manuscript and figure/table plan.

This is intentionally non-destructive: it reads the exact August 1 manuscript
state supplied after the first install guard stopped, validates the authoritative
schema-v2 tables, and writes new versioned files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "draft": "1e4301fed2a2366e6a10f6c66307f3ca6ef8a6f105f5f1153fabf3041d03bd4f",
    "plan": "457b244e15f06c799a5a9619beaa4ebf9cf571bc987d9a9232a5101c5ebe4910",
    "core": "6e8c8f7564e7e7cbe9b836c93776a3463fd984743774f995ef082845bfba7505",
    "paired": "0779585d79b0804ca063f8c568387d04453eb4b7987e8fbe7dcfa856adf83ceb",
    "metadata": "c76919bd96a5417a3e1538371e5b84532d22966121280ab49ed2b6eba2b720ca",
}

METHODS = [
    "uncalibrated",
    "global_T_1.1",
    "true_logit_unweighted_vector_T",
    "true_logit_target_weighted_vector_T",
    "openspliceai_style_full_validation_vector_T_v2",
]

DISPLAY = {
    "uncalibrated": "Uncalibrated",
    "global_T_1.1": "Fixed global T=1.1",
    "true_logit_unweighted_vector_T": "Unweighted true-logit vector T",
    "true_logit_target_weighted_vector_T": "Target-weighted true-logit vector T",
    "openspliceai_style_full_validation_vector_T_v2": (
        "OpenSpliceAI-style full-validation vector T"
    ),
}

METRICS = [
    "weighted_multiclass_ece",
    "weighted_multiclass_nll",
    "weighted_multiclass_brier",
    "target_weighted_acceptor_auprc",
    "target_weighted_donor_auprc",
]

EXPECTED_POPULATION = {
    "cache_schema_version": 2,
    "all_sequences_included": True,
    "padding_excluded_from_sampling": True,
    "total_sequences_seen": 84258,
    "total_positions_seen": 421290000,
    "valid_positions_seen": 407278622,
    "padding_positions_seen": 14011378,
    "positive_positions_seen": 107988,
    "valid_nonsplice_positions_seen": 407170634,
    "sampled_negatives": 500000,
    "cache_random_seed": 11,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(
            f"ERROR: {label} SHA-256 mismatch\n"
            f"expected: {expected}\nactual:   {actual}\npath:     {path}"
        )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"ERROR: {label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_section(
    text: str,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(
            f"ERROR: {label}: start={text.count(start)}, end={text.count(end)}"
        )
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + replacement.rstrip() + "\n\n" + end + after


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct_reduction(baseline: float, calibrated: float) -> float:
    return 100.0 * (baseline - calibrated) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--core-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--output-draft", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "draft": args.draft,
        "plan": args.plan,
        "core": args.core_csv,
        "paired": args.paired_csv,
        "metadata": args.metadata_json,
    }
    for label, path in inputs.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"ERROR: missing or empty {label}: {path}")
        require_hash(path, EXPECTED[label], label)

    for output in [args.output_draft, args.output_plan]:
        if output.exists():
            raise SystemExit(f"ERROR: refusing to overwrite output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_POPULATION.items():
        actual = metadata["population"].get(key)
        if actual != expected:
            raise SystemExit(
                f"ERROR: population mismatch for {key}: {actual!r} != {expected!r}"
            )

    core = read_rows(args.core_csv)
    paired = read_rows(args.paired_csv)
    if len(core) != 10:
        raise SystemExit(f"ERROR: expected 10 core rows, found {len(core)}")
    if len(paired) != 10:
        raise SystemExit(f"ERROR: expected 10 paired rows, found {len(paired)}")
    if any(row["improvement_ci_excludes_zero"] != "False" for row in paired):
        raise SystemExit("ERROR: a paired interval unexpectedly excludes zero")

    expected_order = [
        ("primary", method) for method in METHODS
    ] + [("replication", method) for method in METHODS]
    observed_order = [(row["model_role"], row["method"]) for row in core]
    if observed_order != expected_order:
        raise SystemExit("ERROR: core method/model order is not authoritative")

    by_key = {(row["model_role"], row["method"]): row for row in core}

    table = [
        "| Method | Weighted ECE | Weighted NLL | Weighted Brier | "
        "Target-weighted acceptor AUPRC | Target-weighted donor AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        primary = by_key[("primary", method)]
        replication = by_key[("replication", method)]
        values = []
        for metric in METRICS:
            left = float(primary[metric])
            right = float(replication[metric])
            if metric.endswith("auprc"):
                values.append(f"{left:.6f} / {right:.6f}")
            else:
                values.append(f"{left:.3e} / {right:.3e}")
        table.append(
            f"| {DISPLAY[method]} | " + " | ".join(values) + " |"
        )
    table_md = "\n".join(table)

    primary_uncal = by_key[("primary", "uncalibrated")]
    replication_uncal = by_key[("replication", "uncalibrated")]
    primary_target = by_key[
        ("primary", "true_logit_target_weighted_vector_T")
    ]
    replication_target = by_key[
        ("replication", "true_logit_target_weighted_vector_T")
    ]
    primary_open = by_key[
        ("primary", "openspliceai_style_full_validation_vector_T_v2")
    ]
    replication_open = by_key[
        ("replication", "openspliceai_style_full_validation_vector_T_v2")
    ]

    reductions = {}
    for short, metric in [
        ("ECE", "weighted_multiclass_ece"),
        ("NLL", "weighted_multiclass_nll"),
        ("Brier", "weighted_multiclass_brier"),
    ]:
        reductions[short] = (
            pct_reduction(float(primary_uncal[metric]), float(primary_target[metric])),
            pct_reduction(
                float(replication_uncal[metric]),
                float(replication_target[metric]),
            ),
            pct_reduction(float(primary_uncal[metric]), float(primary_open[metric])),
            pct_reduction(
                float(replication_uncal[metric]),
                float(replication_open[metric]),
            ),
        )

    draft = args.draft.read_text(encoding="utf-8")
    old_abstract = draft.split("## Abstract\n\n", 1)[1].split("\n:::", 1)[0]
    new_abstract = (
        "Deep learning splice-site predictors are usually evaluated as detection "
        "systems, although their output scores are often interpreted as probabilities. "
        "These objectives differ in per-nucleotide prediction, where acceptor and donor "
        "sites are extremely rare relative to non-splice positions. We evaluate "
        "prior-aware probability calibration for focal-loss OpenSpliceAI-style models "
        "trained on human GRCh38/MANE annotations. Using true pre-softmax logits, we "
        "compare uncalibrated predictions, fixed global temperature scaling, unweighted "
        "vector scaling, target-weighted vector scaling, and a locally implemented "
        "OpenSpliceAI-style full-validation vector-temperature baseline. In matched "
        "valid-only evaluations of a validation-selected seed-11/epoch-11 primary model "
        "and seed-23/epoch-14 replication, target-weighted vector scaling reduced "
        f"weighted multiclass ECE by {reductions['ECE'][0]:.2f}%/{reductions['ECE'][1]:.2f}%, "
        f"NLL by {reductions['NLL'][0]:.2f}%/{reductions['NLL'][1]:.2f}%, and Brier score "
        f"by {reductions['Brier'][0]:.2f}%/{reductions['Brier'][1]:.2f}%. The "
        "OpenSpliceAI-style baseline produced closely comparable reductions, and none "
        "of ten seed-by-metric paired 95% bootstrap intervals excluded zero. "
        "Target-weighted acceptor and donor AUPRC changed minimally. Detection and "
        "calibrated probability estimation should therefore be evaluated separately, "
        "and splice-site probabilities should be interpreted only relative to an "
        "explicit target population."
    )
    draft = replace_once(draft, old_abstract, new_abstract, "abstract")

    draft = replace_once(
        draft,
        (
            "We study this distinction using focal-loss OpenSpliceAI-style models "
            "with 80- and 400-nt context. Flank-400 is the primary experiment; "
            "flank-80 provides a controlled weaker-context comparison. We extract "
            "true pre-softmax logits and compare uncalibrated scores, fixed global "
            "temperature scaling, unweighted vector scaling, genome-prior weighted "
            "vector scaling, and a locally implemented OpenSpliceAI-style "
            "class-wise vector-temperature baseline. We evaluate probability "
            "quality using ECE, NLL, reliability diagrams, bootstrap resampling, "
            "prior-sensitivity analysis, and exhaustive streaming across five test "
            "chromosomes in two independently trained flank-400 models. Detection "
            "is evaluated separately using acceptor/donor AUPRC and threshold "
            "precision-recall."
        ),
        (
            "We study this distinction using focal-loss OpenSpliceAI-style models "
            "with 80- and 400-nt context. Flank-400 is the primary experiment; "
            "flank-80 is retained as a secondary context comparison. We extract true "
            "pre-softmax logits and compare uncalibrated scores, fixed global "
            "temperature scaling, unweighted vector scaling, target-weighted vector "
            "scaling, and a locally implemented OpenSpliceAI-style full-validation "
            "class-wise vector-temperature baseline. The primary evidence uses "
            "schema-v2 caches that include every sequence, exclude all-zero padding, "
            "and reconstruct the valid-position target population. Probability "
            "quality is evaluated using ECE, NLL, Brier score, reliability diagrams, "
            "and paired bootstrap resampling; target-weighted AUPRC and operating-point "
            "analyses are reported separately."
        ),
        "introduction scope",
    )
    draft = replace_once(
        draft,
        (
            "4. We demonstrate the calibration-detection separation across flank-80 "
            "and flank-400 checkpoints: probability quality changes substantially "
            "while acceptor and donor ranking remains nearly unchanged.\n"
            "5. We test chromosome and training-run robustness by streaming "
            "416,140,000 sequence positions across chr1, chr3, chr5, chr7, and chr9 "
            "in a validation-selected seed-11/epoch-11 primary model and an "
            "independently trained, validation-selected seed-23/epoch-14 replication "
            "model."
        ),
        (
            "4. We demonstrate the calibration-detection separation in matched "
            "valid-position evaluations: probability quality changes substantially "
            "while target-weighted acceptor and donor AUPRC values change minimally.\n"
            "5. We evaluate a validation-selected seed-11/epoch-11 primary model and "
            "an independently trained, validation-selected seed-23/epoch-14 "
            "replication using identical populations, metrics, calibration methods, "
            "and paired bootstrap design."
        ),
        "contributions 4-5",
    )

    evaluation = """## Evaluation data and valid-only sampled caches

The flank-400 analysis used `data/processed_h5_flank400/dataset_validation.h5` and `data/processed_h5_flank400/dataset_test.h5`. Across all 84,258 test sequences, the H5 contained 421,290,000 raw fixed-length positions. We excluded 14,011,378 all-zero padding positions, leaving a target population of 407,278,622 valid positions: 407,170,634 non-splice positions, 53,994 acceptors, and 53,994 donors.

Each schema-v2 test cache retained all 107,988 valid splice-site positions and a reservoir sample of 500,000 valid non-splice positions, for 607,988 cached positions. The validation population similarly contained 89,016,242 valid positions after excluding 3,413,758 padding positions: 88,990,332 non-splice positions, 12,955 acceptors, and 12,955 donors.

The authoritative flank-400 caches were:

- `results/logit_cache_flank400_seed11_epoch11_fullpopulation_validonly_v2/validation_sampled_logits.npz`
- `results/logit_cache_flank400_seed11_epoch11_fullpopulation_validonly_v2/test_sampled_logits.npz`
- `results/logit_cache_flank400_seed23_epoch14_fullpopulation_validonly_v2/validation_sampled_logits.npz`
- `results/logit_cache_flank400_seed23_epoch14_fullpopulation_validonly_v2/test_sampled_logits.npz`

Both seeds used the same complete sequence inventory and valid-position census. Earlier caches that truncated shard tails or treated all-zero padding as non-splice are historical and excluded from the final comparison. The flank-80 analyses are retained only as a secondary context comparison and are not mixed into the authoritative flank-400 table."""
    draft = replace_section(
        draft,
        "## Evaluation data and sampled caches",
        "## True-logit extraction",
        evaluation,
        "evaluation population",
    )

    calibration = """## Calibration methods

For logit vector `z = [z_nonsplice, z_acceptor, z_donor]` and temperature vector `T = [T_nonsplice, T_acceptor, T_donor]`, calibrated probabilities were computed as `softmax(z / T)`, with elementwise division. Model weights remained frozen.

We compared:

1. **Uncalibrated:** direct softmax probabilities.
2. **Fixed global T=1.1:** the same scalar applied to all three logits.
3. **Unweighted true-logit vector T:** three temperatures fitted on the splice-enriched sampled validation distribution.
4. **Target-weighted true-logit vector T:** three temperatures fitted with valid non-splice importance weights that reconstruct the target-position prior.
5. **OpenSpliceAI-style full-validation vector T:** a locally implemented class-wise vector-temperature baseline fitted from all valid validation positions following the OpenSpliceAI-style objective.

Each model's vector temperatures were fitted independently using its own validation outputs. No temperature vector was transferred between training seeds. Seed-specific calibration artifacts and optimization provenance are recorded in the project results."""
    draft = replace_section(
        draft,
        "## Calibration methods",
        "## Target-prior weighting",
        calibration,
        "calibration methods",
    )

    weighting = """## Target-prior weighting

Because each schema-v2 cache retained every valid splice site but sampled valid non-splice positions, each sampled test non-splice position received reconstruction weight

`407,170,634 / 500,000 = 814.341268`.

The corresponding validation non-splice weight was

`88,990,332 / 500,000 = 177.980664`.

Splice-site positions received unit weight. These weights reconstruct the distribution over valid positions represented in the evaluation H5 records. They do not represent every intergenic base in complete chromosomes."""
    draft = replace_section(
        draft,
        "## Target-prior weighting",
        "## Calibration and detection metrics",
        weighting,
        "target-prior weighting",
    )

    metrics_text = """## Calibration and detection metrics

Probability quality was evaluated with target-weighted multiclass ECE, NLL, and Brier score, together with class-wise acceptor and donor reliability diagrams. Multiclass ECE used 15 equal-width bins of maximum class probability. NLL and Brier are proper scoring rules and were included to prevent apparently small ECE from being interpreted in isolation. Lower ECE, NLL, and Brier indicate better probability quality under the declared target population.

Detection was evaluated separately using acceptor and donor AUPRC, top-k summaries, and threshold-specific precision and recall. The main cross-seed table reports target-weighted AUPRC, computed under the same reconstructed valid-position population. Sampled-cache AUPRC describes the splice-enriched cache and is not interchangeable with target-weighted AUPRC. Precision-recall summaries are more informative than ROC summaries for highly imbalanced tasks [@saito2015precisionrecall]."""
    draft = replace_section(
        draft,
        "## Calibration and detection metrics",
        "## Bootstrap and prior-sensitivity analyses",
        metrics_text,
        "metrics",
    )

    bootstrap = """## Bootstrap and prior-sensitivity analyses

For each trained model, we generated 200 paired position-level bootstrap replicates using random seed 17, stratified resampling by true class, and the same resampled positions across calibration methods. We obtained 95% percentile intervals for all 13 recorded metrics and for paired method contrasts. These intervals quantify evaluated-position uncertainty within each trained model; they do not estimate training-seed, gene, or chromosome variability.

Prior sensitivity was evaluated by varying the validation non-splice weight from the splice-enriched setting toward the valid-position target weight and refitting the temperature vector. Learned temperatures, target-prior ECE and NLL, and argmax class counts were recorded as mechanistic support for the effect of the declared prior."""
    draft = replace_section(
        draft,
        "## Bootstrap and prior-sensitivity analyses",
        "## Five-chromosome streaming robustness evaluation",
        bootstrap,
        "bootstrap",
    )

    robustness = """## Matched cross-seed robustness design

The validation-selected seed-11/epoch-11 checkpoint was designated the primary model and the independently trained seed-23/epoch-14 checkpoint the replication. Both were evaluated using the identical valid-position population, five calibration methods, 13-metric schema, and paired 200-replicate bootstrap design. Agreement across two trained models is treated as descriptive robustness, not a population-level estimate of optimization variability."""
    draft = replace_section(
        draft,
        "## Five-chromosome streaming robustness evaluation",
        "## Implementation",
        robustness,
        "robustness design",
    )
    draft = replace_once(
        draft,
        "context comparison, and five-chromosome streaming evaluation",
        "context comparison, and matched cross-seed schema-v2 evaluation",
        "implementation scope",
    )

    results = f"""## Matched schema-v2 cross-seed probability quality

Target-weighted true-logit vector scaling reduced weighted multiclass ECE by {reductions['ECE'][0]:.2f}% and {reductions['ECE'][1]:.2f}%, NLL by {reductions['NLL'][0]:.2f}% and {reductions['NLL'][1]:.2f}%, and Brier score by {reductions['Brier'][0]:.2f}% and {reductions['Brier'][1]:.2f}% relative to the corresponding uncalibrated seed-11 and seed-23 models. OpenSpliceAI-style full-validation vector scaling produced closely comparable reductions in ECE ({reductions['ECE'][2]:.2f}%/{reductions['ECE'][3]:.2f}%), NLL ({reductions['NLL'][2]:.2f}%/{reductions['NLL'][3]:.2f}%), and Brier score ({reductions['Brier'][2]:.2f}%/{reductions['Brier'][3]:.2f}%). Target-weighted acceptor and donor AUPRC changed only minimally after either prior-aware calibration method.

In contrast, fixed global T=1.1 worsened all three probability-quality metrics in both models. Unweighted vector scaling reduced ECE but worsened NLL and Brier score, showing why ECE alone is insufficient under extreme class imbalance. None of the ten seed-by-metric paired 95% bootstrap intervals comparing target-weighted and OpenSpliceAI-style vector scaling excluded zero. The supported conclusion is therefore descriptive consistency of both prior-aware procedures, not superiority of either method.

:::

::: {{.wide-table}}
**Table 1. Matched flank-400 cross-seed robustness on the schema-v2 valid-position target population.**

Values are seed-11 epoch-11 / seed-23 epoch-14.

{table_md}

ECE, NLL, and Brier are lower-is-better; AUPRC is higher-is-better. All values use the same reconstructed valid-position target population. Bootstrap intervals used 200 paired, true-class-stratified position-level replicates per model and quantify evaluated-position uncertainty, not training-seed variability.
:::

::: {{.paper-body}}

:::

::: {{.figure-grid}}
![**Figure 1a.** Schema-v2 reliability diagnostics for the validation-selected seed-11/epoch-11 primary model.](figures/flank400_schema_v2_reliability_seed11.png)

![**Figure 1b.** Schema-v2 reliability diagnostics for the validation-selected seed-23/epoch-14 replication model.](figures/flank400_schema_v2_reliability_seed23.png)
:::

::: {{.paper-body}}

The class-wise curves are visually variable in rare high-probability bins because those bins contain little target-population mass. Their contribution to weighted ECE is correspondingly small. The unweighted-vector curves show a substantive failure that is also captured by their worse NLL and Brier score."""
    draft = replace_section(
        draft,
        "## Calibration transfers across chromosomes and training runs",
        "## Comparison with flank-80",
        results,
        "primary results",
    )

    draft = replace_section(
        draft,
        "## Comparison with flank-80",
        "## Calibration under the wrong prior can be misleading",
        """## Comparison with flank-80

The weaker-context flank-80 analyses showed the same qualitative separation between detection and calibration and are retained as secondary developmental context. They are not included in Table 1 because the authoritative cross-seed comparison is restricted to the two matched flank-400 schema-v2 evaluations; the flank-80/flank-400 comparison does not isolate the causal effect of context length.""",
        "flank-80 comparison",
    )

    unweighted_p = by_key[("primary", "true_logit_unweighted_vector_T")]
    unweighted_r = by_key[("replication", "true_logit_unweighted_vector_T")]
    global_p = by_key[("primary", "global_T_1.1")]
    global_r = by_key[("replication", "global_T_1.1")]
    wrong_prior = f"""## Calibration under the wrong prior can be misleading

Unweighted vector scaling lowered ECE from {float(primary_uncal['weighted_multiclass_ece']):.3e} to {float(unweighted_p['weighted_multiclass_ece']):.3e} in the primary model and from {float(replication_uncal['weighted_multiclass_ece']):.3e} to {float(unweighted_r['weighted_multiclass_ece']):.3e} in the replication. Nevertheless, NLL increased from {float(primary_uncal['weighted_multiclass_nll']):.3e} to {float(unweighted_p['weighted_multiclass_nll']):.3e} and from {float(replication_uncal['weighted_multiclass_nll']):.3e} to {float(unweighted_r['weighted_multiclass_nll']):.3e}; Brier score also increased more than fivefold in each model. This disagreement shows why ECE alone is insufficient and why the target prior must be declared.

Fixed global T=1.1 also worsened ECE, NLL, and Brier score in both trained models (primary NLL {float(global_p['weighted_multiclass_nll']):.3e}; replication NLL {float(global_r['weighted_multiclass_nll']):.3e}). A positive scalar temperature preserves multiclass argmax ordering, so this deterioration cannot be detected from argmax counts alone."""
    draft = replace_section(
        draft,
        "## Calibration under the wrong prior can be misleading",
        "## Argmax and threshold behavior",
        wrong_prior,
        "wrong-prior results",
    )

    support = """## Bootstrap, reliability, and prior sensitivity

Within-model paired bootstrap resampling and schema-v2 reliability diagrams supported the conclusion that both prior-aware vector-scaling procedures improved probability alignment relative to the uncalibrated models. None of the ten paired intervals contrasting target-weighted and OpenSpliceAI-style vector scaling excluded zero. Position-level intervals are not interpreted as training-seed uncertainty; the second independently trained checkpoint provides only a descriptive model-level replication.

Prior-sensitivity analysis provided a mechanistic explanation: as the validation non-splice weight approached the valid-position target weight, target-prior NLL and ECE improved and excessive splice argmax counts decreased. The flank-80 analyses remain a secondary context comparison rather than primary evidence."""
    draft = replace_section(
        draft,
        "## Bootstrap, reliability, and prior sensitivity",
        "## Summary of results",
        support,
        "supporting results",
    )

    summary = """## Summary of results

Across both validation-selected flank-400 models, target-weighted and OpenSpliceAI-style full-validation vector scaling sharply improved ECE, NLL, and Brier score under the valid-position target population while leaving target-weighted acceptor and donor AUPRC nearly unchanged. Fixed global scaling worsened all three probability-quality metrics, and unweighted vector fitting reduced ECE while worsening NLL and Brier. The qualitative result replicated across two independently trained models, but paired position-level intervals did not establish superiority of either prior-aware procedure."""
    draft = replace_section(
        draft,
        "## Summary of results",
        "# Discussion",
        summary,
        "results summary",
    )

    discussion = """## Detection and calibration answer different questions

The primary finding is not a new splice-site detector or a new temperature-scaling algorithm. It is that detection and calibrated probability estimation answer different questions under extreme genomic imbalance. In seed 11, uncalibrated target-weighted acceptor/donor AUPRC was 0.892620/0.915186; after target-weighted vector scaling it was 0.892953/0.914788, and after OpenSpliceAI-style scaling it was 0.893008/0.915055. Seed 23 showed the same stability, with values near 0.894 for acceptors and 0.923 for donors across the prior-aware methods, even as ECE, NLL, and Brier improved sharply.

Ranking can remain nearly unchanged while probability interpretation changes substantially. A low calibrated splice probability does not necessarily imply that a site is irrelevant for discovery; it may reflect the rarity of splice sites in the valid-position target population. Candidate discovery should therefore use target-appropriate AUPRC, top-k summaries, or validated operating thresholds rather than assuming that multiclass argmax or an arbitrary probability cutoff is appropriate."""
    draft = replace_section(
        draft,
        "## Detection and calibration answer different questions",
        "## Calibration is conditional on a target population",
        discussion,
        "discussion detection",
    )

    interpretation = """## Interpretation of the OpenSpliceAI-style comparison

The local OpenSpliceAI-style full-validation vector-temperature baseline and target-weighted true-logit vector scaling produced closely comparable proper scoring rules in both independently trained flank-400 models. None of the ten seed-by-metric paired 95% bootstrap intervals excluded zero. These results should not be described as our method beating OpenSpliceAI calibration, nor as proof that the two procedures are equivalent. The supported contribution is the explicit prior-aware comparison and valid-position evaluation framework."""
    draft = replace_section(
        draft,
        "## Interpretation of the OpenSpliceAI-style comparison",
        "## Practical implications",
        interpretation,
        "OpenSpliceAI interpretation",
    )

    limitations = """## Limitations

The main flank-400 calibration pattern replicated in a second independently trained model, reducing dependence on a single checkpoint. Nevertheless, two validation-selected training runs do not characterize the population distribution of optimization variability. The paired bootstrap unit is position and therefore does not quantify training-seed, gene, or chromosome uncertainty.

The target population covers valid sequence positions represented in the MANE gene/transcript evaluation H5 after all-zero padding is excluded. It is not complete intergenic whole-genome inference. Calibration is also not demonstrated for all annotated isoforms, noncanonical or cryptic splice sites, variant-altered sequences, or tissue-specific splice usage.

Neither configuration is a full SpliceAI-10k ensemble or a claim of state-of-the-art detection. The flank-80/flank-400 comparison does not isolate the causal effect of context length and remains secondary to the matched flank-400 analysis.

The OpenSpliceAI-style baseline was implemented locally to reproduce its class-wise vector-temperature behavior. We compare calibration protocols and objectives rather than every published OpenSpliceAI model. Finally, post-hoc calibration estimates reliability under observed labels; it does not separately quantify epistemic and aleatoric uncertainty."""
    draft = replace_section(
        draft,
        "## Limitations",
        "## Conclusion",
        limitations,
        "limitations",
    )

    conclusion = """## Conclusion

Across two independently trained focal-loss OpenSpliceAI-style flank-400 models, strong splice-site ranking did not guarantee calibrated valid-position probabilities. Target-weighted true-logit vector scaling and an OpenSpliceAI-style full-validation vector-temperature baseline sharply improved ECE, NLL, and Brier score while preserving target-weighted acceptor and donor AUPRC. Fixed global scaling worsened probability quality, and unweighted vector scaling reduced ECE while worsening proper scoring rules. Splice-site probabilities should therefore be evaluated and interpreted relative to an explicit target population, separately from detection performance. The two trained models support descriptive robustness but do not establish superiority of either prior-aware calibration procedure.

:::"""
    draft = replace_section(
        draft,
        "## Conclusion",
        "::: {.references-section}",
        conclusion,
        "conclusion",
    )

    plan = """# Figure and Table Plan

## Authoritative evidence

- Primary flank-400 model: seed 11, epoch 11.
- Independent replication: seed 23, epoch 14.
- Schema-v2 test population: 84,258 sequences and 421,290,000 raw positions.
- Valid target population: 407,278,622 positions after excluding 14,011,378 all-zero padding positions.
- Valid classes: 407,170,634 non-splice, 53,994 acceptor, and 53,994 donor positions.
- Each sampled test cache: all 107,988 positives plus 500,000 valid non-splice positions.
- Test/validation non-splice weights: 814.341268 / 177.980664.
- Bootstrap: 200 paired position-level replicates, stratified by true class, seed 17.
- Epoch-8, pre-schema-v2, single-chr9, and five-chromosome raw-position outputs are historical only.

## Main text figure

### Figure 1: Matched schema-v2 reliability diagnostics

- `figures/flank400_schema_v2_reliability_seed11.png`
- `figures/flank400_schema_v2_reliability_seed23.png`

Use both accepted combined panels. Explain that jagged rare-class bins carry little target-population mass and that proper scoring rules prevent ECE-only interpretation.

## Main text table

### Table 1: Matched schema-v2 cross-seed probability quality

Source: `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_core_point_estimates.csv`.

Report all five methods and five core metrics as seed-11/seed-23 pairs. Label AUPRC explicitly as target-weighted. Do not insert sampled-cache AUPRC near 0.999 into this table.

## Supplementary tables

### Table S1: Full 13-metric point estimates

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_point_estimates_all_metrics.csv`

### Table S2: Changes relative to uncalibrated

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_change_vs_uncalibrated.csv`

### Table S3: Paired prior-aware comparisons

- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_prior_aware_paired_comparison.csv`

State that none of ten intervals exclude zero and that position-level intervals do not estimate training-seed variability.

### Table S4: Reliability bins and provenance

- Seed-11 schema-v2 reliability output directory.
- Seed-23 schema-v2 reliability output directory.
- `results/flank400_cross_seed_robustness_schema_v2_2026-08-01/cross_seed_metadata.json`

## Operating-point material

Include only validation-selected operating points generated from valid-only schema-v2 caches. Keep sampled and target-weighted AUPRC explicitly separated.

## Retired from the main manuscript

- `figures/flank400_chromosome_transfer_cross_seed.pdf`
- `figures/flank400_primary_calibration.png`
- `figures/flank400_chr9_calibration.png`
- all epoch-8 flank-400 tables and figures
- pre-schema-v2 reliability figures and bootstrap summaries
- the five-chromosome 416,140,000 raw-position table
- single-chr9 calibration and argmax tables
- legacy sampled-cache AUPRC values when the table is labeled target-population or target-weighted
"""

    stale = [
        "416,140,000",
        "416,160,000",
        "416,053,912",
        "832.107824",
        "181.749102",
        "53,024 acceptors",
        "53,064 donors",
        "flank400_chromosome_transfer_cross_seed",
        "five-chromosome streaming",
        "AUPRC near 0.9995",
    ]
    for token in stale:
        if token in draft:
            raise SystemExit(f"ERROR: stale manuscript token remains: {token}")

    required = [
        "407,278,622",
        "14,011,378",
        "814.341268",
        "177.980664",
        "Target-weighted true-logit vector T",
        "None of the ten seed-by-metric paired 95% bootstrap intervals excluded zero",
        "figures/flank400_schema_v2_reliability_seed11.png",
        "figures/flank400_schema_v2_reliability_seed23.png",
    ]
    for token in required:
        if token not in draft:
            raise SystemExit(f"ERROR: required manuscript token missing: {token}")

    if draft.count("**Table 1.") != 1:
        raise SystemExit("ERROR: manuscript must contain exactly one Table 1 caption")
    if draft.count("**Figure 1") != 2:
        raise SystemExit("ERROR: manuscript must contain Figure 1a and Figure 1b")

    args.output_draft.write_text(draft, encoding="utf-8")
    args.output_plan.write_text(plan, encoding="utf-8")

    print("PASS: authoritative source hashes and schema-v2 census validated")
    print("PASS: 10 core rows and 10 paired comparisons validated")
    print("PASS: obsolete pre-schema-v2 population and figure claims removed")
    print("PASS: manuscript table values generated directly from authoritative CSV")
    print("PASS: original manuscript and plan were not modified")
    print(f"OUTPUT_DRAFT={args.output_draft}")
    print(f"OUTPUT_PLAN={args.output_plan}")


if __name__ == "__main__":
    main()
