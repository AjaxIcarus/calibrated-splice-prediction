from pathlib import Path
import argparse
import re
import shutil

MAIN_TABLE_FALLBACK = """| Method | Weighted multiclass ECE | Weighted multiclass NLL | Weighted acceptor ECE | Weighted donor ECE | Acceptor AUPRC | Donor AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 0.005610 | 0.006240 | 0.002737 | 0.002874 | 0.9909 | 0.9950 |
| Global T=1.1 | 0.008538 | 0.009212 | 0.004121 | 0.004413 | 0.9909 | 0.9950 |
| Logit unweighted vector T | 0.000445 | 0.010417 | 0.003115 | 0.002911 | 0.9881 | 0.9932 |
| Logit genome-weighted vector T | 0.000044 | 0.000818 | 0.000028 | 0.000032 | 0.9909 | 0.9950 |
| OpenSpliceAI-style unweighted vector T | 0.000011 | 0.000804 | 0.000015 | 0.000016 | 0.990884 | 0.994971 |
"""

BOOTSTRAP_TABLE = """| Metric | Mean | 95% CI |
|---|---:|---:|
| Weighted multiclass ECE | 0.0000193 | [0.0000107, 0.0000298] |
| Weighted NLL | 0.0008045 | [0.0007857, 0.0008262] |
| Acceptor AUPRC | 0.990866 | [0.990307, 0.991430] |
| Donor AUPRC | 0.994958 | [0.994589, 0.995332] |
"""

ABSTRACT = """Deep learning models for splice-site prediction are typically evaluated by detection performance, but their output scores are often interpreted as calibrated probabilities. This distinction is important in per-nucleotide splice-site prediction, where acceptor and donor sites are extremely rare relative to non-splice genomic positions. We evaluate probability calibration for a flank-80 OpenSpliceAI-style model trained on human GRCh38/MANE annotations using focal loss. Using true pre-softmax logits, we compare uncalibrated predictions, global temperature scaling, unweighted vector temperature scaling, genome-prior weighted vector temperature scaling, and a long-run OpenSpliceAI-style vector-temperature baseline. Class-wise vector calibration substantially reduced genome-prior weighted expected calibration error and negative log-likelihood while preserving acceptor and donor AUPRC near 0.991 and 0.995. The OpenSpliceAI-style vector-temperature baseline achieved weighted multiclass ECE of 0.000011 and weighted NLL of 0.000804 on the sampled test cache under genome-prior weighting. These results show that splice-site detection and calibrated probability estimation are separate objectives under extreme class imbalance, and that splice-site probabilities should be interpreted only relative to an explicit evaluation prior.
"""

CONTRIBUTIONS = """Our contributions are:

1. We evaluate calibration in OpenSpliceAI-style per-nucleotide splice-site prediction using true pre-softmax logits rather than probability-derived logit proxies.
2. We show that detection performance and probability calibration are separable under extreme genome-level class imbalance.
3. We compare uncalibrated predictions, global temperature scaling, unweighted vector scaling, genome-prior weighted vector scaling, and a long-run OpenSpliceAI-style vector-temperature baseline.
4. We evaluate calibration under an explicit genome-position prior using weighted ECE/NLL, reliability diagrams, bootstrap confidence intervals, prior-sensitivity analysis, and threshold precision-recall.
5. We find that class-wise true-logit vector calibration gives strong genome-prior calibration while preserving acceptor/donor AUPRC, but that calibrated probabilities remain tied to the target prior used for evaluation.
"""

RESULTS_SECTION = f"""Class-wise vector calibration improves genome-prior reliability

Table 1. Main calibration and detection results.

{MAIN_TABLE_FALLBACK}

The added OpenSpliceAI-style vector-temperature baseline was fit as class-wise true-logit temperature scaling on the validation data. This baseline achieved the strongest weighted test-cache calibration among the evaluated methods, with weighted multiclass ECE of 0.000011 and weighted NLL of 0.000804. Detection performance was preserved, with acceptor AUPRC of 0.990884 and donor AUPRC of 0.994971.

This result changes the interpretation of the study. The main contribution is not that genome-prior weighted temperature scaling outperforms OpenSpliceAI-style calibration. Instead, the contribution is a prior-aware evaluation framework showing that splice-site detection and calibrated probability estimation should be evaluated separately, and that calibrated probabilities must be interpreted relative to an explicit target prior.

Table 2. Bootstrap confidence intervals for the OpenSpliceAI-style vector-temperature baseline.

{BOOTSTRAP_TABLE}

The bootstrap results confirm that the OpenSpliceAI-style vector-temperature baseline remains well calibrated under genome-prior weighting while preserving splice-site ranking performance.
"""

LIMITATIONS_TEXT = """This study has several limitations. First, it is a controlled flank-80 study rather than a full SpliceAI-10k or full OpenSpliceAI state-of-the-art analysis. Second, the evaluation uses sampled negative positions with genome-prior weighting rather than exhaustive full-genome inference. Third, the OpenSpliceAI-style calibration baseline was implemented locally to match the package's class-wise vector-temperature behavior in this environment. Fourth, the analysis is based on one focal-loss checkpoint, so future work should test longer contexts, more checkpoints or random seeds, larger negative samples, full-chromosome evaluation, and variant-effect calibration.
"""

def replace_first(pattern, repl, text, flags=re.S | re.I):
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    return new_text, n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="paper_draft/full_draft_current.md")
    parser.add_argument("--output", default="paper_draft/full_draft_v16_openspliceai_revised.md")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input draft not found: {in_path}")

    text = in_path.read_text(encoding="utf-8")

    backup = in_path.with_suffix(in_path.suffix + ".bak_before_openspliceai_revision")
    if not backup.exists():
        shutil.copy2(in_path, backup)

    report = []

    # Title replacement.
    text, n = replace_first(
        r"Prior-dependent calibration for OpenSpliceAI[\s\S]*?extreme class\s+imbalance",
        "Prior-aware calibration of OpenSpliceAI-style splice-site prediction under genome-level class imbalance",
        text,
    )
    report.append(("title", n))

    # Abstract replacement: ABSTRACT ... Introduction
    text, n = replace_first(
        r"(ABSTRACT\s*\n)(.*?)(\n\s*Introduction\s*\n)",
        r"\1" + ABSTRACT.strip() + r"\3",
        text,
    )
    report.append(("abstract", n))

    # Contribution replacement: Our contributions are: ... Related Work
    text, n = replace_first(
        r"(Our contributions are:\s*\n)(.*?)(\n\s*Related Work\s*\n)",
        CONTRIBUTIONS.strip() + r"\3",
        text,
    )
    report.append(("contributions", n))

    # Main results replacement.
    # This targets the old subsection beginning with genome-prior weighted vector scaling.
    text, n = replace_first(
        r"Genome-prior weighted vector scaling\s+improves calibration[\s\S]*?(?=\n\s*Figure 1A\.|\n\s*Figure 1|\n\s*Reliability|\n\s*Prior-sensitivity|\n\s*Bootstrap|\n\s*Discussion|\Z)",
        RESULTS_SECTION.strip() + "\n\n",
        text,
    )
    report.append(("main_results_section", n))

    # If the exact old heading was not found, insert before first Figure 1A.
    if n == 0:
        text, n2 = replace_first(
            r"(?=\n\s*Figure 1A\.)",
            "\n" + RESULTS_SECTION.strip() + "\n\n",
            text,
        )
        report.append(("main_results_insert_before_figure1A", n2))

    # Limitations replacement or insertion.
    text, n = replace_first(
        r"(Limitations\s*\n)(.*?)(?=\n\s*Conclusion|\n\s*Reproducibility|\n\s*References|\Z)",
        r"\1" + LIMITATIONS_TEXT.strip() + "\n\n",
        text,
    )
    report.append(("limitations", n))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"Wrote revised draft: {out_path}")
    print(f"Backup of original input: {backup}")
    print("\nPatch report:")
    for name, count in report:
        print(f"  {name}: {count}")

    print("\nSanity checks:")
    for phrase in [
        "OpenSpliceAI-style unweighted vector T",
        "0.000011",
        "0.000804",
        "prior-aware evaluation framework",
    ]:
        print(f"  {phrase!r}: {'YES' if phrase in text else 'NO'}")

if __name__ == "__main__":
    main()
