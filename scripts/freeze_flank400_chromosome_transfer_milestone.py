#!/usr/bin/env python3
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

DATE = "2026-07-24"
CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]
SEEDS = ["seed11_epoch11", "seed23_epoch14"]
SUFFIXES = [
    "by_chromosome.csv",
    "ece_reconstruction_check.csv",
    "pooled_reliability_bins.csv",
    "summary_with_pooled_ece.csv",
    "relative_changes.csv",
]

note_path = Path(
    f"notes/flank400_chromosome_transfer_milestone_{DATE}.md"
)
manifest_path = Path(
    f"results/flank400_chromosome_transfer_milestone_{DATE}_manifest.csv"
)
marker_path = Path(
    f"results/FLANK400_CHROMOSOME_TRANSFER_FROZEN_{DATE}.txt"
)

for path in [note_path, manifest_path, marker_path]:
    if path.exists():
        raise SystemExit(f"Refusing to overwrite frozen artifact: {path}")

robustness_path = Path(
    "results/"
    "flank400_chromosome_transfer_cross_seed_method_robustness.csv"
)
observed = pd.read_csv(robustness_path)

expected = pd.DataFrame(
    [
        ("seed11_epoch11", "uncalibrated",
         0.001507295858, 0.001698358538),
        ("seed11_epoch11", "fixed_global_T1p1",
         0.002519030113, 0.002716969665),
        ("seed11_epoch11", "unweighted_vectorT",
         0.000046921955, 0.002062224107),
        ("seed11_epoch11", "genome_weighted_vectorT",
         0.000020870360, 0.000252214137),
        ("seed11_epoch11", "openspliceai_style_vectorT",
         0.000004553865, 0.000244193205),
        ("seed23_epoch14", "uncalibrated",
         0.001544696386, 0.001725679388),
        ("seed23_epoch14", "fixed_global_T1p1",
         0.002586726680, 0.002775464228),
        ("seed23_epoch14", "unweighted_vectorT",
         0.000050493954, 0.001974532575),
        ("seed23_epoch14", "genome_weighted_vectorT",
         0.000027388731, 0.000240350851),
        ("seed23_epoch14", "openspliceai_style_vectorT",
         0.000003084788, 0.000230533274),
    ],
    columns=[
        "model",
        "method",
        "pooled_multiclass_ece_expected",
        "pooled_multiclass_nll_expected",
    ],
)

required_columns = {
    "model",
    "method",
    "total_positions",
    "pooled_multiclass_ece",
    "pooled_multiclass_nll",
}
if not required_columns.issubset(observed.columns):
    raise SystemExit("Cross-seed robustness schema is invalid")

if observed.duplicated(["model", "method"]).any():
    raise SystemExit("Duplicate model/method rows detected")

merged = observed.merge(
    expected,
    on=["model", "method"],
    how="outer",
    indicator=True,
    validate="one_to_one",
)

if not (merged["_merge"] == "both").all():
    raise SystemExit("Unexpected or missing model/method rows")

if not (
    pd.to_numeric(merged["total_positions"]).astype(int)
    == 416_140_000
).all():
    raise SystemExit("Unexpected position count")

for metric in ["pooled_multiclass_ece", "pooled_multiclass_nll"]:
    error = np.abs(
        pd.to_numeric(merged[metric])
        - pd.to_numeric(merged[f"{metric}_expected"])
    )
    if error.max() > 1e-12:
        raise SystemExit(
            f"{metric} differs from frozen value: "
            f"maximum error={error.max():.17g}"
        )

required = [
    Path("scripts/aggregate_flank400_chromosome_transfer.py"),
    Path("scripts/make_flank400_chromosome_transfer_cross_seed.py"),
    Path(
        "scripts/freeze_flank400_chromosome_transfer_milestone.py"
    ),
    Path(
        "logs/"
        "aggregate_flank400_chromosome_transfer_seed11_epoch11_"
        "recomputed.log"
    ),
    Path(
        "logs/"
        "aggregate_flank400_chromosome_transfer_seed23_epoch14.log"
    ),
    Path(
        "logs/"
        "make_flank400_chromosome_transfer_cross_seed.log"
    ),
]

for seed in SEEDS:
    prefix = (
        f"results/flank400_chromosome_transfer_{seed}"
        + ("_recomputed" if seed == "seed11_epoch11" else "")
    )
    for suffix in SUFFIXES:
        required.append(Path(f"{prefix}_{suffix}"))

for suffix in [
    "method_robustness.csv",
    "pooled_comparison.csv",
    "paired_by_chromosome.csv",
    "paired_summary.csv",
    "key_contrasts.csv",
]:
    required.append(
        Path(
            "results/"
            f"flank400_chromosome_transfer_cross_seed_{suffix}"
        )
    )

for seed in SEEDS:
    for chrom in CHROMS:
        directory = Path(
            f"results/chromosome_eval_flank400_{seed}_{chrom}"
        )
        required.extend(
            [
                directory / f"{chrom}_streaming_metrics.csv",
                directory / f"{chrom}_multiclass_reliability_bins.csv",
            ]
        )

missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(
        "Missing milestone artifacts:\n  " + "\n  ".join(missing)
    )

note_path.parent.mkdir(parents=True, exist_ok=True)
note_path.write_text(
    """# Flank-400 chromosome-transfer milestone

Frozen: 2026-07-24

## Authoritative models

- Primary: seed 11, epoch 11
- Replication: seed 23, epoch 14
- Epoch 8 is historical and excluded.

## Evaluation scope

Chromosomes: chr1, chr3, chr5, chr7, chr9

Each model was evaluated over 416,140,000 positions using the same
aggregation and pooled-ECE implementation.

## Pooled results

| Model | Method | ECE | NLL |
|---|---|---:|---:|
| seed11_epoch11 | Uncalibrated | 0.001507295858 | 0.001698358538 |
| seed11_epoch11 | Fixed global T=1.1 | 0.002519030113 | 0.002716969665 |
| seed11_epoch11 | Unweighted vector-T | 0.000046921955 | 0.002062224107 |
| seed11_epoch11 | Genome-weighted vector-T | 0.000020870360 | 0.000252214137 |
| seed11_epoch11 | OSAI-style vector-T | 0.000004553865 | 0.000244193205 |
| seed23_epoch14 | Uncalibrated | 0.001544696386 | 0.001725679388 |
| seed23_epoch14 | Fixed global T=1.1 | 0.002586726680 | 0.002775464228 |
| seed23_epoch14 | Unweighted vector-T | 0.000050493954 | 0.001974532575 |
| seed23_epoch14 | Genome-weighted vector-T | 0.000027388731 | 0.000240350851 |
| seed23_epoch14 | OSAI-style vector-T | 0.000003084788 | 0.000230533274 |

## Frozen conclusion

The qualitative result replicated across both models. OSAI-style
class-wise vector temperature scaling was best on pooled ECE and NLL.
Genome-weighted vector-T also improved both metrics. Unweighted
vector-T improved ECE but worsened NLL, while fixed T=1.1 worsened
both metrics.

This is a five-chromosome transfer result, not a claim of complete
genome-wide evaluation or an inferential chromosome-level test.
"""
)

required.append(note_path)

rows = []
for path in sorted(set(required), key=lambda item: str(item)):
    digest = sha256(path.read_bytes()).hexdigest()
    rows.append(
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        }
    )

manifest = pd.DataFrame(rows)
manifest.to_csv(manifest_path, index=False)

manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
marker_path.write_text(
    "PASS: flank-400 chromosome-transfer milestone frozen\n"
    f"date={DATE}\n"
    "primary=seed11_epoch11\n"
    "replication=seed23_epoch14\n"
    "positions_per_model=416140000\n"
    f"manifest={manifest_path}\n"
    f"manifest_sha256={manifest_digest}\n"
)

print("PASS: frozen chromosome-transfer milestone validated")
print(f"PASS: recorded {len(manifest):,} hashed artifacts")
print(f"Note:     {note_path}")
print(f"Manifest: {manifest_path}")
print(f"Marker:   {marker_path}")
print(f"Manifest SHA-256: {manifest_digest}")
