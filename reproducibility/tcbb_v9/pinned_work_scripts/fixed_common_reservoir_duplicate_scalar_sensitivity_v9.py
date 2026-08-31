#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import csv
import sys
import numpy as np

W = Path.home() / "tcbb_v9_work_2026-08-17"
R = Path.home() / "projects/calibrated-splice-prediction"
sys.path[:0] = [str(W), str(R / "scripts")]

from calibration_v2_common import load_and_validate_cache as load_full
from calibration_v2_common_duplicate_sensitivity_v9 import (
    load_and_validate_cache as load_excluded,
)
from fit_target_aware_scalar_temperature_corrected_checkpoints_v9 import (
    fit_scalar,
    evaluate,
)

CFG = {
    "seed11_epoch12": {
        "original": W / "logit_cache_flank400_seed11_epoch12_fullpopulation_validonly_v2/validation_sampled_logits.npz",
        "original_sha": "07a27bad3ee844928c4e9de9f5a7311b4c6033fab2557a4dcf03292b6a960d23",
        "excluded": W / "logit_cache_flank400_seed11_epoch12_duplicate_excluded_validation_v9_retry1/validation_sampled_logits.npz",
        "excluded_sha": "825c428798b94f612cd1fb50c4fa7391d763b53f69ba391d213d84b9b8623efa",
        "test": W / "logit_cache_flank400_seed11_epoch12_fullpopulation_validonly_v2/test_sampled_logits.npz",
        "test_sha": "b01ecf561719efe057ef5775e83223b3b1aca7d16d34a33edf3df0f1673bb728",
    },
    "seed23_epoch13": {
        "original": W / "logit_cache_flank400_seed23_epoch13_fullpopulation_validonly_v2/validation_sampled_logits.npz",
        "original_sha": "2c8240ecff47a97234073c7c60c2b4f5c2d83f38a964aa0b7058fa90b0160d50",
        "excluded": W / "logit_cache_flank400_seed23_epoch13_duplicate_excluded_validation_v9_retry1/validation_sampled_logits.npz",
        "excluded_sha": "cd5704ab529042abec94f3416c319bb299b336aacc2dcee81e6fee203aa148bc",
        "test": W / "logit_cache_flank400_seed23_epoch13_fullpopulation_validonly_v2/test_sampled_logits.npz",
        "test_sha": "f224cf31a508ee80470a5b0de7e62ef43473948dfb0928ab6685938e779a4132",
    },
}

def common_negative_logits(a, b):
    xa = np.ascontiguousarray(a.logits[a.labels.argmax(1) == 0])
    xb = np.ascontiguousarray(b.logits[b.labels.argmax(1) == 0])
    dt = np.dtype((np.void, xa.dtype.itemsize * xa.shape[1]))
    va, vb = xa.view(dt).ravel(), xb.view(dt).ravel()
    ua, ca = np.unique(va, return_counts=True)
    ub, cb = np.unique(vb, return_counts=True)
    common, ia, ib = np.intersect1d(
        ua, ub, assume_unique=True, return_indices=True
    )
    counts = np.minimum(ca[ia], cb[ib])
    rows = common.view(xa.dtype).reshape(-1, 3)
    return np.repeat(rows, counts, axis=0)

def make_arm(neg, source, target_nonsplice):
    mask = source.labels.argmax(1) != 0
    pos_logits = source.logits[mask]
    pos_labels = source.labels[mask]
    neg_labels = np.zeros((len(neg), 3), dtype=pos_labels.dtype)
    neg_labels[:, 0] = 1
    return SimpleNamespace(
        logits=np.concatenate([neg, pos_logits]),
        labels=np.concatenate([neg_labels, pos_labels]),
        negative_weight=float(target_nonsplice) / len(neg),
    )

out = W / "fixed_common_reservoir_duplicate_scalar_sensitivity_v9"
if out.exists():
    raise RuntimeError(f"output already exists: {out}")
out.mkdir()

rows = []
for name, c in CFG.items():
    original = load_full(c["original"])
    excluded = load_excluded(c["excluded"])
    test = load_full(c["test"])

    for obj, key in (
        (original, "original_sha"),
        (excluded, "excluded_sha"),
        (test, "test_sha"),
    ):
        if obj.sha256 != c[key]:
            raise RuntimeError(f"{name}: SHA mismatch for {key}")

    neg = common_negative_logits(original, excluded)
    if len(neg) != 94414:
        raise RuntimeError(f"{name}: common negatives={len(neg)}, expected 94414")

    arms = (
        ("original", original, 88990332, 25910),
        ("duplicate_excluded", excluded, 88982074, 25902),
    )

    for condition, source, target_nonsplice, expected_pos in arms:
        arm = make_arm(neg, source, target_nonsplice)
        positives = int((arm.labels.argmax(1) != 0).sum())
        if positives != expected_pos:
            raise RuntimeError(f"{name}/{condition}: positives={positives}")

        temperature, validation_nll, _ = fit_scalar(arm)
        metrics = evaluate(test, temperature)

        row = {
            "model": name,
            "condition": condition,
            "common_negatives": len(neg),
            "validation_positives": positives,
            "target_nonsplice": target_nonsplice,
            "negative_weight": arm.negative_weight,
            "temperature": temperature,
            "validation_nll": validation_nll,
            "test_ece": metrics["multiclass_ece"],
            "test_nll": metrics["multiclass_nll"],
            "test_brier": metrics["multiclass_brier"],
            "target_acceptor_auprc": metrics["target_weighted_acceptor_auprc"],
            "target_donor_auprc": metrics["target_weighted_donor_auprc"],
        }
        rows.append(row)
        print(
            name, condition,
            f"T={temperature:.12g}",
            f"valNLL={validation_nll:.12g}",
            f"testNLL={metrics['multiclass_nll']:.12g}",
        )

with (out / "fixed_common_reservoir_scalar_results.csv").open(
    "w", newline=""
) as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("PASS")
print(out / "fixed_common_reservoir_scalar_results.csv")
