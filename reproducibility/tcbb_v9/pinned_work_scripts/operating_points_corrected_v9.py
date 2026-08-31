#!/usr/bin/env python3
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

W = Path.home() / "tcbb_v9_work_2026-08-17"
R = Path.home() / "projects/calibrated-splice-prediction"
sys.path.insert(0, str(R / "scripts"))

from calibration_v2_common import load_and_validate_cache

TARGET_RECALL = 0.95
OUT = W / "operating_points_corrected_v9_results"
CLASSES = {1: "acceptor", 2: "donor"}

MODELS = {
    "seed11_epoch12": {
        "val": W / "logit_cache_flank400_seed11_epoch12_fullpopulation_validonly_v2/validation_sampled_logits.npz",
        "val_sha": "07a27bad3ee844928c4e9de9f5a7311b4c6033fab2557a4dcf03292b6a960d23",
        "test": W / "logit_cache_flank400_seed11_epoch12_fullpopulation_validonly_v2/test_sampled_logits.npz",
        "test_sha": "b01ecf561719efe057ef5775e83223b3b1aca7d16d34a33edf3df0f1673bb728",
        "scalar": W / "target_aware_scalar_corrected_checkpoints_results/seed11_epoch12_temperature.txt",
        "scalar_sha": "b511229a378f3ac1da9a8c5ab7e5240a2dfb9664d3bc3c2819a48e5e78da9b53",
        "unweighted": W / "logit_calibration_seed11_epoch12_corrected_v9_steps6000/temperature_unweighted_best.txt",
        "unweighted_sha": "8e5faf23ad3395b2cec0c5270cc3ca303caebb9039814c5ab54d343d3f8e2c16",
        "weighted": W / "logit_calibration_seed11_epoch12_corrected_v9_steps6000/temperature_target_weighted_best.txt",
        "weighted_sha": "c5d49d16580fdeb2cc1e2d18b684f4bb1c4505902a4b23a0502c3eea41c11c76",
        "osa": W / "openspliceai_style_vectorT_seed11_epoch12_corrected_v9_fullval_validonly_v2/temperature_best.txt",
        "osa_sha": "a20ad754ea5667111935e7873b16825098085a0ff56677015dcff424c522d1ec",
    },
    "seed23_epoch13": {
        "val": W / "logit_cache_flank400_seed23_epoch13_fullpopulation_validonly_v2/validation_sampled_logits.npz",
        "val_sha": "2c8240ecff47a97234073c7c60c2b4f5c2d83f38a964aa0b7058fa90b0160d50",
        "test": W / "logit_cache_flank400_seed23_epoch13_fullpopulation_validonly_v2/test_sampled_logits.npz",
        "test_sha": "f224cf31a508ee80470a5b0de7e62ef43473948dfb0928ab6685938e779a4132",
        "scalar": W / "target_aware_scalar_corrected_checkpoints_results/seed23_epoch13_temperature.txt",
        "scalar_sha": "542030f18b486cba339c6d392417d0b15c6168371363617b531f9bac1f5b8191",
        "unweighted": W / "logit_calibration_seed23_epoch13_corrected_v9_steps6000/temperature_unweighted_best.txt",
        "unweighted_sha": "4ee9b9965d5156698c80d77c130ed699ddf750e308fe21dee049f0c3359f7ab6",
        "weighted": W / "logit_calibration_seed23_epoch13_corrected_v9_steps6000/temperature_target_weighted_best.txt",
        "weighted_sha": "8baad0d64d364825a8b3176c891d3bf34a5f5ef6f26216a0a22375b9f4f7459f",
        "osa": W / "openspliceai_style_vectorT_seed23_epoch13_corrected_v9_fullval_validonly_v2/temperature_best.txt",
        "osa_sha": "f44e3e5813e45d37f0953c02ca1ce11ba694318f7b1a091149af8ccf95a93e90",
    },
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read_temperature(path, expected_sha, vector=False):
    if sha256(path) != expected_sha:
        raise RuntimeError(f"SHA mismatch: {path}")
    text = path.read_text().strip()
    if vector:
        value = np.fromstring(text.strip("[]"), sep=" ", dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"Bad vector temperature: {path}")
        return value
    return float(text)

def softmax(logits, temperature):
    t = np.asarray(temperature, dtype=np.float64)
    if t.ndim:
        t = t.reshape(1, 3)
    z = logits.astype(np.float64, copy=False) / t
    z -= z.max(axis=1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(axis=1, keepdims=True)

def select_threshold(scores, positive):
    x = np.asarray(scores[positive], dtype=np.float64)
    needed = int(np.ceil(TARGET_RECALL * len(x)))
    index = len(x) - needed
    return float(np.partition(x, index)[index])

def evaluate(scores, labels, positive_class, threshold, nonsplice_weight):
    positive = labels == positive_class
    nonsplice = labels == 0
    other_splice = (labels != 0) & ~positive
    predicted = scores >= threshold

    tp = int(np.sum(predicted & positive))
    fn = int(np.sum(~predicted & positive))
    fp_nonsplice = int(np.sum(predicted & nonsplice))
    fp_other = int(np.sum(predicted & other_splice))

    weighted_fp = fp_other + nonsplice_weight * fp_nonsplice
    sampled_den = tp + fp_other + fp_nonsplice
    target_den = tp + weighted_fp

    return {
        "tp": tp,
        "fn": fn,
        "fp_other_splice": fp_other,
        "fp_sampled_nonsplice": fp_nonsplice,
        "recall": tp / (tp + fn),
        "sampled_precision": tp / sampled_den if sampled_den else np.nan,
        "target_prior_precision": tp / target_den if target_den else np.nan,
        "estimated_target_false_positives": weighted_fp,
    }

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    metadata = {
        "target_validation_recall": TARGET_RECALL,
        "protocol": "threshold selected from validation positives only; frozen on test",
        "precision": "sampled non-splice false positives reweighted to declared valid-position target population",
        "methods": [
            "uncalibrated",
            "fixed_global_T_1.1",
            "unweighted_true_logit_vector_T",
            "target_weighted_true_logit_vector_T",
            "openspliceai_style_full_validation_vector_T_v2",
            "target_aware_fitted_scalar_T",
        ],
        "models": {},
    }

    for model, c in MODELS.items():
        val = load_and_validate_cache(c["val"])
        test = load_and_validate_cache(c["test"])

        if val.sha256 != c["val_sha"]:
            raise RuntimeError(f"{model}: validation SHA mismatch")
        if test.sha256 != c["test_sha"]:
            raise RuntimeError(f"{model}: test SHA mismatch")

        if val.class_counts.tolist() != [500000, 12955, 12955]:
            raise RuntimeError(f"{model}: validation census mismatch")
        if test.class_counts.tolist() != [500000, 53994, 53994]:
            raise RuntimeError(f"{model}: test census mismatch")

        scalar = read_temperature(c["scalar"], c["scalar_sha"])
        unweighted = read_temperature(c["unweighted"], c["unweighted_sha"], True)
        weighted = read_temperature(c["weighted"], c["weighted_sha"], True)
        osa = read_temperature(c["osa"], c["osa_sha"], True)

        methods = {
            "uncalibrated": 1.0,
            "fixed_global_T_1.1": 1.1,
            "unweighted_true_logit_vector_T": unweighted,
            "target_weighted_true_logit_vector_T": weighted,
            "openspliceai_style_full_validation_vector_T_v2": osa,
            "target_aware_fitted_scalar_T": scalar,
        }

        yv = val.labels.argmax(axis=1)
        yt = test.labels.argmax(axis=1)

        metadata["models"][model] = {
            "validation_cache_sha256": val.sha256,
            "test_cache_sha256": test.sha256,
            "validation_nonsplice_weight": float(val.negative_weight),
            "test_nonsplice_weight": float(test.negative_weight),
            "validation_valid_positions": int(val.metadata["valid_positions_seen"]),
            "test_valid_positions": int(test.metadata["valid_positions_seen"]),
        }

        for method, temperature in methods.items():
            pv = softmax(val.logits, temperature)
            pt = softmax(test.logits, temperature)

            for class_index, class_name in CLASSES.items():
                threshold = select_threshold(pv[:, class_index], yv == class_index)

                vr = evaluate(
                    pv[:, class_index], yv, class_index,
                    threshold, float(val.negative_weight)
                )
                tr = evaluate(
                    pt[:, class_index], yt, class_index,
                    threshold, float(test.negative_weight)
                )

                if vr["recall"] < TARGET_RECALL:
                    raise RuntimeError(f"{model}/{method}/{class_name}: recall target failed")

                rows.append({
                    "model": model,
                    "method": method,
                    "class": class_name,
                    "target_validation_recall": TARGET_RECALL,
                    "selected_threshold": threshold,
                    "validation_recall": vr["recall"],
                    "test_recall": tr["recall"],
                    "test_sampled_precision": tr["sampled_precision"],
                    "test_target_prior_precision": tr["target_prior_precision"],
                    "test_tp": tr["tp"],
                    "test_fn": tr["fn"],
                    "test_fp_other_splice": tr["fp_other_splice"],
                    "test_fp_sampled_nonsplice": tr["fp_sampled_nonsplice"],
                    "estimated_test_target_false_positives": tr["estimated_target_false_positives"],
                    "validation_nonsplice_weight": float(val.negative_weight),
                    "test_nonsplice_weight": float(test.negative_weight),
                    "test_valid_positions": int(test.metadata["valid_positions_seen"]),
                })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "validation_selected_95_recall_operating_points.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("PASS")
    print("rows =", len(frame))
    print("output =", OUT)
    print(frame[
        ["model", "method", "class", "selected_threshold",
         "validation_recall", "test_recall", "test_target_prior_precision"]
    ].to_string(index=False))

if __name__ == "__main__":
    main()
