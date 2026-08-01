#!/usr/bin/env python3

import os
import argparse
import random
import numpy as np
import h5py
import torch

# Compatibility patch for newer PyTorch versions where
# ReduceLROnPlateau no longer accepts verbose=...
_orig_reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau

def _reduce_lr_no_verbose(*args, **kwargs):
    kwargs.pop("verbose", None)
    return _orig_reduce_lr(*args, **kwargs)

torch.optim.lr_scheduler.ReduceLROnPlateau = _reduce_lr_no_verbose


from openspliceai.calibrate.temperature_scaling import (
    ModelWithTemperature,
    get_validation_loader,
)
from openspliceai.calibrate.model_utils import initialize_model_and_optim
from openspliceai.train_base.utils import setup_environment


def choose_shards(h5f, n_shards=None, seed=42):
    shard_ids = sorted(
        int(k[1:]) for k in h5f.keys()
        if k.startswith("X") and k[1:].isdigit()
    )
    if n_shards is not None and n_shards < len(shard_ids):
        rng = random.Random(seed)
        shard_ids = sorted(rng.sample(shard_ids, n_shards))
    return shard_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-validation-shards", type=int, default=None)
    parser.add_argument("--max-test-shards", type=int, default=None)
    parser.add_argument("--loss", default="focal_loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Make args compatible with OpenSpliceAI setup_environment.
    args.train_dataset = args.validation_dataset
    args.test_dataset = args.test_dataset
    args.project_name = "openspliceai_builtin_calibration_fixed"
    args.exp_num = 0
    args.epochs = 10
    args.early_stopping = False
    args.patience = 2
    args.temperature_file = None

    device = setup_environment(args)

    print("Initializing model...")
    model, model_params = initialize_model_and_optim(
        device,
        args.flanking_size,
        args.pretrained_model,
    )

    valid_h5f = h5py.File(args.validation_dataset, "r")
    test_h5f = h5py.File(args.test_dataset, "r")

    val_idxs = choose_shards(valid_h5f, args.max_validation_shards, args.random_seed)
    test_idxs = choose_shards(test_h5f, args.max_test_shards, args.random_seed)

    print("Validation shards:", len(val_idxs), val_idxs[:10], "...")
    print("Test shards:", len(test_idxs), test_idxs[:10], "...")

    print("Building validation loader...")
    validation_loader = get_validation_loader(
        valid_h5f,
        val_idxs,
        model_params["BATCH_SIZE"],
    )

    calibrated_model = ModelWithTemperature(model, num_classes=3)
    print("Fitting OpenSpliceAI built-in vector temperature...")
    calibrated_model.set_temperature(validation_loader, model_params)

    temperature_path = os.path.join(args.output_dir, "temperature.pt")
    temperature_txt_path = os.path.join(args.output_dir, "temperature.txt")

    calibrated_model.save_temperature(temperature_path)

    temp = calibrated_model.temperature.detach().cpu().numpy()
    with open(temperature_txt_path, "w") as f:
        f.write(str(temp))

    print("Saved temperature to:", temperature_path)
    print("Temperature:", temp)

    valid_h5f.close()
    test_h5f.close()


if __name__ == "__main__":
    main()
