import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from openspliceai.train.train import initialize_model_and_optim
from openspliceai.train_base.utils import valid_epoch, create_metric_files


def get_h5_indices(h5f):
    x_keys = [k for k in h5f.keys() if k.startswith("X")]
    idxs = sorted([int(k[1:]) for k in x_keys])
    return np.array(idxs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-name", default="VAL")
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--loss", default="focal_loss", choices=["focal_loss", "cross_entropy_loss"])
    parser.add_argument("--random-seed", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=args.flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    print("Loading checkpoint:", args.model)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Checkpoint loaded.")

    params["RANDOM_SEED"] = args.random_seed

    metric_files = create_metric_files(str(out_dir))

    with h5py.File(args.dataset, "r") as h5f:
        idxs = get_h5_indices(h5f)
        print(f"{args.split_name} idxs count:", len(idxs))
        print(f"{args.split_name} idxs:", idxs)

        loss = valid_epoch(
            model,
            h5f,
            idxs,
            params["BATCH_SIZE"],
            args.loss,
            device,
            params,
            metric_files,
            args.flanking_size,
            args.split_name.lower(),
        )

    print(f"{args.split_name} loss:", loss)


if __name__ == "__main__":
    main()