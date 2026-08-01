import argparse
from pathlib import Path
import pickle

import h5py
import numpy as np
import torch

from openspliceai.train.train import initialize_model_and_optim
from openspliceai.train_base.utils import load_data_from_shard, clip_datapoints
from openspliceai.constants import CL_max


def get_h5_indices(h5f):
    x_keys = [k for k in h5f.keys() if k.startswith("X")]
    return np.array(sorted([int(k[1:]) for k in x_keys]))


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    tmp.replace(path)
    print(f"Saved checkpoint: {path}", flush=True)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--max-negatives", type=int, default=1000000)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / f"{args.split_name}_cache_checkpoint_{args.max_negatives}_negatives.pkl"
    out_npz = out_dir / f"{args.split_name}_sampled_logits_{args.max_negatives}_negatives.npz"

    rng = np.random.default_rng(args.random_seed)

    pos_logits = []
    pos_probs = []
    pos_labels = []

    neg_logits_reservoir = np.empty((args.max_negatives, 3), dtype=np.float32)
    neg_probs_reservoir = np.empty((args.max_negatives, 3), dtype=np.float32)
    neg_labels_reservoir = np.empty((args.max_negatives, 3), dtype=np.float32)

    negatives_seen_total = 0
    negatives_kept = 0
    total_positions_seen = 0
    positive_positions_seen = 0
    completed_shards = set()

    if args.resume and ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        state = load_checkpoint(ckpt_path)

        pos_logits = state["pos_logits"]
        pos_probs = state["pos_probs"]
        pos_labels = state["pos_labels"]

        neg_logits_reservoir = state["neg_logits_reservoir"]
        neg_probs_reservoir = state["neg_probs_reservoir"]
        neg_labels_reservoir = state["neg_labels_reservoir"]

        negatives_seen_total = state["negatives_seen_total"]
        negatives_kept = state["negatives_kept"]
        total_positions_seen = state["total_positions_seen"]
        positive_positions_seen = state["positive_positions_seen"]
        completed_shards = set(state["completed_shards"])

        rng.bit_generator.state = state["rng_state"]

        print("Resume state:", flush=True)
        print("  completed_shards:", sorted(completed_shards), flush=True)
        print("  total_positions_seen:", total_positions_seen, flush=True)
        print("  positive_positions_seen:", positive_positions_seen, flush=True)
        print("  negatives_seen_total:", negatives_seen_total, flush=True)
        print("  negatives_kept:", negatives_kept, flush=True)
    elif args.resume:
        print("Resume requested, but no checkpoint found. Starting fresh.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=args.flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    print("Loading checkpoint:", args.model, flush=True)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if hasattr(model, "apply_softmax"):
        model.apply_softmax = False
        print("Set model.apply_softmax = False", flush=True)
    else:
        print("WARNING: model has no apply_softmax attribute.", flush=True)

    params["RANDOM_SEED"] = args.random_seed

    with h5py.File(args.dataset, "r") as h5f:
        idxs = get_h5_indices(h5f)
        print("Dataset shards:", idxs, flush=True)

        with torch.no_grad():
            for shard_idx in idxs:
                if int(shard_idx) in completed_shards:
                    print(f"Skipping completed shard {shard_idx}", flush=True)
                    continue

                print(f"\nStarting shard {shard_idx}", flush=True)

                loader = load_data_from_shard(
                    h5f,
                    shard_idx,
                    device,
                    params["BATCH_SIZE"],
                    params,
                    shuffle=False,
                )

                for batch in loader:
                    DNAs, labels = batch[0].to(device), batch[1].to(device)

                    DNAs, labels = clip_datapoints(
                        DNAs,
                        labels,
                        params["CL"],
                        CL_max,
                        params["N_GPUS"],
                    )

                    DNAs = DNAs.to(torch.float32).to(device)
                    labels = labels.to(torch.float32).to(device)

                    logits = model(DNAs)
                    probs = torch.softmax(logits, dim=1)

                    logits_np = logits.detach().cpu().numpy()
                    probs_np = probs.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    logits_flat = np.transpose(logits_np, (0, 2, 1)).reshape(-1, 3)
                    probs_flat = np.transpose(probs_np, (0, 2, 1)).reshape(-1, 3)
                    labels_flat = np.transpose(labels_np, (0, 2, 1)).reshape(-1, 3)

                    total_positions_seen += len(labels_flat)

                    y_true = np.argmax(labels_flat, axis=1)
                    pos_mask = y_true != 0
                    neg_mask = y_true == 0

                    if pos_mask.any():
                        pos_logits.append(logits_flat[pos_mask].astype(np.float32))
                        pos_probs.append(probs_flat[pos_mask].astype(np.float32))
                        pos_labels.append(labels_flat[pos_mask].astype(np.float32))
                        positive_positions_seen += int(pos_mask.sum())

                    neg_indices = np.where(neg_mask)[0]

                    if len(neg_indices) > 0:
                        n_neg = len(neg_indices)

                        if negatives_kept < args.max_negatives:
                            fill_n = min(args.max_negatives - negatives_kept, n_neg)
                            fill_indices = neg_indices[:fill_n]

                            neg_logits_reservoir[
                                negatives_kept:negatives_kept + fill_n
                            ] = logits_flat[fill_indices].astype(np.float32)

                            neg_probs_reservoir[
                                negatives_kept:negatives_kept + fill_n
                            ] = probs_flat[fill_indices].astype(np.float32)

                            neg_labels_reservoir[
                                negatives_kept:negatives_kept + fill_n
                            ] = labels_flat[fill_indices].astype(np.float32)

                            negatives_kept += fill_n
                            negatives_seen_total += fill_n

                            neg_indices = neg_indices[fill_n:]
                            n_neg = len(neg_indices)

                        if n_neg > 0:
                            global_positions = np.arange(
                                negatives_seen_total,
                                negatives_seen_total + n_neg,
                                dtype=np.int64,
                            )

                            slots = rng.integers(0, global_positions + 1)
                            replace_mask = slots < args.max_negatives

                            if replace_mask.any():
                                replace_slots = slots[replace_mask]
                                replace_indices = neg_indices[replace_mask]

                                neg_logits_reservoir[replace_slots] = logits_flat[
                                    replace_indices
                                ].astype(np.float32)

                                neg_probs_reservoir[replace_slots] = probs_flat[
                                    replace_indices
                                ].astype(np.float32)

                                neg_labels_reservoir[replace_slots] = labels_flat[
                                    replace_indices
                                ].astype(np.float32)

                            negatives_seen_total += n_neg

                completed_shards.add(int(shard_idx))

                save_checkpoint(
                    ckpt_path,
                    {
                        "pos_logits": pos_logits,
                        "pos_probs": pos_probs,
                        "pos_labels": pos_labels,
                        "neg_logits_reservoir": neg_logits_reservoir,
                        "neg_probs_reservoir": neg_probs_reservoir,
                        "neg_labels_reservoir": neg_labels_reservoir,
                        "negatives_seen_total": negatives_seen_total,
                        "negatives_kept": negatives_kept,
                        "total_positions_seen": total_positions_seen,
                        "positive_positions_seen": positive_positions_seen,
                        "completed_shards": sorted(completed_shards),
                        "rng_state": rng.bit_generator.state,
                    },
                )

                print(f"Completed shard {shard_idx}", flush=True)
                print("  total_positions_seen:", total_positions_seen, flush=True)
                print("  positive_positions_seen:", positive_positions_seen, flush=True)
                print("  negatives_seen_total:", negatives_seen_total, flush=True)
                print("  negatives_kept:", negatives_kept, flush=True)

    if not pos_logits:
        raise RuntimeError("No positive splice positions found.")

    pos_logits = np.concatenate(pos_logits, axis=0)
    pos_probs = np.concatenate(pos_probs, axis=0)
    pos_labels = np.concatenate(pos_labels, axis=0)

    neg_logits = neg_logits_reservoir[:negatives_kept]
    neg_probs = neg_probs_reservoir[:negatives_kept]
    neg_labels = neg_labels_reservoir[:negatives_kept]

    logits_sample = np.concatenate([pos_logits, neg_logits], axis=0)
    probs_sample = np.concatenate([pos_probs, neg_probs], axis=0)
    labels_sample = np.concatenate([pos_labels, neg_labels], axis=0)

    perm = rng.permutation(len(labels_sample))
    logits_sample = logits_sample[perm]
    probs_sample = probs_sample[perm]
    labels_sample = labels_sample[perm]

    max_abs_diff = float(np.max(np.abs(softmax_np(logits_sample) - probs_sample)))
    row_sum_error = float(np.max(np.abs(probs_sample.sum(axis=1) - 1.0)))

    np.savez_compressed(
        out_npz,
        logits=logits_sample,
        probs=probs_sample,
        labels=labels_sample,
        split_name=np.array([args.split_name]),
        max_negatives=np.array([args.max_negatives], dtype=np.int64),
        sampled_negatives=np.array([negatives_kept], dtype=np.int64),
        negatives_seen_total=np.array([negatives_seen_total], dtype=np.int64),
        total_positions_seen=np.array([total_positions_seen], dtype=np.int64),
        positive_positions_seen=np.array([positive_positions_seen], dtype=np.int64),
        random_seed=np.array([args.random_seed], dtype=np.int64),
        flanking_size=np.array([args.flanking_size], dtype=np.int64),
        max_abs_diff_softmax_logits_vs_probs=np.array([max_abs_diff], dtype=np.float64),
        prob_row_sum_max_error=np.array([row_sum_error], dtype=np.float64),
    )

    print("\nWrote:", out_npz, flush=True)
    print("Total positions seen:", total_positions_seen, flush=True)
    print("Positive positions:", len(pos_logits), flush=True)
    print("Positive positions seen:", positive_positions_seen, flush=True)
    print("Total negative positions seen:", negatives_seen_total, flush=True)
    print("Sampled negative positions:", negatives_kept, flush=True)
    print("Total sampled positions:", len(labels_sample), flush=True)
    print("max_abs_diff softmax(logits) vs probs:", max_abs_diff, flush=True)
    print("prob row-sum max error:", row_sum_error, flush=True)


if __name__ == "__main__":
    main()
