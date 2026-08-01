import argparse
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch

from openspliceai.train.train import initialize_model_and_optim
from openspliceai.train_base.utils import load_data_from_shard, clip_datapoints
from openspliceai.constants import CL_max


def decode_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.bytes_):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray) and x.shape == ():
        return decode_str(x.item())
    return str(x)


def get_h5_indices(h5f):
    x_keys = [k for k in h5f.keys() if k.startswith("X")]
    return sorted([int(k[1:]) for k in x_keys])


def stable_softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def object_length(dset, i):
    x = dset[i]
    if isinstance(x, (bytes, np.bytes_)):
        return len(decode_str(x))
    arr = np.asarray(x)
    if arr.shape == ():
        val = arr.item()
        if isinstance(val, (bytes, np.bytes_, str)):
            return len(decode_str(val))
        return 1
    return arr.shape[0]


def infer_gene_to_shard_map(datafile_h5, dataset_h5, chunk_len=5000):
    """
    Infer mapping:
      gene index -> dataset shard index and segment offset.

    This assumes datafile order and dataset shard order match.
    It validates by matching summed per-gene segment counts to Y shard sizes.
    """
    with h5py.File(datafile_h5, "r") as df, h5py.File(dataset_h5, "r") as ds:
        y_counts = []
        shard_ids = get_h5_indices(ds)
        for sid in shard_ids:
            y_counts.append(int(ds[f"Y{sid}"].shape[1]))

        candidates = []

        for key in ["SEQ", "LABEL"]:
            if key not in df:
                continue

            lengths = [object_length(df[key], i) for i in range(len(df["CHROM"]))]
            seg_counts = [int(np.ceil(L / chunk_len)) for L in lengths]

            mapping = []
            gene_i = 0
            ok = True

            for sid, target_segments in zip(shard_ids, y_counts):
                offset = 0
                while gene_i < len(seg_counts) and offset < target_segments:
                    nseg = seg_counts[gene_i]
                    mapping.append(
                        {
                            "gene_index": gene_i,
                            "shard": sid,
                            "segment_offset": offset,
                            "n_segments": nseg,
                        }
                    )
                    offset += nseg
                    gene_i += 1

                if offset != target_segments:
                    ok = False
                    break

            if ok and gene_i == len(seg_counts):
                candidates.append((key, mapping, seg_counts))

        if not candidates:
            raise RuntimeError(
                "Could not infer gene-to-shard mapping from SEQ or LABEL lengths. "
                "Need to inspect OpenSpliceAI dataset construction."
            )

        key, mapping, seg_counts = candidates[0]
        print(f"Using mapping inferred from {key} lengths.", flush=True)
        return mapping, seg_counts


class MulticlassCalib:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.edges = np.linspace(0, 1, n_bins + 1)
        self.count = np.zeros(n_bins, dtype=np.float64)
        self.sum_conf = np.zeros(n_bins, dtype=np.float64)
        self.sum_correct = np.zeros(n_bins, dtype=np.float64)
        self.nll_sum = 0.0
        self.n = 0

    def update(self, probs, y):
        eps = 1e-12
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == y).astype(np.float64)

        b = np.searchsorted(self.edges, conf, side="right") - 1
        b = np.clip(b, 0, self.n_bins - 1)

        self.count += np.bincount(b, minlength=self.n_bins)
        self.sum_conf += np.bincount(b, weights=conf, minlength=self.n_bins)
        self.sum_correct += np.bincount(b, weights=correct, minlength=self.n_bins)

        self.nll_sum += float(np.sum(-np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))))
        self.n += len(y)

    def metrics(self):
        mask = self.count > 0
        avg_conf = np.zeros_like(self.count)
        avg_acc = np.zeros_like(self.count)
        avg_conf[mask] = self.sum_conf[mask] / self.count[mask]
        avg_acc[mask] = self.sum_correct[mask] / self.count[mask]
        ece = np.sum((self.count[mask] / self.n) * np.abs(avg_acc[mask] - avg_conf[mask]))
        return float(ece), float(self.nll_sum / self.n)

    def bins_df(self, method):
        rows = []
        for i in range(self.n_bins):
            if self.count[i] == 0:
                continue
            mean_conf = self.sum_conf[i] / self.count[i]
            obs_acc = self.sum_correct[i] / self.count[i]
            rows.append(
                {
                    "method": method,
                    "bin": i,
                    "bin_low": self.edges[i],
                    "bin_high": self.edges[i + 1],
                    "count": int(self.count[i]),
                    "mean_confidence": mean_conf,
                    "observed_accuracy": obs_acc,
                    "gap": obs_acc - mean_conf,
                }
            )
        return pd.DataFrame(rows)


class BinaryCalib:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.edges = np.linspace(0, 1, n_bins + 1)
        self.count = np.zeros(n_bins, dtype=np.float64)
        self.sum_p = np.zeros(n_bins, dtype=np.float64)
        self.sum_y = np.zeros(n_bins, dtype=np.float64)

    def update(self, p, y_bin):
        b = np.searchsorted(self.edges, p, side="right") - 1
        b = np.clip(b, 0, self.n_bins - 1)

        self.count += np.bincount(b, minlength=self.n_bins)
        self.sum_p += np.bincount(b, weights=p, minlength=self.n_bins)
        self.sum_y += np.bincount(b, weights=y_bin, minlength=self.n_bins)

    def ece(self):
        total = self.count.sum()
        mask = self.count > 0
        mean_p = self.sum_p[mask] / self.count[mask]
        obs = self.sum_y[mask] / self.count[mask]
        return float(np.sum((self.count[mask] / total) * np.abs(obs - mean_p)))


def update_threshold_counts(threshold_counts, probs, y, thresholds):
    for cls in [1, 2]:
        p = probs[:, cls]
        y_pos = y == cls
        for t in thresholds:
            pred = p >= t
            key = (cls, t)
            threshold_counts[key]["predicted_positive"] += int(pred.sum())
            threshold_counts[key]["tp"] += int((pred & y_pos).sum())
            threshold_counts[key]["fp"] += int((pred & ~y_pos).sum())
            threshold_counts[key]["fn"] += int((~pred & y_pos).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", default="data/processed_h5/datafile_test.h5")
    parser.add_argument("--dataset", default="data/processed_h5/dataset_test.h5")
    parser.add_argument("--model", default="results/best_models/flank80_focal_epoch2_best.pt")
    parser.add_argument("--temperature-txt", default="results/openspliceai_style_vectorT_flank80_epoch2_fullval/temperature_best.txt")
    parser.add_argument("--chrom", default="chr9")
    parser.add_argument("--out-dir", default="results/chromosome_eval_flank80_epoch2_chr9")
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping, seg_counts = infer_gene_to_shard_map(args.datafile, args.dataset)

    with h5py.File(args.datafile, "r") as df:
        chroms = [decode_str(x) for x in df["CHROM"][:]]
        names = [decode_str(x) for x in df["NAME"][:]]

    target_genes = [i for i, c in enumerate(chroms) if c == args.chrom]
    if not target_genes:
        raise RuntimeError(f"No genes found for {args.chrom}")

    map_by_gene = {m["gene_index"]: m for m in mapping}

    keep_by_shard = defaultdict(list)
    rows = []
    for gi in target_genes:
        m = map_by_gene[gi]
        start = m["segment_offset"]
        end = start + m["n_segments"]
        keep_by_shard[m["shard"]].extend(range(start, end))
        rows.append(
            {
                "gene_index": gi,
                "name": names[gi],
                "chrom": chroms[gi],
                "shard": m["shard"],
                "segment_offset": start,
                "n_segments": m["n_segments"],
                "positions": m["n_segments"] * 5000,
            }
        )

    gene_df = pd.DataFrame(rows)
    gene_df.to_csv(out_dir / f"{args.chrom}_gene_segment_map.csv", index=False)

    print(f"Chromosome: {args.chrom}")
    print(f"Genes: {len(gene_df)}")
    print(f"Segments: {gene_df['n_segments'].sum()}")
    print(f"Approx positions before clipping: {gene_df['positions'].sum()}")
    print(f"Shards touched: {sorted(keep_by_shard.keys())}")
    print(f"Wrote gene map: {out_dir / f'{args.chrom}_gene_segment_map.csv'}")

    if args.dry_run:
        return

    # Read temperature file robustly. Handles formats like:
    # 0.393 0.353 0.368
    # [0.393 0.353 0.368]
    # [0.393, 0.353, 0.368]
    raw_T = Path(args.temperature_txt).read_text().strip()
    raw_T = raw_T.replace("[", " ").replace("]", " ").replace(",", " ")
    T = np.fromstring(raw_T, sep=" ").astype(np.float64)

    if T.size != 3:
        raise ValueError(f"Expected 3 temperature values, got {T.size}: {raw_T!r}")

    T = T.reshape(1, 3)
    print("Temperature:", T.flatten())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model, optimizer, scheduler, params = initialize_model_and_optim(
        device=device,
        flanking_size=args.flanking_size,
        epochs=1,
        scheduler="MultiStepLR",
    )

    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if hasattr(model, "apply_softmax"):
        model.apply_softmax = False

    params["RANDOM_SEED"] = 123

    methods = {
        "uncalibrated": {
            "multi": MulticlassCalib(args.n_bins),
            "acc": BinaryCalib(args.n_bins),
            "don": BinaryCalib(args.n_bins),
            "thresholds": defaultdict(lambda: {"predicted_positive": 0, "tp": 0, "fp": 0, "fn": 0}),
            "argmax_pred": np.zeros(3, dtype=np.int64),
            "argmax_tp": np.zeros(3, dtype=np.int64),
        },
        "openspliceai_style_vectorT": {
            "multi": MulticlassCalib(args.n_bins),
            "acc": BinaryCalib(args.n_bins),
            "don": BinaryCalib(args.n_bins),
            "thresholds": defaultdict(lambda: {"predicted_positive": 0, "tp": 0, "fp": 0, "fn": 0}),
            "argmax_pred": np.zeros(3, dtype=np.int64),
            "argmax_tp": np.zeros(3, dtype=np.int64),
        },
    }

    thresholds = [0.01, 0.05, 0.10, 0.50]
    total_positions = 0
    class_counts = np.zeros(3, dtype=np.int64)

    with h5py.File(args.dataset, "r") as h5f:
        with torch.no_grad():
            for shard_idx in sorted(keep_by_shard.keys()):
                keep_segments = np.array(sorted(set(keep_by_shard[shard_idx])), dtype=np.int64)
                keep_set = set(keep_segments.tolist())
                print(f"Shard {shard_idx}: keeping {len(keep_set)} segments")

                loader = load_data_from_shard(
                    h5f,
                    shard_idx,
                    device,
                    params["BATCH_SIZE"],
                    params,
                    shuffle=False,
                )

                cursor = 0

                for batch in loader:
                    DNAs, labels = batch[0].to(device), batch[1].to(device)
                    batch_n = int(DNAs.shape[0])
                    segment_ids = np.arange(cursor, cursor + batch_n)
                    local_keep = np.array([j for j, sid in enumerate(segment_ids) if int(sid) in keep_set], dtype=np.int64)
                    cursor += batch_n

                    if len(local_keep) == 0:
                        continue

                    DNAs = DNAs[local_keep]
                    labels = labels[local_keep]

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

                    logits_np = logits.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    logits_flat = np.transpose(logits_np, (0, 2, 1)).reshape(-1, 3)
                    labels_flat = np.transpose(labels_np, (0, 2, 1)).reshape(-1, 3)
                    y = labels_flat.argmax(axis=1)

                    total_positions += len(y)
                    class_counts += np.bincount(y, minlength=3)

                    probs_uncal = stable_softmax(logits_flat.astype(np.float64))
                    probs_cal = stable_softmax(logits_flat.astype(np.float64) / T)

                    for method_name, probs in [
                        ("uncalibrated", probs_uncal),
                        ("openspliceai_style_vectorT", probs_cal),
                    ]:
                        obj = methods[method_name]
                        obj["multi"].update(probs, y)
                        obj["acc"].update(probs[:, 1], (y == 1).astype(float))
                        obj["don"].update(probs[:, 2], (y == 2).astype(float))

                        pred = probs.argmax(axis=1)
                        obj["argmax_pred"] += np.bincount(pred, minlength=3)
                        for c in [0, 1, 2]:
                            obj["argmax_tp"][c] += int(((pred == c) & (y == c)).sum())

                        update_threshold_counts(obj["thresholds"], probs, y, thresholds)

    metric_rows = []
    threshold_rows = []
    argmax_rows = []
    reliability_rows = []

    for method_name, obj in methods.items():
        ece, nll = obj["multi"].metrics()

        metric_rows.append(
            {
                "method": method_name,
                "chrom": args.chrom,
                "multiclass_ece": ece,
                "multiclass_nll": nll,
                "acceptor_ece": obj["acc"].ece(),
                "donor_ece": obj["don"].ece(),
                "total_positions": total_positions,
                "nonsplice_count": class_counts[0],
                "acceptor_count": class_counts[1],
                "donor_count": class_counts[2],
            }
        )

        for c in [0, 1, 2]:
            argmax_rows.append(
                {
                    "method": method_name,
                    "predicted_class": c,
                    "predicted_count": int(obj["argmax_pred"][c]),
                    "true_positive_count": int(obj["argmax_tp"][c]),
                }
            )

        for (cls, t), d in obj["thresholds"].items():
            precision = d["tp"] / d["predicted_positive"] if d["predicted_positive"] else np.nan
            recall = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else np.nan
            threshold_rows.append(
                {
                    "method": method_name,
                    "class": cls,
                    "threshold": t,
                    **d,
                    "precision": precision,
                    "recall": recall,
                }
            )

        reliability_rows.append(obj["multi"].bins_df(method_name))

    pd.DataFrame(metric_rows).to_csv(out_dir / f"{args.chrom}_streaming_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out_dir / f"{args.chrom}_threshold_metrics.csv", index=False)
    pd.DataFrame(argmax_rows).to_csv(out_dir / f"{args.chrom}_argmax.csv", index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(out_dir / f"{args.chrom}_multiclass_reliability_bins.csv", index=False)

    print("\nDone.")
    print("Class counts [non, acceptor, donor]:", class_counts)
    print("Total positions:", total_positions)
    print("Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
