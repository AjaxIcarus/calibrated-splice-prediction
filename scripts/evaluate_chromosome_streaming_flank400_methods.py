import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from openspliceai.constants import CL_max
from openspliceai.train.train import initialize_model_and_optim
from openspliceai.train_base.utils import (
    clip_datapoints,
    load_data_from_shard,
)


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
    return sorted(int(k[1:]) for k in x_keys)


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
    Infer the mapping:

        gene index -> dataset shard index and segment offset

    This assumes that datafile order and dataset shard order match.
    The mapping is validated by matching summed per-gene segment
    counts to the Y shard sizes.
    """
    with (
        h5py.File(datafile_h5, "r") as df,
        h5py.File(dataset_h5, "r") as ds,
    ):
        shard_ids = get_h5_indices(ds)
        y_counts = [
            int(ds[f"Y{sid}"].shape[1])
            for sid in shard_ids
        ]

        candidates = []

        for key in ["SEQ", "LABEL"]:
            if key not in df:
                continue

            lengths = [
                object_length(df[key], i)
                for i in range(len(df["CHROM"]))
            ]
            seg_counts = [
                int(np.ceil(length / chunk_len))
                for length in lengths
            ]

            mapping = []
            gene_i = 0
            ok = True

            for sid, target_segments in zip(shard_ids, y_counts):
                offset = 0

                while (
                    gene_i < len(seg_counts)
                    and offset < target_segments
                ):
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
                "Could not infer gene-to-shard mapping from "
                "SEQ or LABEL lengths. Need to inspect the "
                "OpenSpliceAI dataset construction."
            )

        key, mapping, seg_counts = candidates[0]

        print(
            f"Using mapping inferred from {key} lengths.",
            flush=True,
        )

        return mapping, seg_counts


def read_temperature_file(path):
    """
    Read temperature formats such as:

        0.393 0.353 0.368
        [0.393 0.353 0.368]
        [0.393, 0.353, 0.368]
    """
    raw = Path(path).read_text().strip()
    cleaned = (
        raw.replace("[", " ")
        .replace("]", " ")
        .replace(",", " ")
    )

    temperature = np.fromstring(
        cleaned,
        sep=" ",
    ).astype(np.float64)

    if temperature.size != 3:
        raise ValueError(
            f"Expected three temperature values in {path}, "
            f"but found {temperature.size}: {raw!r}"
        )

    return temperature.reshape(1, 3)


class MulticlassCalib:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.edges = np.linspace(0, 1, n_bins + 1)

        self.count = np.zeros(
            n_bins,
            dtype=np.float64,
        )
        self.sum_conf = np.zeros(
            n_bins,
            dtype=np.float64,
        )
        self.sum_correct = np.zeros(
            n_bins,
            dtype=np.float64,
        )

        self.nll_sum = 0.0
        self.n = 0

    def update(self, probs, y):
        eps = 1e-12

        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = (pred == y).astype(np.float64)

        bins = (
            np.searchsorted(
                self.edges,
                conf,
                side="right",
            )
            - 1
        )
        bins = np.clip(
            bins,
            0,
            self.n_bins - 1,
        )

        self.count += np.bincount(
            bins,
            minlength=self.n_bins,
        )
        self.sum_conf += np.bincount(
            bins,
            weights=conf,
            minlength=self.n_bins,
        )
        self.sum_correct += np.bincount(
            bins,
            weights=correct,
            minlength=self.n_bins,
        )

        true_probs = probs[
            np.arange(len(y)),
            y,
        ]

        self.nll_sum += float(
            np.sum(
                -np.log(
                    np.clip(
                        true_probs,
                        eps,
                        1.0,
                    )
                )
            )
        )
        self.n += len(y)

    def metrics(self):
        mask = self.count > 0

        avg_conf = np.zeros_like(self.count)
        avg_acc = np.zeros_like(self.count)

        avg_conf[mask] = (
            self.sum_conf[mask]
            / self.count[mask]
        )
        avg_acc[mask] = (
            self.sum_correct[mask]
            / self.count[mask]
        )

        ece = np.sum(
            (self.count[mask] / self.n)
            * np.abs(
                avg_acc[mask]
                - avg_conf[mask]
            )
        )

        nll = self.nll_sum / self.n

        return float(ece), float(nll)

    def bins_df(self, method):
        rows = []

        for i in range(self.n_bins):
            if self.count[i] == 0:
                continue

            mean_conf = (
                self.sum_conf[i]
                / self.count[i]
            )
            observed_accuracy = (
                self.sum_correct[i]
                / self.count[i]
            )

            rows.append(
                {
                    "method": method,
                    "bin": i,
                    "bin_low": self.edges[i],
                    "bin_high": self.edges[i + 1],
                    "count": int(self.count[i]),
                    "mean_confidence": mean_conf,
                    "observed_accuracy": observed_accuracy,
                    "gap": (
                        observed_accuracy
                        - mean_conf
                    ),
                }
            )

        return pd.DataFrame(rows)


class BinaryCalib:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins
        self.edges = np.linspace(0, 1, n_bins + 1)

        self.count = np.zeros(
            n_bins,
            dtype=np.float64,
        )
        self.sum_p = np.zeros(
            n_bins,
            dtype=np.float64,
        )
        self.sum_y = np.zeros(
            n_bins,
            dtype=np.float64,
        )

    def update(self, p, y_bin):
        bins = (
            np.searchsorted(
                self.edges,
                p,
                side="right",
            )
            - 1
        )
        bins = np.clip(
            bins,
            0,
            self.n_bins - 1,
        )

        self.count += np.bincount(
            bins,
            minlength=self.n_bins,
        )
        self.sum_p += np.bincount(
            bins,
            weights=p,
            minlength=self.n_bins,
        )
        self.sum_y += np.bincount(
            bins,
            weights=y_bin,
            minlength=self.n_bins,
        )

    def ece(self):
        total = self.count.sum()
        mask = self.count > 0

        mean_p = (
            self.sum_p[mask]
            / self.count[mask]
        )
        observed_frequency = (
            self.sum_y[mask]
            / self.count[mask]
        )

        ece = np.sum(
            (self.count[mask] / total)
            * np.abs(
                observed_frequency
                - mean_p
            )
        )

        return float(ece)


def new_method_accumulators(n_bins):
    return {
        "multi": MulticlassCalib(n_bins),
        "acc": BinaryCalib(n_bins),
        "don": BinaryCalib(n_bins),
        "thresholds": defaultdict(
            lambda: {
                "predicted_positive": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }
        ),
        "argmax_pred": np.zeros(
            3,
            dtype=np.int64,
        ),
        "argmax_tp": np.zeros(
            3,
            dtype=np.int64,
        ),
    }


def update_threshold_counts(
    threshold_counts,
    probs,
    y,
    thresholds,
):
    for cls in [1, 2]:
        p = probs[:, cls]
        y_pos = y == cls

        for threshold in thresholds:
            pred = p >= threshold
            key = (cls, threshold)

            threshold_counts[key][
                "predicted_positive"
            ] += int(pred.sum())

            threshold_counts[key]["tp"] += int(
                (pred & y_pos).sum()
            )
            threshold_counts[key]["fp"] += int(
                (pred & ~y_pos).sum()
            )
            threshold_counts[key]["fn"] += int(
                (~pred & y_pos).sum()
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datafile",
        default=(
            "data/processed_h5_flank400/"
            "datafile_test.h5"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "data/processed_h5_flank400/"
            "dataset_test.h5"
        ),
    )
    parser.add_argument(
        "--model",
        default=(
            "results/best_models/"
            "flank400_focal_best.pt"
        ),
    )
    parser.add_argument(
        "--temperature-txt",
        default=(
            "results/"
            "openspliceai_style_vectorT_"
            "flank400_epoch8_fullval_3000batched/"
            "temperature_best.txt"
        ),
        help=(
            "OpenSpliceAI-style vector-temperature "
            "file in non-splice, acceptor, donor order."
        ),
    )
    parser.add_argument(
        "--unweighted-temperature",
        nargs=3,
        type=float,
        default=[
            0.5890325,
            0.31363136,
            0.3490945,
        ],
        metavar=(
            "T_NON",
            "T_ACC",
            "T_DON",
        ),
    )
    parser.add_argument(
        "--weighted-temperature",
        nargs=3,
        type=float,
        default=[
            0.47286093,
            0.46725452,
            0.49169588,
        ],
        metavar=(
            "T_NON",
            "T_ACC",
            "T_DON",
        ),
    )
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--chrom",
        default="chr9",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "results/"
            "chromosome_eval_flank400_"
            "epoch8_chr9"
        ),
    )
    parser.add_argument(
        "--flanking-size",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping, _ = infer_gene_to_shard_map(
        args.datafile,
        args.dataset,
    )

    with h5py.File(args.datafile, "r") as datafile:
        chroms = [
            decode_str(x)
            for x in datafile["CHROM"][:]
        ]
        names = [
            decode_str(x)
            for x in datafile["NAME"][:]
        ]

    target_genes = [
        i
        for i, chromosome in enumerate(chroms)
        if chromosome == args.chrom
    ]

    if not target_genes:
        raise RuntimeError(
            f"No genes found for {args.chrom}"
        )

    map_by_gene = {
        item["gene_index"]: item
        for item in mapping
    }

    keep_by_shard = defaultdict(list)
    rows = []

    for gene_index in target_genes:
        mapping_item = map_by_gene[gene_index]

        start = mapping_item["segment_offset"]
        end = (
            start
            + mapping_item["n_segments"]
        )

        keep_by_shard[
            mapping_item["shard"]
        ].extend(
            range(start, end)
        )

        rows.append(
            {
                "gene_index": gene_index,
                "name": names[gene_index],
                "chrom": chroms[gene_index],
                "shard": mapping_item["shard"],
                "segment_offset": start,
                "n_segments": (
                    mapping_item["n_segments"]
                ),
                "positions": (
                    mapping_item["n_segments"]
                    * 5000
                ),
            }
        )

    gene_df = pd.DataFrame(rows)

    gene_map_path = (
        out_dir
        / f"{args.chrom}_gene_segment_map.csv"
    )
    gene_df.to_csv(
        gene_map_path,
        index=False,
    )

    print(
        f"Chromosome: {args.chrom}",
        flush=True,
    )
    print(
        f"Genes: {len(gene_df)}",
        flush=True,
    )
    print(
        "Segments: "
        f"{gene_df['n_segments'].sum()}",
        flush=True,
    )
    print(
        "Approx positions before clipping: "
        f"{gene_df['positions'].sum()}",
        flush=True,
    )
    print(
        "Shards touched: "
        f"{sorted(keep_by_shard.keys())}",
        flush=True,
    )
    print(
        f"Wrote gene map: {gene_map_path}",
        flush=True,
    )

    if args.dry_run:
        return

    T_osai = read_temperature_file(
        args.temperature_txt
    )
    T_unweighted = np.asarray(
        args.unweighted_temperature,
        dtype=np.float64,
    ).reshape(1, 3)
    T_weighted = np.asarray(
        args.weighted_temperature,
        dtype=np.float64,
    ).reshape(1, 3)
    T_global = np.full(
        (1, 3),
        args.global_temperature,
        dtype=np.float64,
    )

    print(
        "Fixed global temperature:",
        T_global.flatten(),
        flush=True,
    )
    print(
        "Unweighted vector temperature:",
        T_unweighted.flatten(),
        flush=True,
    )
    print(
        "Genome-weighted vector temperature:",
        T_weighted.flatten(),
        flush=True,
    )
    print(
        "OpenSpliceAI-style temperature:",
        T_osai.flatten(),
        flush=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
        flush=True,
    )

    model, optimizer, scheduler, params = (
        initialize_model_and_optim(
            device=device,
            flanking_size=args.flanking_size,
            epochs=1,
            scheduler="MultiStepLR",
        )
    )

    state = torch.load(
        args.model,
        map_location=device,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if hasattr(model, "apply_softmax"):
        model.apply_softmax = False

    params["RANDOM_SEED"] = 123

    method_temperatures = {
        "uncalibrated": None,
        "fixed_global_T1p1": T_global,
        "unweighted_vectorT": T_unweighted,
        "genome_weighted_vectorT": T_weighted,
        "openspliceai_style_vectorT": T_osai,
    }

    methods = {
        method_name: new_method_accumulators(
            args.n_bins
        )
        for method_name in method_temperatures
    }

    thresholds = [
        0.01,
        0.05,
        0.10,
        0.50,
    ]

    total_positions = 0
    class_counts = np.zeros(
        3,
        dtype=np.int64,
    )

    with h5py.File(args.dataset, "r") as h5f:
        with torch.no_grad():
            for shard_idx in sorted(
                keep_by_shard.keys()
            ):
                keep_segments = np.array(
                    sorted(
                        set(
                            keep_by_shard[
                                shard_idx
                            ]
                        )
                    ),
                    dtype=np.int64,
                )
                keep_set = set(
                    keep_segments.tolist()
                )

                print(
                    f"Shard {shard_idx}: "
                    f"keeping {len(keep_set)} "
                    "segments",
                    flush=True,
                )

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
                    DNAs = batch[0].to(device)
                    labels = batch[1].to(device)

                    batch_n = int(
                        DNAs.shape[0]
                    )

                    segment_ids = np.arange(
                        cursor,
                        cursor + batch_n,
                    )

                    local_keep = np.array(
                        [
                            j
                            for j, segment_id
                            in enumerate(segment_ids)
                            if int(segment_id)
                            in keep_set
                        ],
                        dtype=np.int64,
                    )

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

                    DNAs = DNAs.to(
                        torch.float32
                    ).to(device)
                    labels = labels.to(
                        torch.float32
                    ).to(device)

                    logits = model(DNAs)

                    logits_np = (
                        logits.detach()
                        .cpu()
                        .numpy()
                    )
                    labels_np = (
                        labels.detach()
                        .cpu()
                        .numpy()
                    )

                    logits_flat = np.transpose(
                        logits_np,
                        (0, 2, 1),
                    ).reshape(-1, 3)

                    labels_flat = np.transpose(
                        labels_np,
                        (0, 2, 1),
                    ).reshape(-1, 3)

                    y = labels_flat.argmax(
                        axis=1
                    )

                    total_positions += len(y)
                    class_counts += np.bincount(
                        y,
                        minlength=3,
                    )

                    logits64 = logits_flat.astype(
                        np.float64
                    )

                    for (
                        method_name,
                        temperature,
                    ) in method_temperatures.items():
                        if temperature is None:
                            scaled_logits = logits64
                        else:
                            scaled_logits = (
                                logits64
                                / temperature
                            )

                        probs = stable_softmax(
                            scaled_logits
                        )

                        obj = methods[method_name]

                        obj["multi"].update(
                            probs,
                            y,
                        )
                        obj["acc"].update(
                            probs[:, 1],
                            (y == 1).astype(float),
                        )
                        obj["don"].update(
                            probs[:, 2],
                            (y == 2).astype(float),
                        )

                        pred = probs.argmax(
                            axis=1
                        )

                        obj[
                            "argmax_pred"
                        ] += np.bincount(
                            pred,
                            minlength=3,
                        )

                        for class_index in [
                            0,
                            1,
                            2,
                        ]:
                            obj["argmax_tp"][
                                class_index
                            ] += int(
                                (
                                    (
                                        pred
                                        == class_index
                                    )
                                    & (
                                        y
                                        == class_index
                                    )
                                ).sum()
                            )

                        update_threshold_counts(
                            obj["thresholds"],
                            probs,
                            y,
                            thresholds,
                        )

                        del probs

    metric_rows = []
    threshold_rows = []
    argmax_rows = []
    reliability_rows = []

    for method_name, obj in methods.items():
        multiclass_ece, multiclass_nll = (
            obj["multi"].metrics()
        )

        metric_rows.append(
            {
                "method": method_name,
                "chrom": args.chrom,
                "multiclass_ece": (
                    multiclass_ece
                ),
                "multiclass_nll": (
                    multiclass_nll
                ),
                "acceptor_ece": (
                    obj["acc"].ece()
                ),
                "donor_ece": (
                    obj["don"].ece()
                ),
                "total_positions": (
                    total_positions
                ),
                "nonsplice_count": (
                    int(class_counts[0])
                ),
                "acceptor_count": (
                    int(class_counts[1])
                ),
                "donor_count": (
                    int(class_counts[2])
                ),
            }
        )

        for class_index in [0, 1, 2]:
            argmax_rows.append(
                {
                    "method": method_name,
                    "predicted_class": (
                        class_index
                    ),
                    "predicted_count": int(
                        obj["argmax_pred"][
                            class_index
                        ]
                    ),
                    "true_positive_count": int(
                        obj["argmax_tp"][
                            class_index
                        ]
                    ),
                }
            )

        for (
            class_index,
            threshold,
        ), counts in obj[
            "thresholds"
        ].items():
            if counts["predicted_positive"]:
                precision = (
                    counts["tp"]
                    / counts[
                        "predicted_positive"
                    ]
                )
            else:
                precision = np.nan

            positive_count = (
                counts["tp"]
                + counts["fn"]
            )

            if positive_count:
                recall = (
                    counts["tp"]
                    / positive_count
                )
            else:
                recall = np.nan

            threshold_rows.append(
                {
                    "method": method_name,
                    "class": class_index,
                    "threshold": threshold,
                    **counts,
                    "precision": precision,
                    "recall": recall,
                }
            )

        reliability_rows.append(
            obj["multi"].bins_df(
                method_name
            )
        )

    metrics_path = (
        out_dir
        / f"{args.chrom}_streaming_metrics.csv"
    )
    thresholds_path = (
        out_dir
        / f"{args.chrom}_threshold_metrics.csv"
    )
    argmax_path = (
        out_dir
        / f"{args.chrom}_argmax.csv"
    )
    reliability_path = (
        out_dir
        / (
            f"{args.chrom}_"
            "multiclass_reliability_bins.csv"
        )
    )

    pd.DataFrame(
        metric_rows
    ).to_csv(
        metrics_path,
        index=False,
    )

    pd.DataFrame(
        threshold_rows
    ).to_csv(
        thresholds_path,
        index=False,
    )

    pd.DataFrame(
        argmax_rows
    ).to_csv(
        argmax_path,
        index=False,
    )

    pd.concat(
        reliability_rows,
        ignore_index=True,
    ).to_csv(
        reliability_path,
        index=False,
    )

    print(
        "\nDone.",
        flush=True,
    )
    print(
        "Class counts "
        "[non, acceptor, donor]:",
        class_counts,
        flush=True,
    )
    print(
        "Total positions:",
        total_positions,
        flush=True,
    )
    print(
        "Wrote outputs to:",
        out_dir,
        flush=True,
    )


if __name__ == "__main__":
    main()