#!/usr/bin/env python3
"""Fit the full-validation OpenSpliceAI-style vector temperature safely."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn

from calibration_v2_common import (
    create_new_output_dir,
    write_json,
    write_temperature_text,
)
from openspliceai.calibrate.model_utils import (
    initialize_model_and_optim,
)
from openspliceai.calibrate.temperature_scaling import (
    get_validation_loader,
)
from openspliceai.train_base.utils import CL_max, setup_environment


PIPELINE_VERSION = 2


def choose_shards(
    h5_file: h5py.File,
    n_shards: int | None = None,
    seed: int = 42,
) -> list[int]:
    shard_ids = sorted(
        int(key[1:])
        for key in h5_file.keys()
        if key.startswith("X") and key[1:].isdigit()
    )
    if not shard_ids:
        raise RuntimeError("Validation H5 contains no X<number> shards")
    if n_shards is not None:
        if n_shards <= 0:
            raise ValueError("--max-validation-shards must be positive")
        if n_shards < len(shard_ids):
            generator = random.Random(seed)
            shard_ids = sorted(generator.sample(shard_ids, n_shards))
    return shard_ids


def inspect_selected_shards(
    h5_file: h5py.File,
    shard_ids: list[int],
) -> dict[str, Any]:
    total_sequences = 0
    raw_positions = 0
    output_length: int | None = None
    shard_rows: list[dict[str, Any]] = []

    for shard_id in shard_ids:
        x_name = f"X{shard_id}"
        y_name = f"Y{shard_id}"
        if x_name not in h5_file or y_name not in h5_file:
            raise RuntimeError(
                f"Selected shard {shard_id} is missing {x_name} or {y_name}"
            )
        x_shape = tuple(h5_file[x_name].shape)
        y_shape = tuple(h5_file[y_name].shape)
        if len(x_shape) != 3 or x_shape[2] != 4:
            raise RuntimeError(
                f"{x_name} must have shape (N, input_length, 4); "
                f"found {x_shape}"
            )
        if (
            len(y_shape) != 4
            or y_shape[0] != 1
            or y_shape[1] != x_shape[0]
            or y_shape[3] != 3
        ):
            raise RuntimeError(
                f"{y_name} has incompatible shape {y_shape} for "
                f"{x_name} shape {x_shape}"
            )
        if output_length is None:
            output_length = int(y_shape[2])
        elif output_length != int(y_shape[2]):
            raise RuntimeError(
                "Validation shards have inconsistent output lengths"
            )

        sequences = int(x_shape[0])
        total_sequences += sequences
        raw_positions += sequences * int(y_shape[2])
        shard_rows.append(
            {
                "shard_id": shard_id,
                "x_shape": list(x_shape),
                "y_shape": list(y_shape),
            }
        )

    return {
        "selected_shards": shard_ids,
        "shard_count": len(shard_ids),
        "total_sequences": total_sequences,
        "raw_positions": raw_positions,
        "output_length": output_length,
        "shards": shard_rows,
    }


def crop_context_without_dropping_sequences(
    inputs: torch.Tensor,
    model_context_length: int,
    maximum_context_length: int,
) -> torch.Tensor:
    context_difference = (
        int(maximum_context_length) - int(model_context_length)
    )
    if context_difference < 0 or context_difference % 2 != 0:
        raise RuntimeError(
            "Invalid context lengths: "
            f"CL={model_context_length}, CL_max={maximum_context_length}"
        )
    clip = context_difference // 2
    if clip == 0:
        return inputs
    if inputs.shape[-1] <= 2 * clip:
        raise RuntimeError(
            f"Cannot clip {clip} bases from each side of input shape "
            f"{tuple(inputs.shape)}"
        )
    return inputs[:, :, clip:-clip]


def collect_valid_logits_and_labels(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    model_parameters: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    census = {
        "observed_batches": 0,
        "observed_sequences": 0,
        "valid_positions": 0,
        "padding_positions": 0,
        "positive_positions": 0,
        "class_counts": [0, 0, 0],
        "observed_batch_sizes": [],
    }

    with torch.no_grad():
        for inputs, labels_one_hot in loader:
            if (
                labels_one_hot.ndim != 3
                or labels_one_hot.shape[1] != 3
            ):
                raise RuntimeError(
                    "Expected labels with shape (batch, 3, length); "
                    f"found {tuple(labels_one_hot.shape)}"
                )

            batch_size = int(labels_one_hot.shape[0])
            census["observed_batches"] += 1
            census["observed_sequences"] += batch_size
            census["observed_batch_sizes"].append(batch_size)

            inputs = inputs.to(device)
            labels_one_hot = labels_one_hot.to(device)
            inputs = crop_context_without_dropping_sequences(
                inputs,
                model_parameters["CL"],
                CL_max,
            )
            logits = model(inputs)

            if logits.ndim != 3 or logits.shape[1] != 3:
                raise RuntimeError(
                    "Expected model logits with shape (batch, 3, length); "
                    f"found {tuple(logits.shape)}"
                )
            if logits.shape != labels_one_hot.shape:
                raise RuntimeError(
                    f"Logit shape {tuple(logits.shape)} does not match "
                    f"label shape {tuple(labels_one_hot.shape)}"
                )

            label_sums = labels_one_hot.sum(dim=1)
            valid_mask = label_sums.eq(1)
            padding_mask = label_sums.eq(0)
            malformed_mask = ~(valid_mask | padding_mask)
            malformed_count = int(
                torch.count_nonzero(malformed_mask).item()
            )
            if malformed_count:
                raise RuntimeError(
                    f"Found {malformed_count:,} malformed label positions"
                )

            valid_count = int(torch.count_nonzero(valid_mask).item())
            padding_count = int(
                torch.count_nonzero(padding_mask).item()
            )
            census["valid_positions"] += valid_count
            census["padding_positions"] += padding_count

            logits_by_position = (
                logits.permute(0, 2, 1).contiguous()
            )
            labels_by_position = (
                labels_one_hot.permute(0, 2, 1).contiguous()
            )
            valid_logits = logits_by_position[valid_mask]
            valid_labels_one_hot = labels_by_position[valid_mask]
            valid_labels = valid_labels_one_hot.argmax(dim=1).long()

            batch_counts = torch.bincount(
                valid_labels,
                minlength=3,
            )
            for class_index in range(3):
                census["class_counts"][class_index] += int(
                    batch_counts[class_index].item()
                )
            census["positive_positions"] += int(
                batch_counts[1].item() + batch_counts[2].item()
            )

            logits_parts.append(valid_logits.detach().cpu())
            label_parts.append(valid_labels.detach().cpu())

    if not logits_parts:
        raise RuntimeError("Validation loader produced no batches")
    logits_cpu = torch.cat(logits_parts, dim=0)
    labels_cpu = torch.cat(label_parts, dim=0)
    if len(logits_cpu) != census["valid_positions"]:
        raise RuntimeError(
            "Collected logit count does not match valid-position census"
        )
    if len(labels_cpu) != census["valid_positions"]:
        raise RuntimeError(
            "Collected label count does not match valid-position census"
        )
    return logits_cpu, labels_cpu, census


def check_expected_census(
    census: dict[str, Any],
    direct_h5_census: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    observed_sequences = int(census["observed_sequences"])
    direct_sequences = int(direct_h5_census["total_sequences"])
    if observed_sequences != direct_sequences:
        raise RuntimeError(
            "Validation loader or preprocessing dropped sequences: "
            f"direct H5={direct_sequences:,}, observed={observed_sequences:,}"
        )

    observed_total_positions = (
        int(census["valid_positions"])
        + int(census["padding_positions"])
    )
    direct_total_positions = int(direct_h5_census["raw_positions"])
    if observed_total_positions != direct_total_positions:
        raise RuntimeError(
            "Valid plus padding positions do not match direct H5 census: "
            f"direct={direct_total_positions:,}, "
            f"observed={observed_total_positions:,}"
        )

    expectations = {
        "observed_sequences": args.expected_sequences,
        "valid_positions": args.expected_valid_positions,
        "padding_positions": args.expected_padding_positions,
        "positive_positions": args.expected_positive_positions,
    }
    for field, expected in expectations.items():
        if expected is None:
            continue
        actual = int(census[field])
        if actual != expected:
            raise RuntimeError(
                f"{field}: expected {expected:,}, found {actual:,}"
            )

    if args.expected_acceptor_positions is not None:
        actual = int(census["class_counts"][1])
        if actual != args.expected_acceptor_positions:
            raise RuntimeError(
                "acceptor positions: expected "
                f"{args.expected_acceptor_positions:,}, found {actual:,}"
            )
    if args.expected_donor_positions is not None:
        actual = int(census["class_counts"][2])
        if actual != args.expected_donor_positions:
            raise RuntimeError(
                "donor positions: expected "
                f"{args.expected_donor_positions:,}, found {actual:,}"
            )


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    ece = torch.zeros((), device=logits.device)
    edges = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        device=logits.device,
    )
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            ece += mask.float().mean() * torch.abs(
                correct[mask].float().mean()
                - confidence[mask].mean()
            )
    return ece


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if int(checkpoint.get("pipeline_version", -1)) != PIPELINE_VERSION:
        raise RuntimeError(
            f"Resume checkpoint is not a calibration-v2 checkpoint: {path}"
        )
    return checkpoint


def prepare_output_dir(
    output_dir: Path,
    resume_path: Path | None,
) -> Path:
    if resume_path is None:
        return create_new_output_dir(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError(
            "--output-dir must already exist when --resume is used"
        )
    if not resume_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {resume_path}"
        )
    if resume_path.resolve().parent != output_dir.resolve():
        raise RuntimeError(
            "Resume checkpoint must be directly inside --output-dir"
        )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the OpenSpliceAI-style class-wise vector temperature on "
            "every valid position in the full validation H5. Zero-labelled "
            "padding is excluded before argmax and no batch rows are "
            "dropped during context clipping."
        )
    )
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-validation-shards", type=int)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume")
    parser.add_argument("--loss", default="focal_loss")
    parser.add_argument("--expected-sequences", type=int)
    parser.add_argument("--expected-valid-positions", type=int)
    parser.add_argument("--expected-padding-positions", type=int)
    parser.add_argument("--expected-positive-positions", type=int)
    parser.add_argument("--expected-acceptor-positions", type=int)
    parser.add_argument("--expected-donor-positions", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.flanking_size <= 0:
        raise ValueError("--flanking-size must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    model_path = Path(args.pretrained_model)
    validation_path = Path(args.validation_dataset)
    output_dir = Path(args.output_dir)
    resume_path = Path(args.resume) if args.resume else None
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validation H5 does not exist: {validation_path}"
        )
    output_dir = prepare_output_dir(output_dir, resume_path)

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Compatibility fields required by OpenSpliceAI setup_environment.
    args.train_dataset = args.validation_dataset
    args.test_dataset = args.validation_dataset
    args.project_name = (
        "openspliceai_style_vector_temperature_long_v2"
    )
    args.exp_num = 0
    args.early_stopping = False
    args.patience = 2
    args.temperature_file = None

    device = setup_environment(args)
    print("Initializing model...", flush=True)
    model, model_parameters = initialize_model_and_optim(
        device,
        args.flanking_size,
        args.pretrained_model,
    )

    with h5py.File(validation_path, "r") as validation_h5:
        validation_shards = choose_shards(
            validation_h5,
            args.max_validation_shards,
            args.random_seed,
        )
        direct_h5_census = inspect_selected_shards(
            validation_h5,
            validation_shards,
        )
        print(
            f"Validation shards ({len(validation_shards)}): "
            f"{validation_shards}"
        )
        print(
            "Direct H5 sequences:",
            f"{direct_h5_census['total_sequences']:,}",
        )
        print(
            "Direct H5 raw positions:",
            f"{direct_h5_census['raw_positions']:,}",
        )
        print("Building validation loader...", flush=True)
        loader = get_validation_loader(
            validation_h5,
            validation_shards,
            model_parameters["BATCH_SIZE"],
        )
        print(
            "Collecting logits and masking zero-labelled padding...",
            flush=True,
        )
        logits_cpu, labels_cpu, observed_census = (
            collect_valid_logits_and_labels(
                model,
                loader,
                model_parameters,
                device,
            )
        )

    check_expected_census(
        observed_census,
        direct_h5_census,
        args,
    )
    print("Observed sequences:", f"{observed_census['observed_sequences']:,}")
    print("Valid positions:", f"{observed_census['valid_positions']:,}")
    print(
        "Padding positions excluded:",
        f"{observed_census['padding_positions']:,}",
    )
    print(
        "Positive positions:",
        f"{observed_census['positive_positions']:,}",
    )
    print("Class counts:", observed_census["class_counts"])
    print(
        "PASS: every selected sequence was retained and zero-labelled "
        "padding was excluded before argmax"
    )

    collection_metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "pretrained_model": str(model_path),
        "validation_dataset": str(validation_path),
        "flanking_size": args.flanking_size,
        "random_seed": args.random_seed,
        "model_parameters": {
            "CL": int(model_parameters["CL"]),
            "BATCH_SIZE": int(model_parameters["BATCH_SIZE"]),
            "N_GPUS_reported": int(model_parameters["N_GPUS"]),
            "CL_max": int(CL_max),
        },
        "direct_h5_census": direct_h5_census,
        "observed_census": observed_census,
    }
    write_json(
        output_dir / "validation_census.json",
        collection_metadata,
    )

    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)
    del logits_cpu
    del labels_cpu

    temperature = nn.Parameter(torch.ones(3, device=device))
    optimizer = torch.optim.Adam([temperature], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    best_loss = float("inf")
    best_temperature: torch.Tensor | None = None

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        temperature.data = checkpoint["temperature"].to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        checkpoint_best = checkpoint.get("best_temperature")
        if checkpoint_best is not None:
            best_temperature = checkpoint_best.to(device)
        if checkpoint.get("observed_census") != observed_census:
            raise RuntimeError(
                "Resume checkpoint validation census does not match "
                "the current run"
            )
        print(
            f"Resuming at epoch {start_epoch} with "
            f"best_loss={best_loss:.10f}"
        )

    log_path = output_dir / "training_log.tsv"
    if resume_path is None:
        log_path.write_text(
            "epoch\tnll\tece\tT_nonsplice\tT_acceptor\tT_donor\n",
            encoding="utf-8",
        )

    print(
        "Fitting unweighted full-validation OpenSpliceAI-style "
        "vector temperature...",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()
        evaluated_temperature = torch.clamp(
            temperature,
            min=0.05,
            max=5.0,
        )
        scaled_logits = logits / evaluated_temperature
        loss = criterion(scaled_logits, labels)
        loss_value = float(loss.detach().item())
        evaluated_temperature_copy = (
            evaluated_temperature.detach().clone()
        )

        if loss_value < best_loss:
            best_loss = loss_value
            best_temperature = evaluated_temperature_copy

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            temperature.clamp_(0.05, 5.0)

        should_print = (
            epoch == start_epoch
            or epoch % args.print_every == 0
            or epoch == args.epochs
        )
        if should_print:
            with torch.no_grad():
                ece_value = float(
                    compute_ece(
                        scaled_logits.detach(),
                        labels,
                    ).item()
                )
                temperature_np = (
                    evaluated_temperature_copy.cpu().numpy()
                )
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"NLL={loss_value:.10f} "
                f"ECE={ece_value:.10f} "
                f"T={temperature_np}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch}\t{loss_value:.10f}\t{ece_value:.10f}\t"
                    f"{temperature_np[0]:.10f}\t"
                    f"{temperature_np[1]:.10f}\t"
                    f"{temperature_np[2]:.10f}\n"
                )

        should_save = (
            epoch % args.save_every == 0
            or epoch == args.epochs
        )
        if should_save:
            if best_temperature is None:
                raise RuntimeError("No best temperature is available")
            checkpoint = {
                "pipeline_version": PIPELINE_VERSION,
                "epoch": epoch,
                "temperature": temperature.detach().cpu(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
                "best_temperature": best_temperature.detach().cpu(),
                "observed_census": observed_census,
                "direct_h5_census": direct_h5_census,
            }
            latest_path = output_dir / "checkpoint_latest.pt"
            epoch_path = (
                output_dir / f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(checkpoint, latest_path)
            torch.save(checkpoint, epoch_path)
            current_temperature_cpu = temperature.detach().cpu()
            torch.save(
                current_temperature_cpu,
                output_dir / "temperature_latest.pt",
            )
            write_temperature_text(
                output_dir / "temperature_latest.txt",
                current_temperature_cpu.numpy(),
            )
            best_temperature_cpu = best_temperature.detach().cpu()
            torch.save(
                best_temperature_cpu,
                output_dir / "temperature_best.pt",
            )
            write_temperature_text(
                output_dir / "temperature_best.txt",
                best_temperature_cpu.numpy(),
            )
            print(f"Saved checkpoint: {latest_path}", flush=True)

    if best_temperature is None:
        raise RuntimeError("Training completed without a best temperature")
    best_temperature_cpu = best_temperature.detach().cpu()
    torch.save(best_temperature_cpu, output_dir / "temperature.pt")
    torch.save(
        best_temperature_cpu,
        output_dir / "temperature_best.pt",
    )
    write_temperature_text(
        output_dir / "temperature.txt",
        best_temperature_cpu.numpy(),
    )
    write_temperature_text(
        output_dir / "temperature_best.txt",
        best_temperature_cpu.numpy(),
    )
    write_json(
        output_dir / "fit_summary.json",
        {
            **collection_metadata,
            "epochs_requested": args.epochs,
            "learning_rate": args.lr,
            "best_validation_nll": best_loss,
            "best_temperature": (
                best_temperature_cpu.numpy().tolist()
            ),
            "padding_mask": "labels.sum(dim=1) == 1",
            "sequence_clipping": (
                "Context-only clipping; no batch rows dropped"
            ),
        },
    )
    print("Best NLL:", best_loss)
    print("Best temperature:", best_temperature_cpu.numpy())
    print("Saved to:", output_dir)
    print(
        "PASS: full-validation OpenSpliceAI-style calibration v2 "
        "completed"
    )


if __name__ == "__main__":
    main()
