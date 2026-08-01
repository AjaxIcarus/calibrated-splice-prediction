#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np
import h5py
import torch
import torch.nn as nn

from openspliceai.calibrate.temperature_scaling import get_validation_loader
from openspliceai.calibrate.model_utils import initialize_model_and_optim
from openspliceai.train_base.utils import setup_environment, clip_datapoints, CL_max


def choose_shards(h5f, n_shards=None, seed=42):
    shard_ids = sorted(
        int(k[1:]) for k in h5f.keys()
        if k.startswith("X") and k[1:].isdigit()
    )
    if n_shards is not None and n_shards < len(shard_ids):
        rng = random.Random(seed)
        shard_ids = sorted(rng.sample(shard_ids, n_shards))
    return shard_ids


def collect_logits_labels(model, loader, model_params, device):
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = clip_datapoints(
                inputs,
                labels,
                model_params["CL"],
                CL_max,
                model_params["N_GPUS"],
            )
            logits = model(inputs)
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())

    logits = torch.cat(logits_list).permute(0, 2, 1).contiguous()
    labels = torch.cat(labels_list).permute(0, 2, 1).contiguous().argmax(dim=-1)

    n, l, c = logits.shape
    logits = logits.view(-1, c)
    labels = labels.view(-1).long()

    return logits, labels


def compute_ece(logits, labels, n_bins=15):
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)

    ece = torch.zeros((), device=logits.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * torch.abs(
                correct[mask].float().mean() - conf[mask].mean()
            )

    return ece


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--flanking-size", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-validation-shards", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--loss", default="focal_loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Compatibility fields for setup_environment
    args.train_dataset = args.validation_dataset
    args.test_dataset = args.validation_dataset
    args.project_name = "openspliceai_style_vector_temperature_long"
    args.exp_num = 0
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
    val_idxs = choose_shards(valid_h5f, args.max_validation_shards, args.random_seed)

    print("Validation shards:", len(val_idxs), val_idxs[:10], "...")
    print("Building validation loader...")
    loader = get_validation_loader(valid_h5f, val_idxs, model_params["BATCH_SIZE"])

    print("Collecting logits and labels once...")
    logits_cpu, labels_cpu = collect_logits_labels(model, loader, model_params, device)
    valid_h5f.close()

    print("Logits:", tuple(logits_cpu.shape))
    print("Labels:", tuple(labels_cpu.shape))

    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)

    temperature = nn.Parameter(torch.ones(3, device=device))
    optimizer = torch.optim.Adam([temperature], lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_loss = float("inf")
    best_temp = None

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        temperature.data = ckpt["temperature"].to(device)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_loss = float(ckpt.get("best_loss", float("inf")))
        if ckpt.get("best_temp") is not None:
            best_temp = ckpt["best_temp"].to(device)
        print(f"Resuming from epoch {start_epoch} with best_loss={best_loss:.10f}")

    log_path = os.path.join(args.output_dir, "training_log.tsv")
    if not args.resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch\tnll\tece\tT_nonsplice\tT_acceptor\tT_donor\n")

    print("Fitting unweighted OpenSpliceAI-style vector temperature...")
    for epoch in range(start_epoch, args.epochs + 1):
        optimizer.zero_grad()

        temp_clamped = torch.clamp(temperature, min=0.05, max=5.0)
        scaled_logits = logits / temp_clamped

        loss = criterion(scaled_logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            temperature.clamp_(0.05, 5.0)

        current_loss = loss.item()

        if current_loss < best_loss:
            best_loss = current_loss
            best_temp = temperature.detach().clone()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                temp_clamped = torch.clamp(temperature, min=0.05, max=5.0)
                scaled_logits = logits / temp_clamped
                ece = compute_ece(scaled_logits, labels).item()
                temp_np = temp_clamped.detach().cpu().numpy()

            print(
                f"Epoch {epoch}/{args.epochs} "
                f"NLL={current_loss:.8f} "
                f"ECE={ece:.8f} "
                f"T={temp_np}"
            )

            with open(log_path, "a") as f:
                f.write(
                    f"{epoch}\t{current_loss:.10f}\t{ece:.10f}\t"
                    f"{temp_np[0]:.10f}\t{temp_np[1]:.10f}\t{temp_np[2]:.10f}\n"
                )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "temperature": temperature.detach().cpu(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
                "best_temp": best_temp.detach().cpu() if best_temp is not None else None,
            }
            latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
            epoch_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(ckpt, latest_path)
            torch.save(ckpt, epoch_path)

            current_temp_cpu = temperature.detach().cpu()
            torch.save(current_temp_cpu, os.path.join(args.output_dir, "temperature_latest.pt"))
            with open(os.path.join(args.output_dir, "temperature_latest.txt"), "w") as f:
                f.write(str(current_temp_cpu.numpy()))

            if best_temp is not None:
                best_temp_cpu_mid = best_temp.detach().cpu()
                torch.save(best_temp_cpu_mid, os.path.join(args.output_dir, "temperature_best.pt"))
                with open(os.path.join(args.output_dir, "temperature_best.txt"), "w") as f:
                    f.write(str(best_temp_cpu_mid.numpy()))

            print(f"Saved checkpoint at epoch {epoch}: {latest_path}")

    best_temp_cpu = best_temp.detach().cpu()
    torch.save(best_temp_cpu, os.path.join(args.output_dir, "temperature.pt"))
    torch.save(best_temp_cpu, os.path.join(args.output_dir, "temperature_best.pt"))

    with open(os.path.join(args.output_dir, "temperature.txt"), "w") as f:
        f.write(str(best_temp_cpu.numpy()))
    with open(os.path.join(args.output_dir, "temperature_best.txt"), "w") as f:
        f.write(str(best_temp_cpu.numpy()))

    print("Best NLL:", best_loss)
    print("Best temperature:", best_temp_cpu.numpy())
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
