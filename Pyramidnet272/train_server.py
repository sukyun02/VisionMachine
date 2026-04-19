"""
Training script for PyramidNet-272 on CIFAR-100.

Server variant:
  - no wandb
  - CSV logging
  - matplotlib plots saved with the Agg backend
  - checkpoint format that loads cleanly into pyramidnet272()
"""

import argparse
import csv
import os
import random
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.cifar100 import fine_to_coarse_tensor, get_dataloaders
from losses.hierarchical_loss import HierarchicalLoss
from models.pyramidnet import pyramidnet272


def set_seed(seed: int, fast_cudnn: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not fast_cudnn
    torch.backends.cudnn.benchmark = fast_cudnn


def build_scheduler(optimizer, epochs, warmup_epochs=5, eta_min=1e-4):
    warmup_epochs = max(1, min(warmup_epochs, epochs))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    if epochs <= warmup_epochs:
        return warmup

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def move_images(imgs, device, channels_last=False):
    if channels_last:
        return imgs.to(
            device,
            memory_format=torch.channels_last,
            non_blocking=True,
        )
    return imgs.to(device, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    epoch=0, total_epochs=0, channels_last=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    pbar = tqdm(loader, ncols=100, leave=False,
                desc=f"  Train {epoch:4d}/{total_epochs}")

    for batch in pbar:
        if len(batch) == 4:
            imgs, la, lb, lam = batch
            imgs = move_images(imgs, device, channels_last)
            la = la.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            lam = lam.to(device, non_blocking=True)
        else:
            imgs, la = batch
            imgs = move_images(imgs, device, channels_last)
            la = la.to(device, non_blocking=True)
            lb = la
            lam = torch.ones(1, device=device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, la, lb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, la, lb, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == la).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_n += imgs.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / total_n:.4f}",
            "acc": f"{total_correct / total_n * 100:.1f}%",
        })

    pbar.close()
    optimizer.zero_grad(set_to_none=True)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device, channels_last=False):
    model.eval()
    mapping = torch.tensor(
        fine_to_coarse_tensor(torch.arange(100)).tolist(),
        dtype=torch.long,
        device=device,
    )

    fine_correct = 0
    sc_density_sum = 0.0
    total_n = 0

    pbar = tqdm(loader, ncols=100, leave=False, desc="  Eval ")
    for imgs, labels in pbar:
        imgs = move_images(imgs, device, channels_last)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)

        preds = logits.argmax(dim=1)
        fine_correct += (preds == labels).sum().item()

        _, top5 = logits.topk(5, dim=1)
        top5_coarse = mapping[top5]
        gt_coarse = mapping[labels]
        matches = top5_coarse.eq(gt_coarse.unsqueeze(1))
        sc_density_sum += matches.float().sum(dim=1).div(5).sum().item()

        total_n += imgs.size(0)
        pbar.set_postfix({
            "top1": f"{fine_correct / total_n * 100:.1f}%",
            "sc": f"{sc_density_sum / total_n * 100:.1f}%",
        })

    pbar.close()
    return fine_correct / total_n, sc_density_sum / total_n


def append_log(log_path, epoch, lr, train_loss, train_acc,
               val_top1, val_sc_density, is_swa,
               epoch_sec, imgs_per_sec, cuda_max_alloc_gib,
               cuda_max_reserved_gib):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{lr:.8f}",
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{val_top1:.6f}",
            f"{val_sc_density:.6f}",
            int(is_swa),
            f"{epoch_sec:.2f}",
            f"{imgs_per_sec:.2f}",
            f"{cuda_max_alloc_gib:.3f}",
            f"{cuda_max_reserved_gib:.3f}",
        ])


def append_perf_log(perf_path, epoch, lr, is_swa, did_eval,
                    train_sec, epoch_sec, train_imgs_per_sec,
                    epoch_imgs_per_sec, cuda_max_alloc_gib,
                    cuda_max_reserved_gib):
    with open(perf_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{lr:.8f}",
            int(is_swa),
            int(did_eval),
            f"{train_sec:.2f}",
            f"{epoch_sec:.2f}",
            f"{train_imgs_per_sec:.2f}",
            f"{epoch_imgs_per_sec:.2f}",
            f"{cuda_max_alloc_gib:.3f}",
            f"{cuda_max_reserved_gib:.3f}",
        ])


def load_log_rows(log_path):
    if not os.path.exists(log_path):
        return []

    rows = []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            epoch = row.get("epoch", "")
            if not epoch.isdigit():
                continue
            rows.append({
                "epoch": int(epoch),
                "lr": float(row["lr"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]) * 100,
                "val_top1": float(row["val_top1"]) * 100,
                "val_sc_density": float(row["val_sc_density"]) * 100,
                "is_swa": int(row["is_swa"]),
                "epoch_sec": float(row.get("epoch_sec", 0.0)),
                "imgs_per_sec": float(row.get("imgs_per_sec", 0.0)),
                "cuda_max_alloc_gib": float(row.get("cuda_max_alloc_gib", 0.0)),
                "cuda_max_reserved_gib": float(row.get("cuda_max_reserved_gib", 0.0)),
            })
    return rows


def plot_training_curves(log_path: str, ckpt_dir: str, seed: int, swa_start: int):
    rows = load_log_rows(log_path)
    if not rows:
        print("  [plot] no data yet; skipped")
        return

    epochs = [r["epoch"] for r in rows]
    lrs = [r["lr"] for r in rows]
    losses = [r["train_loss"] for r in rows]
    train_accs = [r["train_acc"] for r in rows]
    top1s = [r["val_top1"] for r in rows]
    sc_density = [r["val_sc_density"] for r in rows]

    plot_dir = os.path.join(ckpt_dir, f"plots_seed{seed}")
    os.makedirs(plot_dir, exist_ok=True)

    style = {"linewidth": 1.5}
    swa_color = "#F59E0B"

    def decorate(ax, ylabel, title):
        ax.axvline(swa_start, color=swa_color, linestyle="--",
                   linewidth=1.2, label=f"SWA start (ep {swa_start})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, losses, color="#6366F1", label="Train Loss", **style)
    decorate(ax, "Loss", f"Train Loss (seed={seed})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "train_loss.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, train_accs, color="#2563EB", linestyle="--",
            label="Train Acc (%)", **style)
    ax.plot(epochs, top1s, color="#0D9488", label="Val Top-1 (%)", **style)
    decorate(ax, "Accuracy (%)", f"Top-1 Accuracy (seed={seed})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "top1_acc.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, sc_density, color="#EA580C",
            label="Val SC Density (%)", **style)
    decorate(ax, "SC Density (%)", f"SC Density (seed={seed})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "sc_density.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, lrs, color="#7C3AED", label="LR", **style)
    decorate(ax, "Learning Rate", f"Learning Rate (seed={seed})")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "learning_rate.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
    fig.suptitle(f"PyramidNet-272 CIFAR-100 seed={seed}", fontsize=13)
    panels = [
        (losses, "#6366F1", "Train Loss", "Loss"),
        (top1s, "#0D9488", "Val Top-1 (%)", "Top-1 (%)"),
        (sc_density, "#EA580C", "Val SC Density (%)", "SC Density (%)"),
        (lrs, "#7C3AED", "LR", "LR"),
    ]
    for ax, (values, color, label, ylabel) in zip(axes, panels):
        ax.plot(epochs, values, color=color, label=label, **style)
        ax.axvline(swa_start, color=swa_color, linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "combined.png"), dpi=120)
    plt.close(fig)

    print(f"  [plot] saved curves to {plot_dir}")


def save_summary(summary_path, args, best_top1, best_sc_density,
                 final_top1, final_sc_density, swa_start, total_hours):
    with open(summary_path, "w") as f:
        f.write(f"seed: {args.seed}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"lam_coarse: {args.lam_coarse}\n")
        f.write(f"epsilon: {args.epsilon}\n")
        f.write(f"intra_ratio: {args.intra_ratio}\n")
        f.write(f"cutmix_prob: {args.cutmix_prob}\n")
        f.write(f"val_batch_mult: {args.val_batch_mult}\n")
        f.write(f"prefetch_factor: {args.prefetch_factor}\n")
        f.write(f"fast_cudnn: {args.fast_cudnn}\n")
        f.write(f"channels_last: {args.channels_last}\n")
        f.write(f"swa_start_ratio: {args.swa_start_ratio}\n")
        f.write(f"swa_start_epoch: {swa_start}\n")
        f.write(f"swa_lr: {args.swa_lr if args.swa_lr > 0 else args.lr * 0.1}\n")
        f.write(f"best_top1: {best_top1:.6f}\n")
        f.write(f"best_sc_density: {best_sc_density:.6f}\n")
        f.write(f"final_swa_top1: {final_top1:.6f}\n")
        f.write(f"final_swa_sc_density: {final_sc_density:.6f}\n")
        f.write(f"elapsed_hours: {total_hours:.4f}\n")


def main(args):
    set_seed(args.seed, args.fast_cudnn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.ckpt_dir, exist_ok=True)

    swa_start = max(int(args.epochs * args.swa_start_ratio), 1)
    swa_lr = args.swa_lr if args.swa_lr > 0 else args.lr * 0.1

    print(f"\n{'=' * 64}")
    print("  PyramidNet-272 CIFAR-100 server training")
    print(f"  Device       : {device}")
    print(f"  Seed         : {args.seed}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  SWA start    : epoch {swa_start} (ratio={args.swa_start_ratio})")
    print(f"  SWA LR       : {swa_lr}")
    print(f"  Val batch x  : {args.val_batch_mult}")
    print(f"  fast_cudnn   : {args.fast_cudnn}")
    print(f"  channels_last: {args.channels_last}")
    print(f"  skip_eval    : {args.skip_eval}")
    print(f"  Checkpoints  : {args.ckpt_dir}")
    print(f"{'=' * 64}\n")

    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cutmix=True,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
        seed=args.seed,
        val_batch_multiplier=args.val_batch_mult,
        prefetch_factor=args.prefetch_factor,
    )

    model = pyramidnet272(num_classes=100).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.2f} M")

    criterion = HierarchicalLoss(
        lam_coarse=args.lam_coarse,
        epsilon=args.epsilon,
        intra_ratio=args.intra_ratio,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_epochs=10,
        anneal_strategy="cos",
    )

    best_top1 = 0.0
    best_sc_density = 0.0
    val_top1 = 0.0
    val_sc_density = 0.0

    log_path = os.path.join(args.ckpt_dir, f"log_seed{args.seed}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr", "train_loss", "train_acc",
            "val_top1", "val_sc_density", "is_swa",
            "epoch_sec", "imgs_per_sec",
            "cuda_max_alloc_gib", "cuda_max_reserved_gib",
        ])

    perf_path = os.path.join(args.ckpt_dir, f"perf_seed{args.seed}.csv")
    with open(perf_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr", "is_swa", "did_eval",
            "train_sec", "epoch_sec",
            "train_imgs_per_sec", "epoch_imgs_per_sec",
            "cuda_max_alloc_gib", "cuda_max_reserved_gib",
        ])

    t0 = time.time()
    epoch_bar = tqdm(range(1, args.epochs + 1), ncols=110,
                     desc=f"Seed {args.seed}")

    for epoch in epoch_bar:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        epoch_t0 = time.time()
        tqdm.write(f"[phase] train epoch={epoch}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, total_epochs=args.epochs,
            channels_last=args.channels_last,
        )

        is_swa = epoch >= swa_start
        if is_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed_h = (time.time() - t0) / 3600
        eta_h = elapsed_h / epoch * (args.epochs - epoch) if epoch > 0 else 0

        should_eval = (
            not args.skip_eval
            and (epoch % args.eval_interval == 0 or epoch == args.epochs)
        )
        train_sec = time.time() - epoch_t0
        if should_eval:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            eval_model = swa_model if is_swa else model
            if is_swa:
                tqdm.write(f"[phase] update_bn epoch={epoch}")
                update_bn(train_loader, swa_model, device=device)
            tqdm.write(f"[phase] eval epoch={epoch}")
            val_top1, val_sc_density = evaluate(
                eval_model, val_loader, device,
                channels_last=args.channels_last,
            )
            if device.type == "cuda":
                cuda_max_alloc_gib = torch.cuda.max_memory_allocated() / 1024**3
                cuda_max_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
            else:
                cuda_max_alloc_gib = 0.0
                cuda_max_reserved_gib = 0.0
            epoch_sec = time.time() - epoch_t0
            imgs_per_sec = len(train_loader.dataset) / epoch_sec if epoch_sec > 0 else 0.0
            append_log(
                log_path, epoch, cur_lr, train_loss, train_acc,
                val_top1, val_sc_density, is_swa,
                epoch_sec, imgs_per_sec, cuda_max_alloc_gib,
                cuda_max_reserved_gib,
            )

            if val_top1 > best_top1:
                best_top1 = val_top1
                best_sc_density = val_sc_density
                best_state = (
                    swa_model.module.state_dict()
                    if is_swa else model.state_dict()
                )
                ckpt_path = os.path.join(
                    args.ckpt_dir, f"best_seed{args.seed}.pth")
                torch.save({
                    "epoch": epoch,
                    "best_state": state_dict_to_cpu(best_state),
                    "best_is_swa": is_swa,
                    "val_top1": val_top1,
                    "val_sc_density": val_sc_density,
                    "seed": args.seed,
                    "args": vars(args),
                }, ckpt_path)
                tqdm.write(f"  Best saved: top1={val_top1 * 100:.2f}% "
                          f"sc_density={val_sc_density * 100:.2f}% [{ckpt_path}]")

        if args.plot_interval > 0 and epoch % args.plot_interval == 0:
            plot_training_curves(log_path, args.ckpt_dir, args.seed, swa_start)

        if device.type == "cuda":
            cuda_max_alloc_gib = torch.cuda.max_memory_allocated() / 1024**3
            cuda_max_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
        else:
            cuda_max_alloc_gib = 0.0
            cuda_max_reserved_gib = 0.0
        epoch_sec = time.time() - epoch_t0
        train_images = len(train_loader) * args.batch_size
        train_imgs_per_sec = train_images / train_sec if train_sec > 0 else 0.0
        imgs_per_sec = len(train_loader.dataset) / epoch_sec if epoch_sec > 0 else 0.0
        append_perf_log(
            perf_path, epoch, cur_lr, is_swa, should_eval,
            train_sec, epoch_sec, train_imgs_per_sec, imgs_per_sec,
            cuda_max_alloc_gib, cuda_max_reserved_gib,
        )
        tqdm.write(
            f"[epoch] {epoch}/{args.epochs} "
            f"time={epoch_sec:.1f}s train={train_sec:.1f}s "
            f"imgs/s={train_imgs_per_sec:.1f} "
            f"cuda_alloc={cuda_max_alloc_gib:.2f}GiB "
            f"cuda_reserved={cuda_max_reserved_gib:.2f}GiB"
        )

        swa_flag = "SWA" if is_swa else "-"
        epoch_bar.set_postfix({
            "loss": f"{train_loss:.4f}",
            "acc": f"{train_acc * 100:.1f}%",
            "top1": f"{val_top1 * 100:.2f}%",
            "sc": f"{val_sc_density * 100:.2f}%",
            "lr": f"{cur_lr:.5f}",
            "eta": f"{eta_h:.1f}h",
            "mode": swa_flag,
        })

    epoch_bar.close()

    if args.skip_eval:
        print("[skip_eval] Skipping final SWA BN update/evaluation/checkpoint.")
        print("[skip_eval] No metric-based checkpoint or summary was written.")
        plot_training_curves(log_path, args.ckpt_dir, args.seed, swa_start)
        return

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[phase] final_swa_update_bn", flush=True)
    update_bn(train_loader, swa_model, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[phase] final_swa_eval", flush=True)
    final_top1, final_sc_density = evaluate(
        swa_model, val_loader, device,
        channels_last=args.channels_last,
    )
    elapsed_h = (time.time() - t0) / 3600

    print(f"\n{'=' * 64}")
    print(f"  Final SWA Results (seed={args.seed})")
    print(f"  Top-1 Accuracy : {final_top1 * 100:.2f}%")
    print(f"  SC Density     : {final_sc_density * 100:.2f}%")
    print(f"{'=' * 64}\n")

    swa_path = os.path.join(args.ckpt_dir, f"swa_final_seed{args.seed}.pth")
    torch.save(state_dict_to_cpu(swa_model.module.state_dict()), swa_path)
    print(f"SWA checkpoint saved: {swa_path}")

    summary_path = os.path.join(args.ckpt_dir, f"summary_seed{args.seed}.txt")
    save_summary(
        summary_path, args, best_top1, best_sc_density,
        final_top1, final_sc_density, swa_start, elapsed_h,
    )
    print(f"Summary saved: {summary_path}")

    plot_training_curves(log_path, args.ckpt_dir, args.seed, swa_start)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PyramidNet-272 on CIFAR-100 (server, no wandb)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.125)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--swa_start_ratio", type=float, default=0.75)
    parser.add_argument("--swa_lr", type=float, default=0.0125)
    parser.add_argument("--lam_coarse", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--intra_ratio", type=float, default=0.5)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_batch_mult", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--plot_interval", type=int, default=100)
    parser.add_argument("--fast_cudnn", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--channels_last", action="store_true", default=False)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
