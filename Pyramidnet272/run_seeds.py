"""
Run PyramidNet-272 training for multiple seeds and summarize CSV metrics.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def add_if_set(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def build_train_cmd(seed, args):
    cmd = [
        sys.executable,
        "train_server.py",
        "--seed",
        str(seed),
    ]

    add_if_set(cmd, "--epochs", args.epochs)
    add_if_set(cmd, "--batch_size", args.batch_size)
    add_if_set(cmd, "--lr", args.lr)
    add_if_set(cmd, "--weight_decay", args.weight_decay)
    add_if_set(cmd, "--swa_start_ratio", args.swa_start_ratio)
    add_if_set(cmd, "--swa_lr", args.swa_lr)
    add_if_set(cmd, "--lam_coarse", args.lam_coarse)
    add_if_set(cmd, "--epsilon", args.epsilon)
    add_if_set(cmd, "--intra_ratio", args.intra_ratio)
    add_if_set(cmd, "--cutmix_prob", args.cutmix_prob)
    add_if_set(cmd, "--data_root", args.data_root)
    add_if_set(cmd, "--ckpt_dir", args.ckpt_dir)
    add_if_set(cmd, "--num_workers", args.num_workers)
    add_if_set(cmd, "--val_batch_mult", args.val_batch_mult)
    add_if_set(cmd, "--prefetch_factor", args.prefetch_factor)
    add_if_set(cmd, "--eval_interval", args.eval_interval)
    add_if_set(cmd, "--plot_interval", args.plot_interval)
    if args.fast_cudnn:
        cmd.append("--fast_cudnn")
    if args.channels_last:
        cmd.append("--channels_last")
    return cmd


def run_seed(seed, args):
    cmd = build_train_cmd(seed, args)

    print(f"\n{'=' * 60}")
    print(f"  Running seed {seed}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR, text=True)
    if result.returncode != 0:
        print(f"[ERROR] seed {seed} failed with return code {result.returncode}")
        return False
    return True


def read_last_metrics(log_path):
    if not log_path.exists():
        return None

    with log_path.open(newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row.get("epoch", "").isdigit()
        ]

    if not rows:
        return None

    last = rows[-1]
    return float(last["val_top1"]), float(last["val_sc_density"])


def resolve_ckpt_dir(path):
    ckpt_dir = Path(path)
    if not ckpt_dir.is_absolute():
        ckpt_dir = SCRIPT_DIR / ckpt_dir
    return ckpt_dir


def main(args):
    top1_list = []
    sc_density_list = []
    ckpt_dir = resolve_ckpt_dir(args.ckpt_dir or "./checkpoints")

    for seed in args.seeds:
        ok = run_seed(seed, args)
        if not ok:
            continue

        log_path = ckpt_dir / f"log_seed{seed}.csv"
        metrics = read_last_metrics(log_path)
        if metrics is None:
            print(f"[WARN] Log metrics not found: {log_path}")
            continue

        top1, sc_density = metrics
        top1_list.append(top1)
        sc_density_list.append(sc_density)
        print(f"Seed {seed} -> Top-1: {top1 * 100:.2f}%  "
              f"SC Density: {sc_density * 100:.2f}%")

    print(f"\n{'=' * 60}")
    if top1_list:
        print(f"Mean Top-1      : {np.mean(top1_list) * 100:.2f}% "
              f"+/- {np.std(top1_list) * 100:.2f}%")
        print(f"Mean SC Density : {np.mean(sc_density_list) * 100:.2f}% "
              f"+/- {np.std(sc_density_list) * 100:.2f}%")
    else:
        print("No completed seed metrics found.")
    print("Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--swa_start_ratio", type=float, default=None)
    parser.add_argument("--swa_lr", type=float, default=None)
    parser.add_argument("--lam_coarse", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--intra_ratio", type=float, default=None)
    parser.add_argument("--cutmix_prob", type=float, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_batch_mult", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--plot_interval", type=int, default=None)
    parser.add_argument("--fast_cudnn", action="store_true", default=False)
    parser.add_argument("--channels_last", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
