"""
Evaluate a PyramidNet-272 checkpoint on CIFAR-100.

Metrics:
  - Top-1 Accuracy
  - SC Density: fraction of top-5 predictions in the target superclass
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.cifar100 import CIFAR100_MEAN, CIFAR100_STD, FINE_TO_COARSE
from models.pyramidnet import pyramidnet272


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_raw_state_dict(obj):
    return (
        isinstance(obj, dict)
        and bool(obj)
        and all(torch.is_tensor(value) for value in obj.values())
    )


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def select_state_dict(ckpt):
    if is_raw_state_dict(ckpt):
        return clean_state_dict(ckpt), "raw state_dict"

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")

    if "best_state" in ckpt:
        return clean_state_dict(ckpt["best_state"]), "best_state"
    if "model_state" in ckpt:
        return clean_state_dict(ckpt["model_state"]), "model_state"
    if "swa_state" in ckpt:
        return clean_state_dict(ckpt["swa_state"]), "swa_state"

    raise KeyError("Checkpoint has no best_state, model_state, or swa_state")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_logits = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs = F.softmax(all_logits, dim=-1)

    preds = probs.argmax(dim=1)
    top1_acc = (preds == all_labels).float().mean().item()

    mapping = torch.tensor(FINE_TO_COARSE, dtype=torch.long, device=DEVICE)
    _, top5_preds = probs.topk(5, dim=1)
    true_coarse = mapping[all_labels]
    top5_coarse = mapping[top5_preds]
    sc_density = (
        top5_coarse.eq(true_coarse.unsqueeze(1))
        .float()
        .mean(dim=1)
        .mean()
        .item()
    )

    return top1_acc, sc_density


def main(args):
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_set = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=val_transform,
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = pyramidnet272(num_classes=100).to(DEVICE)
    ckpt = safe_torch_load(args.ckpt, map_location=DEVICE)
    state, source = select_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Checkpoint loaded: {args.ckpt}")
    print(f"State source     : {source}")
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    top1_acc, sc_density = evaluate(model, loader)
    total = top1_acc * 100 + sc_density * 100

    print("\n===== Evaluation Results =====")
    print(f"Top-1 Accuracy : {top1_acc * 100:.2f}%")
    print(f"SC Density     : {sc_density * 100:.2f}%")
    print(f"Total Score    : {total:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
