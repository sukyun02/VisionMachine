"""
Ensemble inference with TTA for WRN + DHVT on CIFAR-100.

TTA augmentations:
  - original
  - horizontal flip
  - 4-corner crops (with padding)

Ensemble strategy: weighted probability averaging across models & TTA variants.

Usage:
  python inference.py \
    --wrn-checkpoint best_wrn_28_10.pth \
    --dhvt-checkpoint best_dhvt_tiny.pth \
    --data-path ./data/cifar-100-python
"""
import argparse

import torch
import torch.nn.functional as F


def _load_ckpt(path, map_location='cpu'):
    """확장자에 따라 .pth 또는 .safetensors 체크포인트를 로드한다."""
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        device = str(map_location) if map_location else 'cpu'
        return load_file(str(path), device=device)
    return torch.load(str(path), map_location=map_location, weights_only=False)
from torchvision import datasets, transforms

from models.wideresnet import wrn_28_10
from models.dhvt import dhvt_tiny_cifar_patch4, dhvt_small_cifar_patch4
from datasets import (
    CIFAR100_MEAN, CIFAR100_STD,
    CIFAR100_FINE_TO_COARSE,
)


# ---------------------------------------------------------------------------
# TTA transforms
# ---------------------------------------------------------------------------

def get_tta_transforms(input_size=32, padding=4, include_flipped_corners=True):
    """Returns list of (name, transform) for test-time augmentation."""
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    tta_list = [
        ("original", transforms.Compose([normalize])),
        ("hflip", transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            normalize,
        ])),
    ]

    # 4 corner crops with padding
    crop_specs = [
        ("crop_tl", 0, 0),
        ("crop_tr", 0, 2 * padding),
        ("crop_bl", 2 * padding, 0),
        ("crop_br", 2 * padding, 2 * padding),
    ]
    for name, top, left in crop_specs:
        def make_crop_fn(t, l):
            def crop_fn(img):
                padded = transforms.functional.pad(img, padding)
                return transforms.functional.crop(padded, t, l, input_size, input_size)
            return crop_fn

        crop = transforms.Lambda(make_crop_fn(top, left))
        tta_list.append((name, transforms.Compose([crop, normalize])))
        if include_flipped_corners:
            tta_list.append((f"{name}_hflip", transforms.Compose([
                crop,
                transforms.RandomHorizontalFlip(p=1.0),
                normalize,
            ])))

    return tta_list


# ---------------------------------------------------------------------------
# Collect logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_logits(model, dataloader, device):
    """Run model on dataloader and return concatenated logits (N, C)."""
    model.eval()
    all_logits = []
    for images, _ in dataloader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            out = model(images)
        all_logits.append(out.float())
    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def collect_tta_probs(model, data_path, device, input_size=32, batch_size=256,
                      include_flipped_corners=True):
    """Collect TTA logit-averaged probabilities for a single model."""
    tta_transforms = get_tta_transforms(input_size, include_flipped_corners=include_flipped_corners)
    all_logits = []

    for name, tfm in tta_transforms:
        ds = datasets.CIFAR100(data_path, train=False, transform=tfm, download=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        logits = collect_logits(model, loader, device)
        all_logits.append(logits)
        print(f"    {name}: done")

    # Logit avg → softmax (better than prob avg)
    return F.softmax(torch.stack(all_logits, dim=0).mean(dim=0), dim=1)


@torch.no_grad()
def collect_tta_logits_with_coarse(model, data_path, device, input_size=32, batch_size=256,
                                   include_flipped_corners=True):
    """Collect TTA-averaged fine and coarse logits from a DHVT model with aux head."""
    tta_transforms = get_tta_transforms(input_size, include_flipped_corners=include_flipped_corners)
    fine_views, coarse_views = [], []

    for name, tfm in tta_transforms:
        ds = datasets.CIFAR100(data_path, train=False, transform=tfm, download=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        fine_all, coarse_all = [], []
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                features = model.forward_features(images)
                fine_logits = model.head(features)
                coarse_logits = model.head_superclass(features)
            fine_all.append(fine_logits.float())
            coarse_all.append(coarse_logits.float())
        fine_views.append(torch.cat(fine_all, dim=0))
        coarse_views.append(torch.cat(coarse_all, dim=0))
        print(f"    {name}: done")

    # Logit avg across TTA views (better than prob avg for fusion)
    fine_avg = torch.stack(fine_views, dim=0).mean(dim=0)    # (N, 100)
    coarse_avg = torch.stack(coarse_views, dim=0).mean(dim=0)  # (N, 20)
    return fine_avg, coarse_avg


def apply_fusion(fine_logits, coarse_logits, fine_to_coarse_t, beta):
    """Hierarchical score fusion: log p_fine(c) + beta * log p_coarse(sc(c))."""
    log_p_fine = F.log_softmax(fine_logits, dim=1)
    log_p_coarse = F.log_softmax(coarse_logits, dim=1)
    log_p_coarse_broadcast = log_p_coarse[:, fine_to_coarse_t]  # (N, 100)
    return log_p_fine + beta * log_p_coarse_broadcast


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(probs, targets, fine_to_coarse_t):
    """Compute Acc@1, Acc@5, and SC Density."""
    _, top1_pred = probs.topk(1, dim=1)
    _, top5_pred = probs.topk(5, dim=1)

    acc1 = (top1_pred.squeeze(1) == targets).float().mean().item() * 100
    acc5 = (top5_pred == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100

    # Superclass density
    target_sc = fine_to_coarse_t[targets]
    pred_sc = fine_to_coarse_t[top5_pred]
    sc_density = (pred_sc == target_sc.unsqueeze(1)).float().mean(dim=1).mean().item() * 100

    return acc1, acc5, sc_density


def print_results(label, acc1, acc5, sc_density):
    print(f"\n  [{label}]")
    print(f"    Acc@1: {acc1:.2f}%  Acc@5: {acc5:.2f}%  SC_Density: {sc_density:.2f}%")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'wrn_28_10': wrn_28_10,
    'dhvt_tiny_cifar_patch4': dhvt_tiny_cifar_patch4,
    'dhvt_small_cifar_patch4': dhvt_small_cifar_patch4,
}


def load_model(model_name, checkpoint_path, device, num_classes=100, num_superclasses=20):
    """Load a model from checkpoint."""
    if model_name.startswith('wrn'):
        model = MODEL_REGISTRY[model_name](num_classes=num_classes)
    else:
        model = MODEL_REGISTRY[model_name](
            num_classes=num_classes, num_superclasses=num_superclasses)

    model = model.to(device)
    ckpt = _load_ckpt(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected}")

    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    print(f"  Loaded {model_name} from {checkpoint_path} (epoch {epoch})")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ensemble + TTA inference for CIFAR-100")
    parser.add_argument('--wrn-checkpoint', type=str, default='../WRN/checkpoints/best_wrn_28_10.pth',
                        help='Path to WRN checkpoint (.pth or .safetensors)')
    parser.add_argument('--wrn-model', type=str, default='wrn_28_10',
                        choices=['wrn_28_10'])
    parser.add_argument('--dhvt-checkpoint', type=str,
                        default='../DHVT/output/best.pth',
                        help='Path to DHVT checkpoint (.pth or .safetensors)')
    parser.add_argument('--dhvt-model', type=str, default='dhvt_tiny_cifar_patch4',
                        choices=['dhvt_tiny_cifar_patch4', 'dhvt_small_cifar_patch4'])
    parser.add_argument('--data-path', type=str, default='../data/cifar-100')
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-superclasses', type=int, default=20)
    parser.add_argument('--wrn-weight', type=float, default=0.85,
                        help='Weight for WRN in ensemble (DHVT gets 1 - this)')
    parser.add_argument('--fusion-beta', type=float, default=1.0,
                        help='Hierarchical score fusion weight for DHVT aux head. '
                             '0=disabled. Recommended: 0.3 (balanced) ~ 1.0 (SC_Density focus). '
                             'score(c) = log p_fine(c) + beta * log p_coarse(sc(c))')
    args = parser.parse_args()

    device = torch.device(args.device)
    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device)

    # --- Load targets ---
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_ds = datasets.CIFAR100(args.data_path, train=False, transform=normalize, download=True)
    target_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    targets = torch.cat([t for _, t in target_loader]).to(device)

    model_probs = {}

    # --- WRN ---
    if args.wrn_checkpoint:
        print(f"\n=== WRN ({args.wrn_model}) ===")
        wrn = load_model(args.wrn_model, args.wrn_checkpoint, device)

        # No TTA
        no_tta_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        wrn_logits = collect_logits(wrn, no_tta_loader, device)
        wrn_probs_no_tta = F.softmax(wrn_logits, dim=1)
        acc1, acc5, sc = compute_metrics(wrn_probs_no_tta, targets, fine_to_coarse_t)
        print_results("WRN - No TTA", acc1, acc5, sc)

        # TTA
        print("  Running TTA...")
        wrn_probs_tta = collect_tta_probs(wrn, args.data_path, device,
                                          args.input_size, args.batch_size)
        acc1, acc5, sc = compute_metrics(wrn_probs_tta, targets, fine_to_coarse_t)
        print_results("WRN - TTA", acc1, acc5, sc)
        model_probs['wrn'] = wrn_probs_tta
        del wrn

    # --- DHVT ---
    if args.dhvt_checkpoint:
        has_aux = args.num_superclasses > 0
        use_fusion = has_aux and args.fusion_beta != 0.0
        print(f"\n=== DHVT ({args.dhvt_model}) ===")
        if use_fusion:
            print(f"  Hierarchical score fusion: beta={args.fusion_beta}")
        dhvt = load_model(args.dhvt_model, args.dhvt_checkpoint, device,
                          num_superclasses=args.num_superclasses)

        # No TTA
        no_tta_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        dhvt_logits = collect_logits(dhvt, no_tta_loader, device)
        dhvt_probs_no_tta = F.softmax(dhvt_logits, dim=1)
        acc1, acc5, sc = compute_metrics(dhvt_probs_no_tta, targets, fine_to_coarse_t)
        print_results("DHVT - No TTA", acc1, acc5, sc)

        # TTA
        print("  Running TTA...")
        if use_fusion:
            fine_logits, coarse_logits = collect_tta_logits_with_coarse(
                dhvt, args.data_path, device, args.input_size, args.batch_size)
            fine_logits = fine_logits.to(device)
            coarse_logits = coarse_logits.to(device)
            # Without fusion (β=0)
            dhvt_probs_tta = F.softmax(fine_logits, dim=1)
            acc1, acc5, sc = compute_metrics(dhvt_probs_tta, targets, fine_to_coarse_t)
            print_results("DHVT - TTA (no fusion)", acc1, acc5, sc)
            # With fusion
            fused_scores = apply_fusion(fine_logits, coarse_logits, fine_to_coarse_t, args.fusion_beta)
            dhvt_probs_tta = F.softmax(fused_scores, dim=1)
            acc1, acc5, sc = compute_metrics(dhvt_probs_tta, targets, fine_to_coarse_t)
            print_results(f"DHVT - TTA + fusion (β={args.fusion_beta})", acc1, acc5, sc)
        else:
            dhvt_probs_tta = collect_tta_probs(dhvt, args.data_path, device,
                                               args.input_size, args.batch_size)
            acc1, acc5, sc = compute_metrics(dhvt_probs_tta, targets, fine_to_coarse_t)
            print_results("DHVT - TTA", acc1, acc5, sc)

        model_probs['dhvt'] = dhvt_probs_tta
        del dhvt

    # --- Ensemble ---
    if len(model_probs) == 2:
        w = args.wrn_weight
        ensemble_probs = w * model_probs['wrn'] + (1 - w) * model_probs['dhvt']
        acc1, acc5, sc = compute_metrics(ensemble_probs, targets, fine_to_coarse_t)
        print(f"\n{'='*50}")
        print_results(f"Ensemble (WRN={w:.1f}, DHVT={1-w:.1f}) + TTA", acc1, acc5, sc)
        print(f"{'='*50}")
    elif len(model_probs) == 1:
        name = list(model_probs.keys())[0]
        print(f"\n  (Only one model provided — skipping ensemble, showing {name} + TTA above)")
    else:
        print("\n  No checkpoints provided. Use --wrn-checkpoint and/or --dhvt-checkpoint.")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
