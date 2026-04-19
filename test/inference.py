"""
Ensemble inference with TTA for WRN + DHVT on CIFAR-100.

TTA augmentations:
  - original
  - horizontal flip
  - 4-corner crops (with model-specific padding)
  - optional horizontal flips of corner crops

Ensemble strategy: logit averaging across TTA variants, then weighted
probability or geometric fusion across models.

Usage:
  python inference.py \
    --wrn-checkpoint best_wrn_28_10.pth \
    --dhvt-checkpoint best_dhvt_tiny.pth \
    --data-path ./data/cifar-100-python
"""
import argparse

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models.wideresnet import wrn_28_10
from models.dhvt import dhvt_tiny_cifar_patch4, dhvt_small_cifar_patch4
from datasets import (
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD,
    CIFAR100_MEAN, CIFAR100_STD,
    CIFAR100_FINE_TO_COARSE,
)


# ---------------------------------------------------------------------------
# TTA transforms
# ---------------------------------------------------------------------------

def get_tta_transforms(input_size=32, padding=4,
                       mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
                       padding_mode='constant', include_flipped_corners=True):
    """Returns list of (name, transform) for test-time augmentation."""
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
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
                padded = transforms.functional.pad(
                    img, padding, padding_mode=padding_mode)
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


def build_eval_transform(mean, std):
    """Build a no-TTA evaluation transform."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make_loader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def autocast_for(device):
    device_type = torch.device(device).type
    return torch.amp.autocast(device_type, enabled=device_type == 'cuda')


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
        with autocast_for(device):
            out = model(images)
        all_logits.append(out.float())
    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def collect_tta_probs(model, dataset, tta_transforms, device, batch_size=256):
    """Collect probabilities after averaging logits across TTA views."""
    all_logits = []

    original_transform = dataset.transform
    try:
        for name, tfm in tta_transforms:
            dataset.transform = tfm
            logits = collect_logits(model, make_loader(dataset, batch_size), device)
            all_logits.append(logits)
            print(f"    {name}: done")
    finally:
        dataset.transform = original_transform

    # TTA is logit avg -> softmax, not probability averaging across views.
    return F.softmax(torch.stack(all_logits, dim=0).mean(dim=0), dim=1)


@torch.no_grad()
def collect_tta_logits_with_coarse(model, dataset, tta_transforms, device, batch_size=256):
    """Collect TTA-averaged fine and coarse logits from a DHVT model with aux head."""
    fine_views, coarse_views = [], []

    original_transform = dataset.transform
    try:
        for name, tfm in tta_transforms:
            dataset.transform = tfm
            fine_all, coarse_all = [], []
            for images, _ in make_loader(dataset, batch_size):
                images = images.to(device, non_blocking=True)
                with autocast_for(device):
                    features = model.forward_features(images)
                    fine_logits = model.head(features)
                    coarse_logits = model.head_superclass(features)
                fine_all.append(fine_logits.float())
                coarse_all.append(coarse_logits.float())
            fine_views.append(torch.cat(fine_all, dim=0))
            coarse_views.append(torch.cat(coarse_all, dim=0))
            print(f"    {name}: done")
    finally:
        dataset.transform = original_transform

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


def combine_model_probs(wrn_probs, dhvt_probs, wrn_weight, method):
    """Combine model probabilities with arithmetic or geometric fusion."""
    if method == 'prob':
        return wrn_weight * wrn_probs + (1 - wrn_weight) * dhvt_probs

    if method == 'geom':
        eps = 1e-12
        log_probs = (
            wrn_weight * torch.log(wrn_probs.clamp_min(eps))
            + (1 - wrn_weight) * torch.log(dhvt_probs.clamp_min(eps))
        )
        return F.softmax(log_probs, dim=1)

    raise ValueError(f"Unknown ensemble method: {method}")


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


def resolve_state_dict(ckpt, prefer_ema=False):
    """Resolve common checkpoint layouts to a model state_dict."""
    if not isinstance(ckpt, dict):
        return ckpt

    if prefer_ema and isinstance(ckpt.get('model_ema'), dict):
        return ckpt['model_ema']

    for key in ('model_state_dict', 'state_dict', 'model'):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value

    return ckpt


def load_model(model_name, checkpoint_path, device, num_classes=100,
               num_superclasses=20, prefer_ema=False):
    """Load a model from checkpoint."""
    if model_name.startswith('wrn'):
        model = MODEL_REGISTRY[model_name](num_classes=num_classes)
    else:
        model = MODEL_REGISTRY[model_name](
            num_classes=num_classes, num_superclasses=num_superclasses)

    model = model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = resolve_state_dict(ckpt, prefer_ema=prefer_ema)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected}")

    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    ema_note = " using EMA weights" if prefer_ema and isinstance(ckpt, dict) and 'model_ema' in ckpt else ""
    print(f"  Loaded {model_name} from {checkpoint_path} (epoch {epoch}){ema_note}")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ensemble + TTA inference for CIFAR-100")
    parser.add_argument('--wrn-checkpoint', type=str, default='../WRN/checkpoints/best_wrn_28_10.pth',
                        help='Path to WRN checkpoint (.pth)')
    parser.add_argument('--wrn-model', type=str, default='wrn_28_10',
                        choices=['wrn_28_10'])
    parser.add_argument('--dhvt-checkpoint', type=str, default='../DHVT/output/best.pth',
                        help='Path to DHVT checkpoint (.pth)')
    parser.add_argument('--dhvt-model', type=str, default='dhvt_tiny_cifar_patch4',
                        choices=['dhvt_tiny_cifar_patch4', 'dhvt_small_cifar_patch4'])
    parser.add_argument('--data-path', type=str, default='../data/cifar-100')
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-superclasses', type=int, default=20)
    parser.add_argument('--wrn-weight', type=float, default=0.6,
                        help='Weight for WRN in ensemble (DHVT gets 1 - this)')
    parser.add_argument('--ensemble-method', type=str, default='prob',
                        choices=['prob', 'geom'],
                        help='prob=weighted arithmetic mean, geom=normalized weighted geometric mean')
    parser.add_argument('--fusion-beta', type=float, default=0.5,
                        help='Hierarchical score fusion weight for DHVT aux head. '
                             '0=disabled. Recommended: 0.3 (balanced) ~ 1.0 (SC_Density focus). '
                             'score(c) = log p_fine(c) + beta * log p_coarse(sc(c))')
    parser.add_argument('--use-model-ema', action='store_true',
                        help='Load DHVT model_ema weights when present in the checkpoint')
    args = parser.parse_args()

    device = torch.device(args.device)
    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device)

    # --- Load targets ---
    target_ds = datasets.CIFAR100(args.data_path, train=False, download=True)
    targets = torch.tensor(target_ds.targets, dtype=torch.long, device=device)

    wrn_transform = build_eval_transform(CIFAR100_MEAN, CIFAR100_STD)
    dhvt_transform = build_eval_transform(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    wrn_tta = get_tta_transforms(
        args.input_size, mean=CIFAR100_MEAN, std=CIFAR100_STD, padding_mode='reflect')
    dhvt_tta = get_tta_transforms(
        args.input_size, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        padding_mode='constant')

    model_probs = {}

    # --- WRN ---
    if args.wrn_checkpoint:
        print(f"\n=== WRN ({args.wrn_model}) ===")
        wrn = load_model(args.wrn_model, args.wrn_checkpoint, device)
        wrn_test_ds = datasets.CIFAR100(
            args.data_path, train=False, transform=wrn_transform, download=True)

        # No TTA
        wrn_logits = collect_logits(wrn, make_loader(wrn_test_ds, args.batch_size), device)
        wrn_probs_no_tta = F.softmax(wrn_logits, dim=1)
        acc1, acc5, sc = compute_metrics(wrn_probs_no_tta, targets, fine_to_coarse_t)
        print_results("WRN - No TTA", acc1, acc5, sc)

        # TTA
        print("  Running TTA...")
        wrn_probs_tta = collect_tta_probs(
            wrn, wrn_test_ds, wrn_tta, device, args.batch_size)
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
                          num_superclasses=args.num_superclasses,
                          prefer_ema=args.use_model_ema)
        dhvt_test_ds = datasets.CIFAR100(
            args.data_path, train=False, transform=dhvt_transform, download=True)

        # No TTA
        dhvt_logits = collect_logits(dhvt, make_loader(dhvt_test_ds, args.batch_size), device)
        dhvt_probs_no_tta = F.softmax(dhvt_logits, dim=1)
        acc1, acc5, sc = compute_metrics(dhvt_probs_no_tta, targets, fine_to_coarse_t)
        print_results("DHVT - No TTA", acc1, acc5, sc)

        # TTA
        print("  Running TTA...")
        if use_fusion:
            fine_logits, coarse_logits = collect_tta_logits_with_coarse(
                dhvt, dhvt_test_ds, dhvt_tta, device, args.batch_size)
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
            dhvt_probs_tta = collect_tta_probs(
                dhvt, dhvt_test_ds, dhvt_tta, device, args.batch_size)
            acc1, acc5, sc = compute_metrics(dhvt_probs_tta, targets, fine_to_coarse_t)
            print_results("DHVT - TTA", acc1, acc5, sc)

        model_probs['dhvt'] = dhvt_probs_tta
        del dhvt

    # --- Ensemble ---
    if len(model_probs) == 2:
        w = args.wrn_weight
        ensemble_probs = combine_model_probs(
            model_probs['wrn'], model_probs['dhvt'], w, args.ensemble_method)
        acc1, acc5, sc = compute_metrics(ensemble_probs, targets, fine_to_coarse_t)
        print(f"\n{'='*50}")
        print_results(
            f"Ensemble ({args.ensemble_method}, WRN={w:.1f}, DHVT={1-w:.1f}) + TTA",
            acc1, acc5, sc)
        print(f"{'='*50}")
    elif len(model_probs) == 1:
        name = list(model_probs.keys())[0]
        print(f"\n  (Only one model provided — skipping ensemble, showing {name} + TTA above)")
    else:
        print("\n  No checkpoints provided. Use --wrn-checkpoint and/or --dhvt-checkpoint.")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
