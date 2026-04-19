"""
TTA + coarse-aware reranking combined evaluation.

Pipeline:
  1. For each TTA augmentation, collect (fine_logits, coarse_logits)
  2. Average across TTA (logit avg)
  3. Apply reranking with score = log p_fine + beta * log p_coarse(sc(c))
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
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
from datasets import CIFAR100_FINE_TO_COARSE
import vision_transformer  # register models


def get_tta_transforms(input_size=32, padding=4):
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    tta_list = [
        ("original", transforms.Compose([normalize])),
        ("hflip", transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0), normalize])),
    ]
    for name, crop_fn in [
        ("crop_tl", lambda img: transforms.functional.crop(
            transforms.functional.pad(img, padding), 0, 0, input_size, input_size)),
        ("crop_tr", lambda img: transforms.functional.crop(
            transforms.functional.pad(img, padding), 0, 2 * padding, input_size, input_size)),
        ("crop_bl", lambda img: transforms.functional.crop(
            transforms.functional.pad(img, padding), 2 * padding, 0, input_size, input_size)),
        ("crop_br", lambda img: transforms.functional.crop(
            transforms.functional.pad(img, padding), 2 * padding, 2 * padding, input_size, input_size)),
    ]:
        tta_list.append((name, transforms.Compose([
            transforms.Lambda(crop_fn), normalize])))
    return tta_list


@torch.no_grad()
def collect_logits(model, loader, device):
    fine_all, coarse_all, target_all = [], [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            features = model.forward_features(images)
            fine_logits = model.head(features)
            coarse_logits = model.head_superclass(features)
        fine_all.append(fine_logits.float().cpu())
        coarse_all.append(coarse_logits.float().cpu())
        target_all.append(target)
    return torch.cat(fine_all), torch.cat(coarse_all), torch.cat(target_all)


def metrics(scores, targets, fine_to_coarse_t):
    top1 = scores.argmax(dim=1)
    top5 = scores.topk(5, dim=1).indices
    acc1 = (top1 == targets).float().mean().item() * 100
    acc5 = (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100
    target_sc = fine_to_coarse_t[targets]
    pred_sc = fine_to_coarse_t[top5]
    sc_density = (pred_sc == target_sc.unsqueeze(1)).float().mean(dim=1).mean().item() * 100
    return acc1, acc5, sc_density


def sweep_beta(fine_logits, coarse_logits, targets, fine_to_coarse_t, betas, label):
    log_p_fine = F.log_softmax(fine_logits, dim=1)
    log_p_coarse = F.log_softmax(coarse_logits, dim=1)
    log_p_coarse_b = log_p_coarse[:, fine_to_coarse_t]

    print(f"\n=== {label} ===")
    print(f"{'beta':>6} | {'Acc@1':>7} {'Acc@5':>7} {'SC_D':>7} {'Sum':>8}")
    print("-" * 50)
    for beta in betas:
        scores = log_p_fine + beta * log_p_coarse_b
        a1, a5, scd = metrics(scores, targets, fine_to_coarse_t)
        print(f"{beta:>6.2f} | {a1:>7.2f} {a5:>7.2f} {scd:>7.2f} {a1+scd:>8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data/cifar-100-python')
    parser.add_argument('--model', type=str, default='dhvt_tiny_cifar_patch4')
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-superclasses', type=int, default=20)
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0])
    args = parser.parse_args()

    device = torch.device(args.device)
    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long)

    model = create_model(
        args.model, pretrained=False, num_classes=100,
        drop_rate=0.0, drop_path_rate=0.0, drop_block_rate=None,
        img_size=(args.input_size, args.input_size),
        num_superclasses=args.num_superclasses,
    ).to(device)
    ckpt = _load_ckpt(args.checkpoint, map_location=device)
    state_dict = ckpt['model'] if isinstance(ckpt.get('model'), dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?') if not str(args.checkpoint).endswith('.safetensors') else '?'
    print(f"Loaded epoch {epoch}")

    tta_transforms = get_tta_transforms(args.input_size)
    print(f"TTA augmentations: {[name for name, _ in tta_transforms]}")

    fine_views, coarse_views, targets_ref = [], [], None
    for name, tfm in tta_transforms:
        ds = datasets.CIFAR100(args.data_path, train=False, transform=tfm, download=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True)
        fine_l, coarse_l, target_l = collect_logits(model, loader, device)
        fine_views.append(fine_l)
        coarse_views.append(coarse_l)
        if targets_ref is None:
            targets_ref = target_l
        print(f"  {name}: done")

    # Stack: (V, N, 100), (V, N, 20)
    fine_stack = torch.stack(fine_views, dim=0)
    coarse_stack = torch.stack(coarse_views, dim=0)

    # --- Variant 1: No TTA (original only) ---
    sweep_beta(fine_stack[0], coarse_stack[0], targets_ref, fine_to_coarse_t,
               args.betas, "No TTA + rerank sweep")

    # --- Variant 2: TTA logit avg ---
    fine_avg = fine_stack.mean(dim=0)
    coarse_avg = coarse_stack.mean(dim=0)
    sweep_beta(fine_avg, coarse_avg, targets_ref, fine_to_coarse_t,
               args.betas, "TTA (logit avg) + rerank sweep")


if __name__ == '__main__':
    main()
