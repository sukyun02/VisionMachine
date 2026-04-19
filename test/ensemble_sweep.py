"""
앙상블 최적 비율 탐색 스크립트.

모델별 TTA 확률을 한 번만 계산하고,
WRN 가중치(0.0~1.0)와 DHVT fusion-beta를 그리드 서치.

Usage:
  python ensemble_sweep.py \
    --wrn-checkpoint ../wrn.pth \
    --dhvt-checkpoint ../dhvt.pth \
    --data-path ../../data/cifar-100
"""
import argparse
import sys
import os

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from inference import (
    load_model, collect_logits, collect_tta_probs,
    collect_tta_logits_with_coarse, apply_fusion, compute_metrics,
)
from datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, CIFAR100_FINE_TO_COARSE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrn-checkpoint', type=str, required=True)
    parser.add_argument('--wrn-model', type=str, default='wrn_28_10')
    parser.add_argument('--dhvt-checkpoint', type=str, required=True)
    parser.add_argument('--dhvt-model', type=str, default='dhvt_tiny_cifar_patch4')
    parser.add_argument('--data-path', type=str, default='../../data/cifar-100')
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-superclasses', type=int, default=20)
    # sweep params
    parser.add_argument('--weight-step', type=float, default=0.05,
                        help='Step size for WRN weight sweep (default 0.05)')
    parser.add_argument('--beta-values', type=str, default='0.0,0.1,0.2,0.3,0.5,0.7,1.0',
                        help='Comma-separated fusion-beta values to sweep')
    args = parser.parse_args()

    device = torch.device(args.device)
    fine_to_coarse_t = torch.tensor(CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device)
    beta_values = [float(b) for b in args.beta_values.split(',')]

    # --- 타겟 로드 ---
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    test_ds = datasets.CIFAR100(args.data_path, train=False, transform=normalize, download=True)
    target_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)
    targets = torch.cat([t for _, t in target_loader]).to(device)

    # ------------------------------------------------------------------ WRN
    print("\n" + "="*60)
    print(f"=== WRN ({args.wrn_model}) TTA 수집 중...")
    print("="*60)
    wrn = load_model(args.wrn_model, args.wrn_checkpoint, device)
    wrn_probs_tta = collect_tta_probs(wrn, args.data_path, device,
                                      args.input_size, args.batch_size)
    del wrn
    torch.cuda.empty_cache()

    # 단독 성능
    a1, a5, sc = compute_metrics(wrn_probs_tta, targets, fine_to_coarse_t)
    print(f"\n[WRN TTA 단독]  Acc@1={a1:.2f}%  Acc@5={a5:.2f}%  SC={sc:.2f}%")

    # ----------------------------------------------------------------- DHVT
    print("\n" + "="*60)
    print(f"=== DHVT ({args.dhvt_model}) TTA 수집 중...")
    print("="*60)
    dhvt = load_model(args.dhvt_model, args.dhvt_checkpoint, device,
                      num_superclasses=args.num_superclasses)
    fine_logits, coarse_logits = collect_tta_logits_with_coarse(
        dhvt, args.data_path, device, args.input_size, args.batch_size)
    fine_logits = fine_logits.to(device)
    coarse_logits = coarse_logits.to(device)
    del dhvt
    torch.cuda.empty_cache()

    # beta=0 단독 성능
    dhvt_probs_no_fusion = F.softmax(fine_logits, dim=1)
    a1, a5, sc = compute_metrics(dhvt_probs_no_fusion, targets, fine_to_coarse_t)
    print(f"\n[DHVT TTA 단독 (β=0)]  Acc@1={a1:.2f}%  Acc@5={a5:.2f}%  SC={sc:.2f}%")

    # ---------------------------------------------------------------- SWEEP
    print("\n" + "="*60)
    print("=== 그리드 서치: WRN 가중치 × fusion-beta")
    print("="*60)
    print(f"{'WRN_w':>6}  {'DHVT_w':>6}  {'beta':>5}  {'Acc@1':>7}  {'Acc@5':>7}  {'SC':>7}  {'AVG':>7}")
    print("-"*61)

    best = {'avg': 0, 'row': ''}
    step = args.weight_step
    weights = [round(w * step, 4) for w in range(int(1.0 / step) + 1)]

    for beta in beta_values:
        if beta == 0.0:
            dhvt_probs = dhvt_probs_no_fusion
        else:
            fused = apply_fusion(fine_logits, coarse_logits, fine_to_coarse_t, beta)
            dhvt_probs = F.softmax(fused, dim=1)

        for w in weights:
            ensemble = w * wrn_probs_tta + (1 - w) * dhvt_probs
            a1, a5, sc = compute_metrics(ensemble, targets, fine_to_coarse_t)
            avg = (a1 + sc) / 2
            row = f"{w:>6.2f}  {1-w:>6.2f}  {beta:>5.1f}  {a1:>7.2f}  {a5:>7.2f}  {sc:>7.2f}  {avg:>7.2f}"
            print(row)
            if avg > best['avg']:
                best = {'acc1': a1, 'acc5': a5, 'sc': sc, 'avg': avg,
                        'wrn_w': w, 'dhvt_w': 1 - w, 'beta': beta, 'row': row}

    print("\n" + "="*60)
    print("★ 최적 조합 (Acc@1 기준)")
    print(f"  WRN weight : {best['wrn_w']:.2f}")
    print(f"  DHVT weight: {best['dhvt_w']:.2f}")
    print(f"  fusion-beta: {best['beta']}")
    print(f"  Acc@1={best['acc1']:.2f}%  Acc@5={best['acc5']:.2f}%  SC={best['sc']:.2f}%  AVG={best['avg']:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
