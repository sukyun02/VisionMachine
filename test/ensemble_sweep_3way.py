"""
Search WRN + PyramidNet + DHVT ensemble weights on CIFAR-100.

The expensive part, model TTA inference, is collected once per model. The script
then sweeps ensemble weights in memory and ranks results by SUM/AVG, where:

  SUM = Acc@1 + SC_Density
  AVG = (Acc@1 + SC_Density) / 2

Usage:
  uv run python test/ensemble_sweep_3way.py \
    --wrn-checkpoint WRN/checkpoints/back_best_wrn_28_10.pth \
    --dhvt-checkpoint DHVT/output/best.pth \
    --pyramidnet-checkpoint Pyramidnet272/checkpoints/best_seed42.pth \
    --data-path Pyramidnet272/data
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from inference import (  # noqa: E402
    apply_fusion,
    collect_tta_logits_with_coarse,
    collect_tta_probs,
    compute_metrics,
    load_model,
)
from datasets import CIFAR100_FINE_TO_COARSE, CIFAR100_MEAN, CIFAR100_STD  # noqa: E402


def parse_float_list(raw):
    return [float(item.strip()) for item in raw.split(',') if item.strip()]


def inclusive_float_range(start, stop, step):
    if step <= 0:
        raise ValueError(f"weight-step must be positive, got {step}")
    if start > stop:
        raise ValueError(f"range start must be <= stop, got {start} > {stop}")

    values = []
    value = start
    while value <= stop + (step / 2):
        values.append(round(value, 6))
        value += step
    return values


def print_metric_line(label, probs, targets, fine_to_coarse_t):
    acc1, acc5, sc = compute_metrics(probs, targets, fine_to_coarse_t)
    avg = (acc1 + sc) / 2
    print(
        f"{label}  Acc@1={acc1:.2f}%  Acc@5={acc5:.2f}%  "
        f"SC={sc:.2f}%  SUM={acc1 + sc:.2f}  AVG={avg:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Grid search WRN + PyramidNet + DHVT ensemble weights."
    )
    parser.add_argument('--wrn-checkpoint', type=str, required=True)
    parser.add_argument('--wrn-model', type=str, default='wrn_28_10')
    parser.add_argument('--pyramidnet-checkpoint', type=str, required=True)
    parser.add_argument('--pyramidnet-model', type=str, default='pyramidnet272')
    parser.add_argument('--dhvt-checkpoint', type=str, required=True)
    parser.add_argument('--dhvt-model', type=str, default='dhvt_tiny_cifar_patch4')
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-superclasses', type=int, default=20)
    parser.add_argument('--fusion-beta-values', type=str, default='1.0',
                        help='Comma-separated DHVT fusion beta values to try.')
    parser.add_argument('--wrn-min', type=float, default=0.70)
    parser.add_argument('--wrn-max', type=float, default=0.90)
    parser.add_argument('--pyramidnet-min', type=float, default=0.00)
    parser.add_argument('--pyramidnet-max', type=float, default=0.15)
    parser.add_argument('--weight-step', type=float, default=0.025)
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of best rows to print at the end.')
    parser.add_argument('--dhvt-six-view', action='store_true',
                        help='Use the older 6-view DHVT TTA instead of 10-view.')
    args = parser.parse_args()

    device = torch.device(args.device)
    fine_to_coarse_t = torch.tensor(
        CIFAR100_FINE_TO_COARSE, dtype=torch.long, device=device
    )
    beta_values = parse_float_list(args.fusion_beta_values)
    wrn_weights = inclusive_float_range(args.wrn_min, args.wrn_max, args.weight_step)
    pyramidnet_weights = inclusive_float_range(
        args.pyramidnet_min, args.pyramidnet_max, args.weight_step
    )

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_ds = datasets.CIFAR100(
        args.data_path, train=False, transform=normalize, download=True
    )
    target_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False
    )
    targets = torch.cat([target for _, target in target_loader]).to(device)

    print("\n" + "=" * 70)
    print(f"=== WRN ({args.wrn_model}) TTA 수집 중...")
    print("=" * 70)
    wrn = load_model(args.wrn_model, args.wrn_checkpoint, device)
    wrn_probs = collect_tta_probs(
        wrn, args.data_path, device, args.input_size, args.batch_size
    )
    del wrn
    torch.cuda.empty_cache()
    print_metric_line("[WRN TTA 단독]", wrn_probs, targets, fine_to_coarse_t)

    print("\n" + "=" * 70)
    print(f"=== PyramidNet ({args.pyramidnet_model}) TTA 수집 중...")
    print("=" * 70)
    pyramidnet = load_model(args.pyramidnet_model, args.pyramidnet_checkpoint, device)
    pyramidnet_probs = collect_tta_probs(
        pyramidnet, args.data_path, device, args.input_size, args.batch_size
    )
    del pyramidnet
    torch.cuda.empty_cache()
    print_metric_line("[PyramidNet TTA 단독]", pyramidnet_probs, targets, fine_to_coarse_t)

    print("\n" + "=" * 70)
    print(f"=== DHVT ({args.dhvt_model}) TTA 수집 중...")
    print("=" * 70)
    dhvt = load_model(
        args.dhvt_model,
        args.dhvt_checkpoint,
        device,
        num_superclasses=args.num_superclasses,
    )
    fine_logits, coarse_logits = collect_tta_logits_with_coarse(
        dhvt,
        args.data_path,
        device,
        args.input_size,
        args.batch_size,
        include_flipped_corners=not args.dhvt_six_view,
    )
    del dhvt
    torch.cuda.empty_cache()

    dhvt_probs_by_beta = {}
    for beta in beta_values:
        if beta == 0.0:
            dhvt_probs = F.softmax(fine_logits, dim=1)
        else:
            fused_scores = apply_fusion(fine_logits, coarse_logits, fine_to_coarse_t, beta)
            dhvt_probs = F.softmax(fused_scores, dim=1)
        dhvt_probs_by_beta[beta] = dhvt_probs
        print_metric_line(f"[DHVT TTA 단독 β={beta:g}]", dhvt_probs, targets, fine_to_coarse_t)

    print("\n" + "=" * 70)
    print("=== 3-way 가중치 그리드 서치")
    print("=" * 70)
    print(
        f"{'WRN_w':>6}  {'Pyr_w':>6}  {'DHVT_w':>6}  {'beta':>5}  "
        f"{'Acc@1':>7}  {'Acc@5':>7}  {'SC':>7}  {'SUM':>7}  {'AVG':>7}"
    )
    print("-" * 79)

    rows = []
    for beta, dhvt_probs in dhvt_probs_by_beta.items():
        for wrn_w in wrn_weights:
            for pyr_w in pyramidnet_weights:
                dhvt_w = round(1.0 - wrn_w - pyr_w, 10)
                if dhvt_w < -1e-9:
                    continue
                dhvt_w = max(0.0, dhvt_w)
                ensemble = (
                    wrn_w * wrn_probs
                    + pyr_w * pyramidnet_probs
                    + dhvt_w * dhvt_probs
                )
                acc1, acc5, sc = compute_metrics(ensemble, targets, fine_to_coarse_t)
                metric_sum = acc1 + sc
                avg = metric_sum / 2
                row = {
                    'wrn_w': wrn_w,
                    'pyr_w': pyr_w,
                    'dhvt_w': dhvt_w,
                    'beta': beta,
                    'acc1': acc1,
                    'acc5': acc5,
                    'sc': sc,
                    'sum': metric_sum,
                    'avg': avg,
                }
                rows.append(row)
                print(
                    f"{wrn_w:>6.3f}  {pyr_w:>6.3f}  {dhvt_w:>6.3f}  {beta:>5.2f}  "
                    f"{acc1:>7.2f}  {acc5:>7.2f}  {sc:>7.2f}  "
                    f"{metric_sum:>7.2f}  {avg:>7.2f}"
                )

    rows.sort(key=lambda item: (item['sum'], item['acc1']), reverse=True)
    top_k = max(1, min(args.top_k, len(rows)))

    print("\n" + "=" * 70)
    print(f"★ SUM/AVG 상위 {top_k}")
    print("=" * 70)
    print(
        f"{'rank':>4}  {'WRN_w':>6}  {'Pyr_w':>6}  {'DHVT_w':>6}  {'beta':>5}  "
        f"{'Acc@1':>7}  {'Acc@5':>7}  {'SC':>7}  {'SUM':>7}  {'AVG':>7}"
    )
    print("-" * 87)
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"{rank:>4}  {row['wrn_w']:>6.3f}  {row['pyr_w']:>6.3f}  "
            f"{row['dhvt_w']:>6.3f}  {row['beta']:>5.2f}  "
            f"{row['acc1']:>7.2f}  {row['acc5']:>7.2f}  {row['sc']:>7.2f}  "
            f"{row['sum']:>7.2f}  {row['avg']:>7.2f}"
        )

    best = rows[0]
    print("\nBest command:")
    print(
        "uv run python test/inference.py "
        f"--wrn-checkpoint {args.wrn_checkpoint} "
        f"--dhvt-checkpoint {args.dhvt_checkpoint} "
        f"--pyramidnet-checkpoint {args.pyramidnet_checkpoint} "
        f"--data-path {args.data_path} "
        f"--wrn-weight {best['wrn_w']:.3f} "
        f"--pyramidnet-weight {best['pyr_w']:.3f} "
        f"--fusion-beta {best['beta']:.3f} "
        f"--batch-size {args.batch_size}"
    )


if __name__ == '__main__':
    main()
