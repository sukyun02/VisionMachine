"""
CIFAR100 Supervised Learning Challenge - Training Pipeline
DAI3004: Learning Vision Intelligence
WideResNet-28-10 최적화
"""

import os, sys, argparse, time, random, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.append('/data')
from models.wideresnet import wrn_28_10, wrn_40_10



def get_args():
    parser = argparse.ArgumentParser(description='CIFAR100 Training')
    parser.add_argument('--model', type=str, default='wrn_28_10',
                        choices=['wrn_28_10', 'wrn_40_10'])

    parser.add_argument('--epochs',        type=int,   default=500)
    parser.add_argument('--batch_size',    type=int,   default=512)
    parser.add_argument('--lr',            type=float, default=0.1)
    parser.add_argument('--weight_decay',  type=float, default=5e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=25)
    parser.add_argument('--label_smooth',  type=float, default=0.05)
    parser.add_argument('--cutmix_alpha',  type=float, default=1.0)
    parser.add_argument('--mixup_alpha',   type=float, default=0.4)
    parser.add_argument('--cutmix_prob',   type=float, default=0.5)
    parser.add_argument('--lambda_super',  type=float, default=3.0,
                        help='superclass auxiliary loss 가중치 (logsumexp 집계 + CE)')
    parser.add_argument('--intra_ratio',   type=float, default=0.95,
                        help='label smoothing 중 같은 superclass sibling에 줄 비율')
    parser.add_argument('--data_dir',      type=str,   default='./data')
    parser.add_argument('--save_dir',      type=str,   default='./checkpoints')
    parser.add_argument('--plot_dir',      type=str,   default='./plots')
    parser.add_argument('--plot_every',    type=int,   default=50)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--resume',        action='store_true')
    parser.add_argument('--save_name',     type=str,   default=None, help='체크포인트 이름 (기본: best_{model}.pth)')
    parser.add_argument('--wandb',         action='store_true', help='W&B 로깅 활성화')
    parser.add_argument('--wandb_project', type=str,   default='cifar100-wrn')
    parser.add_argument('--wandb_name',    type=str,   default=None, help='run 이름 (기본: 자동)')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

from test import evaluate, get_cifar100_superclass_mapping  # 교수님 제공 평가 함수

SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_vegetables',
    'household_electrical', 'household_furniture', 'insects', 'large_carnivores',
    'large_outdoor_manmade', 'large_outdoor_natural', 'large_omnivores_herbivores',
    'medium_mammals', 'non_insect_invertebrates', 'people', 'reptiles',
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2',
]

def evaluate_detailed(model, loader, device):
    """superclass별 accuracy, top-5 accuracy 계산"""
    model.eval()
    class_to_super = get_cifar100_superclass_mapping()
    super_correct = [0] * 20
    super_total   = [0] * 20
    top5_correct  = 0
    total         = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # top-5
            _, pred5 = outputs.topk(5, dim=1)
            top5_correct += pred5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            # superclass별
            _, pred1 = outputs.topk(1, dim=1)
            pred1 = pred1.squeeze(1)
            for fine, sup in class_to_super.items():
                mask = labels == fine
                if mask.sum() == 0:
                    continue
                super_total[sup]   += mask.sum().item()
                super_correct[sup] += (pred1[mask] == labels[mask]).sum().item()

            total += labels.size(0)

    top5_acc = top5_correct / total
    per_super = {
        SUPERCLASS_NAMES[i]: (super_correct[i] / super_total[i] * 100 if super_total[i] > 0 else 0.0)
        for i in range(20)
    }
    return top5_acc, per_super

def get_dataloaders(args):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),  # CIFAR 특화 정책, RandAugment 대비 정확도 ↑
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.2)),   # ShakeDrop 중복 규제 방지 위해 완화
    ])
    val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_set = torchvision.datasets.CIFAR100(root=args.data_dir, train=True,  download=True, transform=train_tf)
    val_set   = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_w, cut_h = int(W * (1-lam)**0.5), int(H * (1-lam)**0.5)
    cx, cy = random.randint(0, W), random.randint(0, H)
    return max(cx-cut_w//2,0), max(cy-cut_h//2,0), min(cx+cut_w//2,W), min(cy+cut_h//2,H)

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1 - (x2-x1)*(y2-y1) / (x.size(2)*x.size(3))
    return mixed, y, y[idx], lam

def mixup(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)


def build_super_logits(logits, fine_to_super):
    """100-class logits → 20-superclass log-probabilities (logsumexp 집계)"""
    B = logits.size(0)
    # 각 superclass별 logsumexp로 정확한 log-probability 계산
    # -inf로 초기화 후 scatter_logsumexp 수행
    neg_inf = torch.full((B, 20), -float('inf'), device=logits.device, dtype=logits.dtype)
    expanded_map = fine_to_super.unsqueeze(0).expand(B, -1)  # (B, 100)
    # scatter 방식 대신 마스크 기반 logsumexp
    super_logits = torch.stack([
        torch.logsumexp(logits.masked_fill(fine_to_super.unsqueeze(0) != sc, -float('inf')), dim=1)
        for sc in range(20)
    ], dim=1)  # (B, 20)
    return super_logits


def build_smooth_matrix(class_to_super, super_to_classes, device, confidence=0.85, intra_ratio=0.8):
    """
    superclass-aware soft label matrix (100, 100)
    - 정답 클래스: confidence(0.85)
    - 같은 superclass sibling 4개: intra_ratio 비율로 나머지 분배 → 0.04 each
    - 다른 superclass 80개: 나머지 → 0.0005 each
    기존 uniform smoothing 대비 top-5가 correct superclass에 집중되도록 유도
    """
    n = 100
    mat = torch.zeros(n, n, device=device)
    eps = 1.0 - confidence
    for c in range(n):
        sc = class_to_super[c]
        siblings = [s for s in super_to_classes[sc] if s != c]   # 4개
        others   = [x for x in range(n) if class_to_super[x] != sc]  # 80개
        mat[c, c] = confidence
        for s in siblings:
            mat[c, s] = eps * intra_ratio / len(siblings)    # 0.04 each
        for x in others:
            mat[c, x] = eps * (1.0 - intra_ratio) / len(others)  # 0.0005 each
    return mat  # (100, 100)


def soft_cross_entropy(logits, soft_targets):
    """soft target cross entropy (superclass-aware smooth label용)"""
    log_probs = F.log_softmax(logits.float(), dim=1)
    return -(soft_targets.float() * log_probs).sum(dim=1).mean()


def mixed_soft_criterion(logits, y_a, y_b, lam, smooth_matrix):
    """CutMix/MixUp에 superclass-aware soft label 적용"""
    soft_a = smooth_matrix[y_a]                          # (B, 100)
    soft_b = smooth_matrix[y_b]                          # (B, 100)
    soft_mixed = lam * soft_a + (1.0 - lam) * soft_b    # (B, 100)
    return soft_cross_entropy(logits, soft_mixed)


def train_one_epoch(model, loader, optimizer, criterion, super_criterion, scaler, args, device, super_map, smooth_matrix):
    model.train()
    total_loss, total_correct, total = 0., 0, 0
    grad_norms = []
    pbar = tqdm(loader, desc='Train', leave=False, ncols=80)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        if random.random() < args.cutmix_prob:
            imgs, y_a, y_b, lam = cutmix(imgs, labels, args.cutmix_alpha)
        else:
            imgs, y_a, y_b, lam = mixup(imgs, labels, args.mixup_alpha)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out  = model(imgs)
            # superclass-aware soft label로 fine loss 계산
            fine_loss = mixed_soft_criterion(out, y_a, y_b, lam, smooth_matrix)

            # superclass auxiliary loss (logsumexp 집계 + CE)
            # = -log P(correct superclass) under 100-class softmax (동일)
            super_logits = build_super_logits(out, super_map)
            super_loss = mixed_criterion(super_criterion, super_logits,
                                         super_map[y_a], super_map[y_b], lam)
            loss = fine_loss + args.lambda_super * super_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(grad_norm.item())
        scaler.step(optimizer)
        scaler.update()
        # train acc: y_a(원본 labels) 기준
        with torch.no_grad():
            _, pred = out.detach().topk(1, dim=1)
            total_correct += pred.squeeze(1).eq(y_a).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        pbar.set_postfix(loss=f'{total_loss/total:.4f}')
    return total_loss / total, total_correct / total, float(np.mean(grad_norms))


def save_plot(history, best_acc, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Loss ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
    ax.plot(epochs, history['val_loss'],   label='Val Loss',   color='coral')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Train / Val Loss'); ax.legend(); ax.grid(alpha=0.3)

    # ── Fine-class Accuracy ───────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history['val_acc'], label='Fine Acc (100 cls)', color='green')
    ax.axhline(y=best_acc, color='red', linestyle='--', label=f'Best {best_acc:.2f}%')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fine-class Val Accuracy (Top-1)'); ax.legend(); ax.grid(alpha=0.3)

    # ── Superclass Accuracy ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, history['val_super_acc'], label='Super Acc (20 cls)', color='purple')
    best_super = max(history['val_super_acc'])
    ax.axhline(y=best_super, color='red', linestyle='--', label=f'Best {best_super:.2f}%')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Superclass Val Accuracy'); ax.legend(); ax.grid(alpha=0.3)

    # ── Train-Val Loss Gap (과적합 지표) ──────────────────────────────
    ax = axes[1, 1]
    gap = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
    ax.plot(epochs, gap, color='darkorange', label='Train Loss − Val Loss')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss Gap')
    ax.set_title('Loss Gap (양수 = 정상, 음수 = 과적합 징후)'); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(f'[{filename}]  Best Fine: {best_acc:.2f}%  |  Best Super: {best_super:.2f}%',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=150)
    plt.close()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # W&B 초기화
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print('[경고] wandb 패키지가 없습니다. pip install wandb 후 재실행하세요.')
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )

    print(f'Device: {device}')
    train_loader, val_loader = get_dataloaders(args)

    model_map = {'wrn_28_10': wrn_28_10,
                 'wrn_40_10': wrn_40_10,
                 }

    model = model_map[args.model](num_classes=100).to(device)
    print(f'Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}')

    # superclass auxiliary loss 용 fine→super 매핑 텐서
    _class_to_super = get_cifar100_superclass_mapping()
    super_map = torch.zeros(100, dtype=torch.long, device=device)
    for fine, sup in _class_to_super.items():
        super_map[fine] = sup

    # super_to_classes 역매핑 (20개 superclass → 소속 fine class 목록)
    super_to_classes = [[] for _ in range(20)]
    for fine, sup in _class_to_super.items():
        super_to_classes[sup].append(fine)

    # superclass-aware soft label matrix 빌드
    smooth_matrix = build_smooth_matrix(_class_to_super, super_to_classes, device,
                                         confidence=1.0 - args.label_smooth,
                                         intra_ratio=args.intra_ratio)
    print(f'Superclass auxiliary loss λ={args.lambda_super} | intra_ratio={args.intra_ratio}')

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    super_criterion = nn.CrossEntropyLoss()  # superclass loss는 label smoothing 없이
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch+1) / args.warmup_epochs
        t = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler     = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler  = torch.amp.GradScaler('cuda')
    history     = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_super_acc': []}
    best_top1_acc = 0.
    best_acc    = 0.
    best_super_acc = 0.
    start_epoch = 1

    ckpt_path = os.path.join(args.save_dir, 'WRN-28-10_fine0.6.pth')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        best_acc       = ckpt['best_acc']
        best_super_acc = ckpt.get('best_super_acc', 0.)
        best_top1_acc = best_acc
        history     = ckpt['history']
        # 구 체크포인트 호환: 없는 키 채우기
        for key in ['train_acc', 'val_super_acc']:
            if key not in history:
                history[key] = [0.0] * len(history['train_loss'])
        start_epoch = ckpt['epoch'] + 1
        print(f'체크포인트 로드! Epoch {start_epoch}부터 재시작 | Best: {best_acc*100:.2f}%')

    print(f"\n{'Ep':>4} | {'Train Loss':>10} | {'Val Loss':>8} | {'Top-1':>7} | {'Super':>7} | {'LR':>8} | Time")
    print("-" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, grad_norm = train_one_epoch(model, train_loader, optimizer, criterion, super_criterion, scaler, args, device, super_map, smooth_matrix)
        val_loss, val_acc, val_super_acc  = evaluate(model, val_loader, criterion, device)
        val_top5_acc, per_super_acc       = evaluate_detailed(model, val_loader, device)

        scheduler.step()

        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc * 100)
        history['val_super_acc'].append(val_super_acc * 100)

        print(f"{epoch:>4} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
              f"{val_acc*100:>6.2f}% | {val_super_acc*100:>6.2f}% | {lr_now:>8.6f} | {elapsed:.1f}s")

        if use_wandb:
            log_dict = {
                'epoch':            epoch,
                'train/loss':       train_loss,
                'train/acc':        train_acc * 100,
                'train/grad_norm':  grad_norm,
                'val/loss':         val_loss,
                'val/top1_acc':     val_acc * 100,
                'val/top5_acc':     val_top5_acc * 100,
                'val/super_acc':    val_super_acc * 100,
                'val/acc_gap':      (train_acc - val_acc) * 100,  # overfitting 지표
                'lr':               lr_now,
            }
            # superclass별 accuracy
            for name, acc in per_super_acc.items():
                log_dict[f'val/super/{name}'] = acc
            wandb.log(log_dict)

        best_acc = max(best_acc, val_acc)
        best_super_acc = max(best_super_acc, val_super_acc)

        if val_acc > best_top1_acc:
            best_top1_acc = val_acc
            best_name = args.save_name if args.save_name else f'best_{args.model}.pth'
            torch.save({
                'epoch': epoch, 'model': args.model,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_super_acc': val_super_acc,
                'best_top1_acc': best_top1_acc,
                'best_acc': best_acc,
                'best_super_acc': best_super_acc,
            }, os.path.join(args.save_dir, best_name))
            print(f"  ★ Best saved (Top-1: {val_acc*100:.2f}% | Super: {val_super_acc*100:.2f}%)")
            if use_wandb:
                wandb.run.summary['best_top1_acc']  = val_acc * 100
                wandb.run.summary['best_super_acc'] = val_super_acc * 100

        last_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_top1_acc': best_top1_acc,
            'best_acc': best_acc,
            'best_super_acc': best_super_acc,
            'history': history,
        }
        torch.save(last_ckpt, ckpt_path)

        if epoch % args.plot_every == 0:
            save_plot(history, best_acc*100, args.plot_dir, f'{args.model}_epoch{epoch}')
            print(f'  그래프 저장: {args.plot_dir}/{args.model}_epoch{epoch}.png')

    save_plot(history, best_acc*100, args.plot_dir, f'{args.model}_final')
    print(f'\n학습 완료! Best Val Acc (Fine): {best_acc*100:.2f}%')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
