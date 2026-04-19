# WRN (WideResNet-28-10)

CIFAR-100 분류용 WideResNet-28-10 학습 코드입니다. MixUp/CutMix, Superclass Auxiliary Loss, Superclass-aware Label Smoothing을 적용합니다.

## Requirements

본 프로젝트는 **uv**로 패키지를 관리합니다.

```bash
# uv 설치 (macOS / Linux)
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# uv 설치 (Windows PowerShell)
$ powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

의존 패키지: `torch`, `torchvision`, `tqdm`, `numpy`, `matplotlib`, `wandb`

## 1. 학습

```bash
$ cd WRN
$ uv run python train.py 
```

이어서 학습:

```bash
$ uv run python train.py --resume --model wrn_28_10
```



## 주요 하이퍼파라미터

| 파라미터 | 값    |
|---------|------|
| epochs | 500  |
| batch_size | 512  |
| lr | 0.1  |
| weight_decay | 5e-4 |
| warmup_epochs | 25   |
| label_smooth | 0.05 |
| cutmix_prob | 0.5  |
| cutmix_alpha | 1.0  |
| mixup_alpha | 0.4  |
| lambda_super | 3.0  |
| intra_ratio | 0.95 |

## 성능

| 메트릭 | 값   |
|--------|-----|
| Top-1 Accuracy | 83~ |
| Superclass Accuracy | 90~ |
| Top-1 (TTA 적용) | 83~ |

## 파일 구조

```
WRN/
├── README.md           # 이 파일
├── train.py            # 학습 스크립트
├── test.py             # 평가 스크립트 (수정 금지)
└── models/
    ├── __init__.py
    └── wideresnet.py   # WRN-28-10 구현
```
