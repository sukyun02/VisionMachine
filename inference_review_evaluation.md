# WRN + DHVT Inference Review Evaluation

## 검토 범위

아래 파일을 기준으로 리뷰 내용을 대조했다.

- `test/inference.py`
- `test/datasets.py`
- `WRN/train.py`
- `DHVT/datasets.py`
- `DHVT/main.py`

결론부터 말하면, 원 리뷰의 핵심 진단인 WRN normalization mismatch와 WRN TTA padding mismatch는 맞다. 이 둘은 먼저 고치는 것이 좋다. 다만 몇몇 표현은 더 정확히 바꿔야 한다. 특히 `datasets.py`는 이 저장소에 `test/datasets.py`와 `DHVT/datasets.py`가 모두 있으므로, `test/inference.py`가 실제로 import하는 `test/datasets.py`를 수정 대상으로 명시해야 한다. 또한 center-crop 누락 지적은 CIFAR 32x32 + padding=4 구조에서는 원본 view와 중복되므로, 실제 누락은 flipped corner crops라고 보는 편이 정확하다.

## 항목별 평가

| 번호 | 판정 | 평가 |
| --- | --- | --- |
| 1. WRN normalization mismatch | 맞음, 최우선 | `test/inference.py`는 WRN과 DHVT 모두 ImageNet mean/std를 사용한다. WRN은 `WRN/train.py`에서 CIFAR-100 mean/std로 학습/검증하므로 inference normalization이 틀려 있다. |
| 2. WRN TTA padding mode mismatch | 맞음, 최우선 | WRN train crop은 `padding_mode='reflect'`인데 inference corner crop은 `transforms.functional.pad(img, padding)` 기본값인 constant zero padding이다. WRN용 TTA는 reflect로 분리해야 한다. |
| 3. ensemble weight 미튜닝 | 맞음, 단 보완 필요 | `--wrn-weight=0.6`은 경험값으로 보이며 모델별 정확도/캘리브레이션 차이를 반영하지 않는다. 다만 weight sweep은 최종 test set이 아니라 validation split에서 해야 한다. |
| 4. calibration 차이 | 타당한 가설 | label smoothing, mixup, architecture 차이 때문에 softmax scale이 다를 수 있다. temperature scaling 또는 geometric mean fusion은 검토 가치가 있다. 단, temperature도 validation set에서 fit해야 한다. |
| 5. TTA coverage sparse | 부분 수정 | corner crop의 horizontal flip 조합이 빠진 것은 맞다. 그러나 padded 40x40에서 center crop은 원본 32x32와 같으므로 이미 `original`/`hflip`으로 커버된다. |
| 6. dataset 재생성 낭비 | 맞음 | TTA view마다 `datasets.CIFAR100(..., download=True)`를 새로 만든다. 실제 재다운로드는 보통 발생하지 않지만 dataset 객체 생성과 worker 준비가 반복된다. |
| 7. `fine_logits.to(device)` / `coarse_logits.to(device)` 불필요 | 맞음 | `collect_tta_logits_with_coarse` 내부에서 GPU tensor로 쌓고 반환하므로 다시 `.to(device)` 할 필요가 없다. |
| 8. checkpoint format 분기 | 부분 수정 | 현재 코드는 `isinstance(ckpt[key], dict)`로 WRN의 문자열 `model` 키를 피하므로 즉시 버그는 아니다. 그래도 checkpoint resolver를 명시 함수로 빼는 것이 더 안전하다. |
| 9. `collect_tta_probs` 이름/주석 | 대체로 맞음 | 반환값은 probability이므로 함수명 자체는 틀리지 않다. 다만 내부는 TTA probability average가 아니라 TTA logit average 후 softmax다. 파일 상단 docstring의 “probability averaging across models & TTA variants” 표현은 수정하는 것이 좋다. |

## 우선 적용할 수정

### 1. `test/datasets.py`에 CIFAR-100 normalization 추가

`test/inference.py`가 import하는 것은 실행 위치 기준의 `test/datasets.py`다. 따라서 원 리뷰의 `datasets.py` 수정 제안은 다음처럼 더 구체화하는 것이 좋다.

```python
# test/datasets.py
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
```

기존 `build_cifar100_test_transform`도 확장 가능하다.

```python
def build_cifar100_test_transform(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
```

### 2. model-specific transform으로 분리

WRN과 DHVT는 학습 normalization과 padding mode가 다르므로 TTA transform을 분리해야 한다.

```python
def get_tta_transforms(
    input_size=32,
    padding=4,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    padding_mode='constant',
    include_flipped_corners=True,
):
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
                return transforms.functional.crop(
                    padded, t, l, input_size, input_size)
            return crop_fn

        base_crop = transforms.Lambda(make_crop_fn(top, left))
        tta_list.append((name, transforms.Compose([base_crop, normalize])))

        if include_flipped_corners:
            tta_list.append((f"{name}_hflip", transforms.Compose([
                base_crop,
                transforms.RandomHorizontalFlip(p=1.0),
                normalize,
            ])))

    return tta_list
```

호출은 다음처럼 분리하는 것이 맞다.

```python
wrn_tta = get_tta_transforms(
    args.input_size,
    mean=CIFAR100_MEAN,
    std=CIFAR100_STD,
    padding_mode='reflect',
)

dhvt_tta = get_tta_transforms(
    args.input_size,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    padding_mode='constant',
)
```

### 3. no-TTA dataset도 모델별로 따로 생성

현재 `test_ds`는 ImageNet normalization으로 한 번 생성되어 WRN과 DHVT가 공유한다. WRN no-TTA도 잘못 평가되므로 별도 dataset이 필요하다.

```python
wrn_test_ds = datasets.CIFAR100(
    args.data_path,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ]),
    download=True,
)

dhvt_test_ds = datasets.CIFAR100(
    args.data_path,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]),
    download=True,
)
```

targets는 transform에 의존하지 않으므로 DataLoader를 돌릴 필요 없이 dataset metadata에서 직접 가져와도 된다.

```python
target_ds = datasets.CIFAR100(args.data_path, train=False, download=True)
targets = torch.tensor(target_ds.targets, dtype=torch.long, device=device)
```

## Ensemble 개선안

normalization과 padding을 먼저 고친 뒤에도 ensemble이 WRN 단독보다 낮으면 그때 weight와 calibration을 만지는 순서가 맞다.

1. WRN no-TTA/TTA와 DHVT no-TTA/TTA의 Acc@1, Acc@5, SC Density를 각각 다시 출력한다.
2. validation split이 있으면 `wrn_weight`를 grid sweep한다. 예: `0.5, 0.55, ..., 0.95`.
3. probability arithmetic mean 외에 geometric mean을 비교한다.

```python
eps = 1e-12
ensemble_log_probs = (
    w * torch.log(model_probs['wrn'].clamp_min(eps))
    + (1 - w) * torch.log(model_probs['dhvt'].clamp_min(eps))
)
ensemble_probs = F.softmax(ensemble_log_probs, dim=1)
```

4. temperature scaling은 각 모델별 temperature를 validation set에서 fit한 뒤 적용한다. 최종 test set으로 temperature나 weight를 고르면 leaderboard/test leakage가 된다.
5. rank-based fusion은 calibration 차이를 피하는 빠른 baseline으로 쓸 수 있지만, 확률 해석은 사라진다.

## 추가로 확인할 만한 점

### DHVT EMA checkpoint 사용 여부

`DHVT/main.py`는 checkpoint에 `model`과 `model_ema`를 모두 저장한다. 현재 `test/inference.py`의 `load_model`은 `model`을 먼저 읽고 `model_ema`는 고려하지 않는다. DHVT 학습에서 EMA 성능을 기대했다면 `--use-model-ema` 옵션을 추가해 비교하는 것이 좋다.

권장 resolver는 한 줄 expression보다 명시적인 함수가 안전하다.

```python
def resolve_state_dict(ckpt, prefer_ema=False):
    if not isinstance(ckpt, dict):
        return ckpt

    if prefer_ema and isinstance(ckpt.get('model_ema'), dict):
        return ckpt['model_ema']

    for key in ('model_state_dict', 'state_dict', 'model'):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value

    return ckpt
```

### CPU 실행 시 autocast

`collect_logits`와 DHVT TTA path는 `torch.amp.autocast('cuda')`를 고정 사용한다. 기본 실행은 CUDA라 큰 문제는 아니지만, `--device cpu`를 지원하려면 device type을 분기하는 편이 낫다.

## 수정된 우선순위

| 순위 | 이슈 | 영향 | 판정 |
| --- | --- | --- | --- |
| 1 | WRN normalization mismatch | WRN 단독과 ensemble 모두 직접 손상 | 반드시 수정 |
| 2 | WRN TTA padding mode mismatch | WRN TTA view 품질 저하 | 반드시 수정 |
| 3 | WRN/DHVT no-TTA dataset 공유 | WRN no-TTA 평가가 ImageNet normalization으로 왜곡 | 반드시 수정 |
| 4 | ensemble weight 미튜닝 | 약한 모델 또는 miscalibrated 모델이 ensemble을 낮출 수 있음 | normalization 수정 후 검토 |
| 5 | 모델 간 calibration 차이 | probability average가 불안정할 수 있음 | validation 기반으로 검토 |
| 6 | DHVT EMA 미사용 가능성 | DHVT 단독 성능 저하 가능성 | checkpoint 의도 확인 후 비교 |
| 7 | flipped corner crop 누락 | TTA coverage 부족 | 선택 수정 |
| 8 | dataset 반복 생성 | 속도 낭비 | 선택 수정 |
| 9 | checkpoint resolver/주석 정리 | 유지보수성 | 선택 수정 |

## 최종 판단

원 리뷰는 전체 방향이 좋고, 특히 1번과 2번은 실제 코드와 정확히 맞는 고우선순위 지적이다. 다만 문서화할 때는 다음 세 가지를 고쳐 쓰는 것이 좋다.

- `datasets.py`는 `test/datasets.py`를 명시한다.
- center-crop 누락이 아니라 flipped corner crop 누락이라고 정정한다.
- weight sweep과 temperature scaling은 final test set이 아니라 validation split에서 해야 한다고 명시한다.

추천 실행 순서는 다음과 같다.

1. WRN normalization과 WRN no-TTA dataset을 CIFAR-100 stats로 수정한다.
2. WRN TTA padding을 reflect로 바꾸고, DHVT TTA는 constant padding을 유지한다.
3. 같은 조건에서 WRN/DHVT 단독 성능을 다시 측정한다.
4. 그래도 ensemble이 WRN보다 낮으면 validation 기반으로 weight sweep, geometric mean, temperature scaling을 비교한다.
5. DHVT checkpoint의 `model_ema` 사용 여부를 별도 실험으로 확인한다.
