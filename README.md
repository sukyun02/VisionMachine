# VisionMachine


## Requirements
- 본 프로젝트는 uv로 패키지를 관리합니다.
```bash
# uv 설치
$ curl -LsSf https://astral.sh/uv/install.sh | sh
# 경로 설정
$ export PATH="$HOME/.local/bin:$PATH"
# 의존성 설치
$ uv sync 
```

### 1. DHVT 학습
```bash
$ cd DHVT
$ uv run bash run_code_cifar.sh
```


### 2. WRN 학습
```bash
$ cd WRN
$ uv run python train.py 
```

### 3. PyramidNet 학습
```bash
$ cd Pyramidnet272
$ uv run train_server.py
```

### 4. 테스트
```bash
$ cd test
$ uv run inference.py
```
