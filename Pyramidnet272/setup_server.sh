#!/bin/bash
# =============================================================================
# setup_server.sh - PyramidNet-272 server environment setup
# Usage: bash setup_server.sh
# =============================================================================

set -e

echo "======================================================"
echo "  PyramidNet-272 CIFAR-100 - server setup"
echo "======================================================"

echo ""
echo "[1/5] Creating folders..."
mkdir -p logs checkpoints data
echo "  logs/ checkpoints/ data/ ready"

echo ""
echo "[2/5] Creating and activating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  venv/ created"
else
    echo "  venv/ already exists"
fi

# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip -q
echo "  Python: $(python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "  pip   : $(python -m pip --version)"

echo ""
echo "[3/5] Detecting CUDA..."

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | cut -d. -f1,2)
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    echo "  CUDA detected by nvcc: $CUDA_VERSION"

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_TAG="CUDA 12.x (cu121)"
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_TAG="CUDA 11.8 (cu118)"
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu117"
        TORCH_TAG="CUDA 11.x (cu117)"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_TAG="CPU only"
    fi
else
    echo "  nvcc not found; trying nvidia-smi..."
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//')
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        echo "  CUDA detected by nvidia-smi: $CUDA_VERSION"

        if [ "$CUDA_MAJOR" -ge 12 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_TAG="CUDA 12.x (cu121)"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            TORCH_TAG="CUDA 11.8 (cu118)"
        fi
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_TAG="CPU only"
    fi
fi

echo "  PyTorch target: $TORCH_TAG"

echo ""
echo "[4/5] Installing PyTorch into venv..."
PYTHON_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
PYTHON_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
echo "  Python: $PYTHON_MAJOR.$PYTHON_MINOR"

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -le 8 ]; then
    python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url "$TORCH_INDEX" -q
else
    python -m pip install torch torchvision --index-url "$TORCH_INDEX" -q
fi
echo "  PyTorch installed"

echo ""
echo "[5/5] Installing remaining packages into venv..."
python -m pip install -r requirements_server.txt -q
echo "  Packages installed"

echo ""
echo "======================================================"
echo "  Verification"
echo "======================================================"
python - << 'PYEOF'
import torch
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA usable : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
import tqdm, numpy, matplotlib
print(f"  tqdm        : {tqdm.__version__}")
print(f"  numpy       : {numpy.__version__}")
print(f"  matplotlib  : {matplotlib.__version__}")
PYEOF

echo ""
echo "======================================================"
echo "  Setup complete. Example commands:"
echo ""
echo "  . venv/bin/activate"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  python train_server.py --seed 42 --epochs 1 --batch_size 256 --lr 0.2 --eval_interval 1 --val_batch_mult 1"
echo "  nohup python train_server.py --seed 42 --epochs 1800 --batch_size 256 --lr 0.2 --val_batch_mult 1 --fast_cudnn --channels_last > logs/seed42.log 2>&1 &"
echo ""
echo "  tail -f logs/seed42.log"
echo "  python evaluate.py --ckpt checkpoints/best_seed42.pth"
echo "  python evaluate.py --ckpt checkpoints/swa_final_seed42.pth"
echo "======================================================"
