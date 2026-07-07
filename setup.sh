#!/bin/bash
set -e

ENV_NAME="rs_mllm"

# Auto-detect conda
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    for p in /opt/app/anaconda3 ~/miniconda3 ~/anaconda3; do
        [ -f "$p/etc/profile.d/conda.sh" ] && CONDA_BASE="$p" && break
    done
fi
if [ -z "$CONDA_BASE" ]; then
    echo "Conda not found. Install Miniconda first: https://docs.anaconda.com/miniconda/"
    exit 1
fi

. "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.10
fi

conda activate $ENV_NAME

# ============================================================
# 0. Install newer GCC via conda (system GCC 8.5 is too old for PyTorch 2.8)
# ============================================================
echo "Installing GCC 12 via conda-forge..."
conda install -y -c conda-forge gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64

# Set compiler paths for flash-attn build
GCC_BIN="$CONDA_PREFIX/bin"
export CC="$GCC_BIN/x86_64-conda-linux-gnu-gcc"
export CXX="$GCC_BIN/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
echo "Using CC=$CC"
echo "Using CXX=$CXX"

# ============================================================
# 1. PyTorch (CUDA 12.8) — unified version for both projects
# ============================================================
echo "Installing PyTorch with CUDA 12.8..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ============================================================
# 2. Core RS-MLLM dependencies
# ============================================================
echo "Installing core dependencies..."
pip install transformers>=5.12.0 accelerate>=1.14.0 qwen-vl-utils>=0.0.14 \
    decord av "pillow>=10.0.0" einops huggingface_hub requests

# ============================================================
# 3. Qwen-VL-Series-Finetune dependencies
# ============================================================
echo "Installing Qwen-VL finetune dependencies..."

# Training frameworks
pip install deepspeed==0.17.5 peft==0.15.2 bitsandbytes==0.49.2 \
    liger_kernel==0.8.0 trl==0.25.0

# Data processing
pip install datasets opencv-python ujson

# Experiment tracking
pip install wandb

# flash-attn (requires ninja to build)
pip install ninja
#pip install flash-attn --no-build-isolation

# Additional utilities used by Qwen-VL finetune
pip install matplotlib tensorboardX

echo ""
echo "============================================================"
echo "All done! Activate: conda activate $ENV_NAME"
echo "============================================================"
echo ""
echo "Installed packages support:"
echo "  - RS-MLLM (VQA/CAP/REF inference + training)"
echo "  - Qwen-VL-Series-Finetune (SFT/DPO/GRPO/CLS)"
echo "  - Model: Qwen3-VL"
echo "  - CUDA: 12.8"
echo "  - PyTorch: 2.8.0"
