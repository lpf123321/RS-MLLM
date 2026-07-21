#!/bin/bash
set -e
# ================================================================
# RS-MLLM 一键部署 —— GPU 自适应版
# 支持: A100/3090/4090/A6000/T4/V100/... (AutoDL / 恒源云 / 任意)
#
# 用法:
#   bash deploy/deploy.sh                  # 仅推理+评估
#   bash deploy/deploy.sh --finetune       # 含微调依赖
# ================================================================
ENV_NAME="${1:-rs_mllm}"
PYTHON_VER="3.10"
FINETUNE=false
[[ "$1" == "--finetune" ]] && FINETUNE=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m' GRN='\033[0;32m' YLW='\033[1;33m' NC='\033[0m'
ok()   { echo -e "${GRN}[OK]${NC} $*"; }
warn() { echo -e "${YLW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }

echo "============================================"
echo "  RS-MLLM 部署 (GPU 自适应)"
echo "============================================"

# ================================================================
# Step 0: 检测 GPU 和 CUDA
# ================================================================
echo ""
echo "--- 检测 GPU ---"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NONE')"
CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' || echo '')"
DRIVER_VER="$(nvidia-smi 2>/dev/null | grep -oP 'Driver Version: \K[0-9.]+' || echo '')"

if [ "$GPU_NAME" = "NONE" ]; then
    err "未检测到 NVIDIA GPU。将安装 CPU 版 PyTorch。"
    TORCH_INDEX="cpu"
    USE_BF16=false
    USE_COMPILE=false
else
    ok "GPU: $GPU_NAME"
    ok "Driver: $DRIVER_VER"
    ok "CUDA: $CUDA_VER"

    # 判断 CUDA 主版本 → PyTorch index
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
    if   [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then CUDA_TAG=cu126
    elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then CUDA_TAG=cu124
    elif [ "$CUDA_MAJOR" -ge 12 ]; then CUDA_TAG=cu121
    else CUDA_TAG=cu118
    fi
    TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
    ok "PyTorch CUDA tag: $CUDA_TAG"

    # 判断 bf16 支持 (Ampere SM80+ 才支持 bf16)
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "0")
    SM_MAJOR=$(echo "$SM" | cut -d. -f1)
    if [ "$SM_MAJOR" -ge 8 ]; then
        USE_BF16=true
        ok "Compute Capability $SM → bf16 ✓, torch.compile ✓"
    else
        USE_BF16=false
        warn "Compute Capability $SM (<8.0) → bf16 ✗, 将使用 fp16"
    fi

    # flash-attn 需要 SM80+
    if [ "$SM_MAJOR" -ge 8 ]; then
        USE_FLASH_ATTN=true
        ok "flash-attn: 支持 ($SM)"
    else
        USE_FLASH_ATTN=false
        warn "flash-attn: 不支持 (需要 SM80+)"
    fi
fi

# ================================================================
# Step 1: Conda
# ================================================================
echo ""
echo "--- 配置 Conda ---"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    for p in /opt/conda ~/miniconda3 ~/anaconda3 /root/miniconda3; do
        [ -f "$p/etc/profile.d/conda.sh" ] && CONDA_BASE="$p" && break
    done
fi
if [ -z "$CONDA_BASE" ]; then
    echo "  安装 Miniconda..."
    curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p ~/miniconda3 && rm /tmp/mc.sh
    CONDA_BASE=~/miniconda3
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
ok "Conda: $CONDA_BASE"

if ! conda info --envs | grep -q "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=$PYTHON_VER
fi
conda activate "$ENV_NAME"
ok "环境: $ENV_NAME (python=$PYTHON_VER)"

# ================================================================
# Step 2: PyTorch
# ================================================================
echo ""
echo "--- 安装 PyTorch ---"
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    ok "PyTorch 已安装"
else
    if [ "$TORCH_INDEX" = "cpu" ]; then
        pip install torch torchvision torchaudio
    else
        pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url "$TORCH_INDEX"
    fi
    ok "PyTorch 安装完成"
fi

# 验证 CUDA 可用
python -c "
import torch
t=(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print(f'  CUDA available: {t[0]}, Device: {t[1]}')
print(f'  bf16 support:   {torch.cuda.is_bf16_supported()}')
"

# ================================================================
# Step 3: 核心依赖
# ================================================================
echo ""
echo "--- 核心依赖 ---"
pip install -q \
    "transformers>=5.12.0" \
    "accelerate>=1.14.0" \
    "qwen-vl-utils>=0.0.14" \
    "pillow>=10.0.0" \
    einops huggingface_hub requests tqdm
ok "transformers / qwen-vl-utils / accelerate"

# ================================================================
# Step 4: 评估依赖
# ================================================================
echo ""
echo "--- 评估依赖 ---"
pip install -q nltk rouge_score
python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true
ok "nltk / rouge_score"

# ================================================================
# Step 5: 微调依赖 (可选)
# ================================================================
if $FINETUNE; then
    echo ""
    echo "--- 微调依赖 ---"
    pip install -q deepspeed peft bitsandbytes datasets ujson

    # flash-attn (仅 Ampere+, 安装可能失败)
    if $USE_FLASH_ATTN; then
        pip install -q ninja 2>/dev/null || true
        pip install flash-attn --no-build-isolation 2>/dev/null && \
            ok "flash-attn" || \
            warn "flash-attn 安装失败 (不影响推理)"
    else
        warn "跳过 flash-attn (GPU 不支持)"
    fi
    ok "deepspeed / peft / bitsandbytes"
fi

# ================================================================
# Step 6: 生成 GPU 自适应配置
# ================================================================
echo ""
echo "--- 写入 GPU 配置 ---"
cat > "$REPO_ROOT/.gpu_config" << EOF
# 自动检测的 GPU 配置 — 评估脚本会读取此文件
GPU_NAME="$GPU_NAME"
CUDA_VERSION="$CUDA_VER"
USE_BF16=$USE_BF16
USE_FLASH_ATTN=$USE_FLASH_ATTN
USE_COMPILE=$USE_BF16
EOF
ok "配置写入 .gpu_config"

# ================================================================
# Step 7: 验证
# ================================================================
echo ""
echo "--- 验证 ---"
python -c "
import torch, transformers, accelerate
from qwen_vl_utils import process_vision_info
print(f'  torch={torch.__version__}, transformers={transformers.__version__}')
print(f'  accelerate={accelerate.__version__}')
print('  All imports OK')
" 2>/dev/null && ok "核心库导入通过" || err "导入失败"

# ================================================================
# 完成
# ================================================================
echo ""
echo "============================================"
echo -e "  ${GRN}部署完成!${NC}"
echo "============================================"
echo "  GPU:      $GPU_NAME"
echo "  bf16:     $USE_BF16"
echo "  flash-attn: $USE_FLASH_ATTN"
echo ""
echo "  下一步:"
echo "  1. export DATA_ROOT=$REPO_ROOT/datasets"
echo "  2. bash deploy/fix_data_paths.sh"
echo "  3. bash evaluation/run_eval_qwen35vl.slurm \\ "
echo "       --model_path models/Qwen3.5-4B --max_samples 100"
echo "============================================"
