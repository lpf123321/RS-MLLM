#!/bin/bash
# ============================================================
# 训练环境公共初始化（被 scripts/training/*.slurm source）
#
# 消除硬编码：conda 位置、数据集图片根目录均支持环境变量覆盖，
# 未设置时自动探测（本集群已知路径 → 通用 fallback）。
#
# 可用环境变量:
#   CONDA_HOME       conda 安装根（默认自动探测 /opt/app/anaconda3 或 ~/miniconda3）
#   CONDA_ENV        训练环境名（默认 rs_mllm）
#   TRAIN_DATA_ROOT  训练数据图片根目录（应含 assets/vrsbench、assets/levir_cc）
#   LOCAL_ASSETS     （可选）本地完整资产份；为空则回退 TRAIN_DATA_ROOT
# ============================================================

# --- CONDA 探测 ---
CONDA_HOME="${CONDA_HOME:-}"
if [ -z "$CONDA_HOME" ]; then
  for cand in /opt/app/anaconda3 "$HOME/miniconda3" "$HOME/anaconda3"; do
    [ -f "$cand/etc/profile.d/conda.sh" ] && CONDA_HOME="$cand" && break
  done
fi
CONDA_ENV="${CONDA_ENV:-rs_mllm}"

if [ -z "$CONDA_HOME" ]; then
  echo "[env] 未找到 conda。请设置 CONDA_HOME 指向含 etc/profile.d/conda.sh 的目录。" >&2
  exit 1
fi

activate_conda() {
  . "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  fi
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# --- 数据图片根目录探测 ---
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-}"
if [ -z "$TRAIN_DATA_ROOT" ]; then
  if [ -d "/users/u2024311009/team_share/rs-mllm-ready-20260725/data" ]; then
    TRAIN_DATA_ROOT="/users/u2024311009/team_share/rs-mllm-ready-20260725/data"
  elif [ -n "$LOCAL_ASSETS" ] && [ -d "$LOCAL_ASSETS" ]; then
    TRAIN_DATA_ROOT="$LOCAL_ASSETS"
  elif [ -d "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data/assets" ]; then
    TRAIN_DATA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/data"
  fi
fi

pick_data_root() {
  # 优先本地完整资产，否则回退共享/配置根
  if [ -n "$LOCAL_ASSETS" ] && [ -d "$LOCAL_ASSETS/assets/vrsbench" ] && \
     [ "$(ls -1 "$LOCAL_ASSETS/assets/vrsbench/"*.png 2>/dev/null | wc -l)" -gt 29000 ]; then
    echo "$LOCAL_ASSETS"
  else
    echo "$TRAIN_DATA_ROOT"
  fi
}

resolve_model() {
  # 解析模型引用为本地目录: 别名 -> get_model(ModelScope 懒加载); 本地/merged 路径原样返回。
  # 用法: MODEL_PATH="$(resolve_model base)"  或  MODEL_PATH="$(resolve_model "$MODEL_PATH")"
  local m="$1"
  # 本地路径(local dir 或 merged 完整模型目录)原样返回
  if [ -d "$m" ] || [ -f "$m" ]; then
    echo "$m"; return 0
  fi
  # 否则走 rsmllm 懒加载(基于 REPO_ROOT)
  local prev_py="$PYTHONPATH"
  export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
  local out
  out=$(python -c "from rsmllm.models import get_model; print(get_model('$m'))")
  export PYTHONPATH="$prev_py"
  echo "$out"
}

export CONDA_HOME CONDA_ENV TRAIN_DATA_ROOT