#!/bin/bash
# ============================================================
# 下载原始数据集图片 + 重建 assets/<hash>.png（供训练 json 引用）
#
# 训练 json 的图引用形式: assets/{vrsbench,levir_cc}/<sha256前48>.png
# 图片原始来自 HuggingFace 数据集（VRSBench / LEVIR-CC），其文件名形如
# Pxxxx/xx.png 或 train_000001.png（双时相），此处通过内容哈希重命名还原。
#
# 前提：
#   - 已具备原始图片目录（VRSBench images/、LEVIR-CC images/），通常位于
#     $DATA_RAW（共享区或手动下载的 HF 数据集根目录）。
#   - 本脚本负责：把原始图按内容哈希软链为 data/assets/<sub>/<hash>.png
#
# 用法:
#   bash scripts/fetch_raw_images.sh --raw /path/to/VRSBench/images --sub vrsbench
#   bash scripts/fetch_raw_images.sh --raw /path/to/LEVIR-CC/images --sub levir_cc --jobs 16
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_OUT="${ASSETS_OUT:-$REPO_ROOT/data/assets}"

RAW_MODE=""
declare -a RAW_DIRS=() SUBS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw) RAW_DIRS+=("$2"); shift 2 ;;
    --sub) SUBS+=("$2"); shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --out) ASSETS_OUT="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [ "${#RAW_DIRS[@]}" -eq 0 ]; then
  echo "用法: bash scripts/fetch_raw_images.sh --raw <dir> --sub <vrsbench|levir_cc> [--raw .. --sub ..]"
  exit 1
fi

echo "== 重建 assets（内容哈希重命名）=="
echo "  out: $ASSETS_OUT  jobs: ${JOBS:-8}"
mkdir -p "$ASSETS_OUT"

PYARGS=()
for i in "${!RAW_DIRS[@]}"; do
  PYARGS+=(--raw "${RAW_DIRS[$i]}" --sub "${SUBS[$i]}")
done
python "$REPO_ROOT/scripts/build_assets_from_raw.py" \
  "${PYARGS[@]}" --out "$ASSETS_OUT" --jobs "${JOBS:-8}"

echo "== 完成。assets 根: $ASSETS_OUT =="
echo "（训练脚本 --image_folder 应指向包含 assets/vrsbench 的根目录）"