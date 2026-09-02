#!/bin/bash
# ============================================================
# 下载原始数据集图片 + 重建 assets/<hash>.png（供训练 json 引用）
#
# 训练 json 的图引用形式: assets/{vrsbench,levir_cc}/<sha256前48>.png
# 图片来源于 VRSBench / LEVIR-CC 数据集, 经内容哈希重命名还原到
# <仓库>/data/assets/<sub>/<hash>.png (env.sh 的 LOCAL_ASSETS 探测目标).
#
# 图片来源(二选一):
#   --local (默认)   从 datasets/VRSBench 与 datasets/LEVIR-CC 读取原图
#   --hf <repo>      从 HuggingFace 下载数据集图片(评委无共享区的兜底; 需可精确指图目录)
#
# 用法:
#   bash scripts/fetch_raw_images.sh              # datasets/ 下的 VRS+LEVIR
#   bash scripts/fetch_raw_images.sh --raw-root /path/to/datasets
#   bash scripts/fetch_raw_images.sh --hf <repo_id>
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ASSETS_OUT="${ASSETS_OUT:-$REPO_ROOT/data/assets}"
SHARED_DATA="${RS_MLLM_RAW_DATASETS_ROOT:-${SHARED_DATA:-$REPO_ROOT/datasets}}"
JOBS=8
MODE="local"
HF_REPO=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared|--local) MODE="local"; shift ;;
    --raw-root) SHARED_DATA="$2"; shift 2 ;;
    --hf)     MODE="hf"; HF_REPO="$2"; shift 2 ;;
    --jobs)   JOBS="$2"; shift 2 ;;
    --out)    ASSETS_OUT="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

echo "== 重建 assets（内容哈希重命名）=="
echo "  mode: $MODE   out: $ASSETS_OUT  jobs: $JOBS"
mkdir -p "$ASSETS_OUT"

TIFF="$REPO_ROOT/.raw_images"
case "$MODE" in
  hf)
    [ -n "$HF_REPO" ] || { echo "用法: fetch_raw_images.sh --hf <repo_id>"; exit 1; }
    echo "  HF repo: $HF_REPO"
    rm -rf "$TIFF"; mkdir -p "$TIFF"
    python - "$HF_REPO" "$TIFF" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                  token=os.environ.get("HF_TOKEN") or None)
print("HF 下载完成:", dst)
PY
    # 尝试自动定位包含图片的目录并推断 asset 子目录
    python - "$TIFF" <<'PY'
import sys, glob, os, json
root = sys.argv[1]
png = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
jpg = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
print(f"png={len(png)} jpg={len(jpg)}")
# 输出候选子目录给用户判断
from collections import Counter
dirs = Counter()
for p in png + jpg:
    rel = os.path.relpath(os.path.dirname(p), root)
    dirs[rel] += 1
for d, n in dirs.most_common(8):
    print(f"  {d or '(root)'}: {n} 图")
PY
    ;;
  local)
    VRS="$SHARED_DATA/VRSBench"
    LEV="$SHARED_DATA/LEVIR-CC"
    echo "  本地原数据: $VRS + $LEV"
    [ -d "$VRS" ] || { echo "缺少原数据目录: $VRS" >&2; exit 2; }
    [ -d "$LEV" ] || { echo "缺少原数据目录: $LEV" >&2; exit 2; }
    ;;
esac

# 组装 --raw/--sub（local 固定两对; hf 需用户确认子目录，这里以根递归）
declare -a RAW_DIRS=() SUBS=()
if [ "$MODE" = "local" ]; then
  RAW_DIRS=( "$SHARED_DATA/VRSBench" "$SHARED_DATA/LEVIR-CC" )
  SUBS=( "vrsbench" "levir_cc" )
else
  RAW_DIRS=( "$TIFF" )
  SUBS=( "vrsbench" )   # HF 单源, 默认 vrsbench; 若图是 levir 可在 build 后调整
fi

echo "  哈希配对: ${RAW_DIRS[*]} ⇢ ${SUBS[*]}"
PYARGS=()
# The shared VRSBench extraction contains one known zero-byte member. Recover
# it from the official ZIP and verify the full digest used by all affected
# stage1/GA2/A1 records.
VRS_RECOVER_SHA="26ab63466cc4f113e6b5e446792eef7ecd3e4a0f7e5a708f3b889ec30cacc7e0"
VRS_RECOVER_OUT="$ASSETS_OUT/vrsbench/${VRS_RECOVER_SHA}.png"
if [ "$MODE" = local ] && [ ! -s "$VRS_RECOVER_OUT" ] && [ -f "$SHARED_DATA/VRSBench/Images_train.zip" ]; then
  [ -L "$VRS_RECOVER_OUT" ] && unlink "$VRS_RECOVER_OUT"
  python "$REPO_ROOT/scripts/recover_asset_from_zip.py" \
    --zip "$SHARED_DATA/VRSBench/Images_train.zip" \
    --member Images_train/P7581_0003.png --sha256 "$VRS_RECOVER_SHA" \
    --output "$VRS_RECOVER_OUT"
fi
for i in "${!RAW_DIRS[@]}"; do
  PYARGS+=(--raw "${RAW_DIRS[$i]}" --sub "${SUBS[$i]}")
done
python "$REPO_ROOT/scripts/build_assets_from_raw.py" \
  "${PYARGS[@]}" --out "$ASSETS_OUT" --jobs "$JOBS"

echo "== 完成。assets 根: $ASSETS_OUT =="
echo "（训练脚本 LOCAL_ASSETS=$REPO_ROOT/data 时优先用本地还原资产）"