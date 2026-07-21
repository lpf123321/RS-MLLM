#!/bin/bash
set -e
# ================================================================
# 修复 jsonl 中的绝对路径 → 新服务器路径
#
# jsonl 文件中硬编码了 /users/u2024311136/shared/shared_datasets/
# 此脚本将其替换为 $DATA_ROOT (默认: $REPO_ROOT/datasets)
# ================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OLD_PATH="/users/u2024311136/shared/shared_datasets"
NEW_PATH="${DATA_ROOT:-$REPO_ROOT/datasets}"

echo "============================================"
echo "  Fixing data paths in jsonl files"
echo "============================================"
echo "  Old: $OLD_PATH"
echo "  New: $NEW_PATH"
echo "============================================"

if [ ! -d "$NEW_PATH" ]; then
    echo "  WARN: 目标目录不存在: $NEW_PATH"
    echo "  请先上传数据集到此目录，结构如:"
    echo "    datasets/VRSBench/"
    echo "    datasets/MME-RealWorld-RS/"
    echo "    datasets/XLRS-Bench-lite/"
    echo "    datasets/LEVIR-CC/"
    echo ""
fi

fix_file() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "  SKIP (not found): $f"
        return
    fi
    local tmp="${f}.tmp"
    sed "s|$OLD_PATH|$NEW_PATH|g" "$f" > "$tmp"
    if diff -q "$f" "$tmp" > /dev/null 2>&1; then
        echo "  UNCHANGED: $f"
        rm "$tmp"
    else
        mv "$tmp" "$f"
        local count=$(grep -c "$NEW_PATH" "$f" || true)
        echo "  FIXED ($count refs): $f"
    fi
}

# 修复所有评估用 jsonl
fix_file "$NEW_PATH/VRSBench/vrsbench_eval.jsonl"
fix_file "$NEW_PATH/MME-RealWorld-RS/mme_rs.jsonl"
fix_file "$NEW_PATH/XLRS-Bench-lite/xlrs.jsonl"
fix_file "$NEW_PATH/LEVIR-CC/levircc_test.jsonl"

# 修复训练用 jsonl (VRSBench)
fix_file "$NEW_PATH/VRSBench/vrsbench_train.jsonl"

# 修复 LEVIR-CC 训练/验证数据
fix_file "$NEW_PATH/LEVIR-CC/levircc_train.jsonl"
fix_file "$NEW_PATH/LEVIR-CC/levircc_val.jsonl"

echo ""
echo "  Done. 源文件保留在 .bak 中。"
