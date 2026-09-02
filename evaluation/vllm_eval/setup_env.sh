#!/bin/bash
# ============================================================
# vLLM 评测环境一键还原 (uv 项目标准流程)
# 用法: bash evaluation/vllm_eval/setup_env.sh
# 产出: evaluation/vllm_eval/.venv  (uv sync --locked 精确还原)
# 已验证: 干净环境 uv sync 后 Qwen3.5 加载+生成成功(无手工 symlink)
#
# 关键约束: 评测 venv 必须使用 uv 托管 Python (自带 include/Python.h)。
#   vLLM/Triton 首次运行会即时编译 CUDA 辅助模块, 需要 Python 开发头文件;
#   系统/conda Python 没有 (集群未装 python3.11-devel) 时, 用它们创建的
#   .venv 会在编译阶段报:
#       fatal error: Python.h: No such file or directory
#   uv sync 默认会复用已存在的 .venv (其解释器可能来自系统 python),
#   因此本脚本显式处理: 头文件缺失的旧环境备份为 .venv-system-python,
#   再以 --python-preference only-managed --python 3.11 强制重建
#   (仅使用 uv 托管 Python; 等价于新版 uv 的 --managed-python)。
# ============================================================
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${HOME}/.local/bin/uv"
[ -x "${UV}" ] || UV="$(command -v uv || true)"
[ -x "${UV}" ] || { echo "uv 未安装 (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

cd "${EVAL_DIR}"

# ---- 1. 确保 uv 托管 Python 3.11 (自带 Python.h) ----
echo "==> 确保 uv 托管 Python 3.11 (自带 include/Python.h)"
"${UV}" python install 3.11
# 若下载失败(网络受限), 请自行 export UV_PYTHON_INSTALL_MIRROR 指向可用镜像后重跑本脚本

# ---- 2. 现有 .venv 解释器头文件自检 ----
# 通过 sysconfig include 查找 Python.h; uv 建的 venv 头文件齐全,
# 系统/conda python 建的 venv 通常缺失 (无 python3.11-devel)。
check_python_h() {
    "$1" - <<'PY' >/dev/null 2>&1
import pathlib, sysconfig
p = pathlib.Path(sysconfig.get_paths()["include"]) / "Python.h"
raise SystemExit(0 if p.is_file() else 1)
PY
}

if [ -x ".venv/bin/python" ] && ! check_python_h ".venv/bin/python"; then
    bak=".venv-system-python"
    [ ! -e "${bak}" ] || bak=".venv-system-python-$(date +%Y%m%d-%H%M%S)"
    echo "==> 现有 .venv 缺少 Python.h (系统 Python 无开发头), 备份为 ${bak} 并重建"
    mv .venv "${bak}"
fi

# ---- 3. 依赖还原 (仅 uv 托管 Python) ----
echo "==> uv sync --locked (vllm 0.26 url source + torch 2.11 cu129, uv 托管 Python 3.11)"
"${UV}" sync --locked --python-preference only-managed --python 3.11

# ---- 4. 最终校验 ----
echo "==> 校验 Python.h"
if check_python_h ".venv/bin/python"; then
    echo "    Python.h OK: $(.venv/bin/python -c 'import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()["include"])/"Python.h")')"
else
    echo "    [error] .venv 仍缺少 Python.h; 请确认 uv 托管 python 已安装: ${UV} python install 3.11" >&2
    exit 1
fi

echo ""
echo "==> 完成。运行评测前:"
echo "    ${EVAL_DIR}/.venv/bin/python ${EVAL_DIR}/vision_opd_vllm_eval.py --help"
echo "    (环境变量由评测器自动设置, 无需手动 export)"
