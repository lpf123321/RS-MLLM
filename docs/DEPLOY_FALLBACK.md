# 部署环境：回滚路径与版本管理

## 三层环境（按优先级）

| 层 | 机制 | 版本管理 | 何时使用 |
|---|---|---|---|
| **Docker 容器** | `Dockerfile` + 镜像内 `uv sync --locked` | `uv.lock`（121 包精确锁定）| 完全隔离，推荐 GPU 服务器 |
| **uv 虚拟环境** | `setup.sh` → `uv sync --locked` | `uv.lock` | 无容器时的轻量路径（已多机验证）|
| **pip 回滚** | `setup.sh` conda 兜底 → `pip install -r requirements.txt` | `requirements.txt`（仅顶层钉版）| 前两层都不可用时的最终兜底 |

## ⚠️ pip 回滚的版本漂移风险

`uv.lock` 锁定**全部 121 个包**（含每个间接依赖的精确版本与哈希）；`requirements.txt` 只钉**顶层包版本**（如 `transformers>=5.12.0`、`deepspeed==0.17.5`）。

**没有 lock 的后果**：
- 间接依赖（`numpy`、`protobuf`、`tokenizers` 等）按当天 PyPI 最新版本解析——**不同时间安装可能得到不同环境**；
- 上游小版本发布可能引入不兼容（如新 tokenizers 与旧 transformers 组合）；
- 复现实验室结果时，跨月的两次安装可能有**不可见的数值/行为差异**。

**因此**：pip 回滚仅作为**临时应急**，回滚成功后应尽快恢复 `uv sync --locked` 路径，或重新生成 `uv.lock` 并提交。

## 容器回滚步骤（容器内直接退回 pip）

若容器内 `uv sync --locked` 失败（网络/索引异常），可回滚为 pip 安装：

```bash
# 容器内（以 rs-mllm 镜像为例）
docker run --rm -it rs-mllm bash
cd /app
# 1. 卸载 uv 已装的部分（避免残留）
.venv/bin/pip uninstall -y $(.venv/bin/pip freeze | cut -d= -f1) 2>/dev/null
# 2. 用 requirements 重装（顶层钉版）
.venv/bin/pip install -r requirements.txt
# 3. 生成当前环境的 lock 快照（留证）
.venv/bin/pip freeze > requirements.pip-frozen.txt
```

> `requirements.pip-frozen.txt` 是**当时环境的完整快照**（含全部间接依赖版本），可作为该次回滚的版本记录；但**不是** uv 管理的 lock 文件，后续仍以 `uv.lock` 为准。

## 分层验证记录

| 环境 | 验证结果 |
|---|---|
| slurm uv | 121 包同步 + 全链 import ✓ |
| xd1 uv（A100）| sync + GPU import（cuda=True）✓ |
| xd1 Docker | 镜像构建 ✓ + `--gpus all` 容器 torch 验证 ✓ + cv2 全链 import ✓ |
| slurm conda | （见 conda 检测结果）|
