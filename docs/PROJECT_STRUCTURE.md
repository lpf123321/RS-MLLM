# 项目结构与技术报告章节对照

本仓库为比赛提交代码。顶层以下目录均以**技术报告（`~/tech_report/`）章节**为纲组织，
区域内仍保留原有的 Python 包路径（`evaluation` / `prune` / `token_compression` / `scripts`），
以保证内部 `import` 不被破坏、代码可直接运行。

## 顶层布局

| 目录 | 对应报告章节 | 内容 |
|------|--------------|------|
| `datasets_data/` | 第 3 章 数据集构建与处理 | 各数据集评测清单（jsonl）与数据说明 |
| `deploy/` | 第 8 章 模型推理说明 | 一键部署脚本（GPU 自适应，含推理/评估/可选手微调） |
| `evaluation/` | 第 7 章 实验结果分析 | 评测框架：`adapters/`（含 multiexpert + 剪枝 router）、`router/`、`metrics/`、`evalsets/`、`main.py` |
| `finetune_framework/` | 第 5 章 模型训练方法 | Qwen-VL 系列微调框架（SFT/DPO/GRPO）与各任务训练数据构建 |
| `prune/` | 第 6 章 视觉 Token 压缩 | 免训练视觉 token 剪枝方法（L2、SCOPE）与可视化 |
| `token_compression/` | 第 6 章 视觉 Token 压缩 | 免训练 token 压缩方法集（uniform/random/mmtok/l2norm/scope_l2/divprune/fourier） |
| `scripts/` | 全部 | 预处理、推理、剪枝推理、delta 生成、工具脚本（与 prune.py 的 `import` 依赖强相关，保留原位） |
| `scripts/training/` | 第 5 章 模型训练方法 | 训练阶段脚本（兼容 SLURM 头，`train.sh --local` 前台直跑） |
| `archive/slurm_experiments/` | — | 历史 SLURM 提交脚本归档（原始实验证据，不参与运行） |
| `training/` | 第 5 章 模型训练方法（蒸馏/自进化） | `distillation/`（5.2 OPD/OPSD）、`self_evolution/`（5.3 CVSearch）——**待上传** |
| `quantization/` | 第 6 章 视觉 Token 压缩与轻量化 | 6.2 模型量化（INT8 / GPTQ W4A16，**待上传**）；6.1 剪枝实现见顶层 `prune/`、`token_compression/` |
| `tolerance/` | 容错与故障恢复 | 星载容错、完整性校验、运行时安全状态机——**待上传** |
| `docs/` | — | 本文档等导航说明 |
| `_archive/` | — | 一次性产物 / 探索文档（不参与运行，不入最终交付） |

## 各目录内要点

### evaluation/
- 入口：`evaluation/main.py`（`--adapter` 支持 qwen3vl / qwen35vl / router / router_pruned 等）
- 四专家 delta 路由：`evaluation/adapters/router.py`、`router_pruned.py`，规则在 `evaluation/router/rules.py`
- 任务自适应剪枝配置：`evaluation/router/task_prune_config.py`（`THRESHOLD_TASK_PRUNE_CONFIG`）
- 运行清单默认从 `datasets_data/` 读取（`DATA_ROOT` 可覆盖）

### 核心运行依赖（共享区，需保持可达）
- 模型：ModelScope（`Fun10165/*`），`rsmllm.models.get_model()` 按需下载缓存
- 评测数据：`datasets_data/`（messages 清单）+ `build_sample_manifest.py` 转换为评测器格式
- 历史 SLURM 提交脚本归档于 `archive/slurm_experiments/`（证据保留，不参与运行）

### training/ quantization/ tolerance/（待上传占位）
- `training/distillation/`：5.2 在线策略蒸馏 OPD / 在线自蒸馏 OPSD
- `training/self_evolution/`：5.3 CVSearch 驱动的自进化训练框架
- `quantization/`：6.2 模型量化（INT8 / GPTQ W4A16）
- `tolerance/`：星载容错与故障恢复设计
以上目录当前仅有 README 占位说明，对应代码待上传。

## 快速开始
```bash
# 1) 环境（已有 rs_mllm 环境可跳过）
bash setup.sh
bash evaluation/vllm_eval/setup_env.sh

# 2) 评测/推理（唯一交互入口）
./rsmllm.sh
```