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
| `slurm_scripts/` | — | SLURM 提交脚本：训练 / 评测 / delta 转换 / 冒烟测试 |
| `training/` | 第 5 章 模型训练方法（蒸馏/自进化） | `distillation/`（5.2 OPD/OPSD）、`self_evolution/`（5.3 CVSearch）——**待上传** |
| `compression/` | 第 6 章 视觉 Token 压缩与轻量化 | 导航层：`quantization/`（6.2 模型量化，**待上传**）；6.1 剪枝实现见顶层 `prune/`、`token_compression/` |
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
- 基座模型：`/users/u2024311136/shared/shared_models/lora_expert/base_model`
- 四专家 delta：`/users/u2024311136/shared/shared_models/lora_expert/lora/{general,grounding,change,caption}/delta_model.pt`
- 评测数据：`/users/u2024311136/shared/shared_datasets`

### slurm_scripts/ 分类
- 训练类：`sft_*.slurm`、`run_task_adaptive.slurm`
- delta 转换类：`convert_*_to_delta.slurm`、`convert_ground_*.slurm`
- 评测类：`run_delta_*.slurm`、`run_prune_sweep_router.slurm`
- 冒烟类：`smoke_test_*.slurm`、`smoke_delta_*.slurm`
- 可视化类：`run_viz_l2*.sh`

### training/ compression/ tolerance/（待上传占位）
- `training/distillation/`：5.2 在线策略蒸馏 OPD / 在线自蒸馏 OPSD
- `training/self_evolution/`：5.3 CVSearch 驱动的自进化训练框架
- `compression/quantization/`：6.2 模型量化（INT8 / GPTQ W4A16）
- `tolerance/`：星载容错与故障恢复设计
以上目录当前仅有 README 占位说明，对应代码待上传。

## 快速开始
```bash
# 1) 环境（已有 rs_mllm 环境可跳过）
bash setup.sh

# 2) 评测（SLURM）
sbatch slurm_scripts/run_task_adaptive.slurm
# 或交互式
python evaluation/main.py --model_path <base_model> \
    --adapter router --general_lora <general_delta> --grounding_lora <grounding_delta> \
    --change_lora <change_delta> --caption_lora <caption_delta> --datasets all

# 3) 冒烟验证四专家 + 剪枝
sbatch slurm_scripts/smoke_test_task_adaptive.slurm
```