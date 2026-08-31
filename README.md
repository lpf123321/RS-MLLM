# RS-MLLM（揭榜挂帅 · 星载遥感多模态大模型轻量化部署）

面向星载端侧资源受限环境的多模态大模型部署方案：数据集处理 → 基座选型（Qwen3.5-4B）→ 多专家微调 + 在线蒸馏 → 视觉 Token 剪枝 + 模型量化 → 星载容错与故障恢复。

![overall](assets/overall.png)

## 快速开始

```bash
# 1) 环境（conda）
bash setup.sh

# 2) 评测（全量/固定清单）
sbatch slurm_scripts/run_task_adaptive.slurm
# 或交互式
bash evaluation/run_eval.sh --datasets all --max_samples 500

# 3) 训练 / 量化 / 容错
sbatch slurm_scripts/run_task_adaptive.slurm   # 四专家路由与剪枝评测
sbatch slurm_scripts/quant_eval_950.slurm <model> <profile> <outdir>   # 量化模型 950 评测
```

## 目录

| 目录 | 内容 |
|---|---|
| `datasets_data/` | 四个数据集评测清单（jsonl）|
| `evaluation/` | 评测框架（adapters/router/metrics/evalsets，入口 `main.py`）与结果（`results.json`）|
| `finetune_framework/` | Qwen-VL 系列微调框架（SFT/DPO/GRPO）与训练数据构建 |
| `prune/` `token_compression/` | 免训练视觉 Token 剪枝（L2、SCOPE）与压缩方法集 |
| `quantization/` | W8A8-INT8 / W4A16-GPTQ 量化与配对评测（方法+结果见 `README.md`）|
| `training/` | 在线策略蒸馏（OPD/OPSD）、CVSearch 自进化（见目录内 README）|
| `tolerance/` | 星载容错：故障模型、不可变封套、运行时安全状态机 |
| `deploy/` | 一键部署脚本（GPU 自适应）|
| `scripts/` `slurm_scripts/` | 预处理/推理/delta 工具与 SLURM 提交脚本 |
| `docs/` | 代码结构说明（`PROJECT_STRUCTURE.md`）与待上传清单（`TODO_UPLOAD.md`）|

各目录使用方式见 `docs/PROJECT_STRUCTURE.md`；评测/训练细节见对应目录 README。

## 结果

- 四专家路由 + 剪枝的评测结果：`evaluation/results.json`（各任务 Accuracy/CIDEr/ROUGE-L/BLEU-4）
- 量化实测（770/590 子集）：见 `quantization/README.md`
- 容错实验：见 `tolerance/README.md`
