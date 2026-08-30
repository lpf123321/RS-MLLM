# Token Pruning for Qwen VL

四种免训练的视觉 token 剪枝方法，直接作用于模型 forward，无需微调。

## 快速开始

```bash
# L2 剪枝
python scripts/prune_inference.py \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --data_path datasets/shared_datasets/VRSBench/vrsbench_eval.jsonl \
    --prune_method l2 --prune_r 0.5 \
    --batch_size 4

# SCOPE-L2 剪枝（新方法）
python scripts/prune_inference.py \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --data_path datasets/shared_datasets/VRSBench/vrsbench_eval.jsonl \
    --prune_method scope --prune_r 0.5 \
    --batch_size 4

# 使用 evaluation 框架
python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets all \
    --prune_method scope --prune_r 0.5
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--model_path` | str | 必填 | 模型路径 |
| `--prune_method` | str | `None` | `l2` / `k2` / `divprune` / `scope` |
| `--prune_r` | float | `0.5` | 剪枝比例（0.5 = 保留 50% merged token） |
| `--batch_size` | int | `4` | 批量推理大小 |
| `--max_samples` | int | `0` | 最大样本数（0=全部） |
| `--start_offset` | int | `0` | 跳过前 N 条 |
| `--max_new_tokens` | int | `256` | 最大生成长度 |
| `--output` | str | `None` | 结果保存路径 |

## 四种方法

| 方法 | 打分准则 | 剪枝位置 | 特征维度 | 说明 |
|------|----------|----------|----------|------|
| **l2** | L2 范数 top-k | 进入 LLM 前 | merged (2560-dim) | 保留 L2 模长最大的 merged token |
| **divprune** | 余弦距离 max-min 贪心 | 进入 LLM 前 | merged (2560-dim) | 最大化 token 多样性 |
| **k2** | 第 1 层后 L2-norm，第 2 层起 mask | LLM 内部 | hidden (2560-dim) | 仅 mask 注意力，不删除 token |
| **scope** | L2 范数 × 余弦覆盖增益 (SCOPE-L2) | 进入 LLM 前 | raw (1024-dim) | 联合显著性+覆盖度，块级选择 |

## SCOPE-L2 详细介绍

### 原理

SCOPE (Saliency-Coverage Oriented Token Pruning) 在 **原始视觉特征层**（vision block 22，1024-dim）上同时建模两个维度：

1. **Saliency（显著性）**: 使用 L2 范数（替代原论文的 CLS attention 分数）衡量每个 token 的重要性
2. **Coverage（覆盖度）**: 通过余弦相似度矩阵，贪心选择能最大化已选集合覆盖范围的 token

每步选择的综合分数 = `覆盖率增益 × 显著性^alpha`（默认 alpha=1.0）。

### 块级剪枝

Qwen3.5-4B 使用 2×2 spatial merge（每 4 个相邻 patch token 合并为 1 个 2560-dim merged token）。SCOPE 以 block 为单位操作：
- 每 4 个 raw token 做 mean pooling → 1 个 block feature
- 在 block 级别运行 SCOPE 选取 top-K
- 被选中的 block 对应的 merged token 保留，其余删除

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ALPHA` | `1.0` | 显著性指数。alpha 越大，越偏向高 L2 范数的 token；alpha 越小，越偏向覆盖率 |
| `COMBINED` | `multi` | 组合方式。`multi`: 覆盖率增益 × 显著性；`add`: 覆盖率增益 + 显著性 |

使用示例：
```bash
# 更重视覆盖率（多样性）
ALPHA=0.3 python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets mme --prune_method scope --prune_r 0.5

# 更重视显著性（类似原版 l2）
ALPHA=3.0 python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets mme --prune_method scope --prune_r 0.5
```

### 与原 SCOPE 的对比

| | 原 SCOPE (LLaVA/CLIP) | SCOPE-L2 (Qwen3.5-4B) |
|---|---|---|
| 显著性来源 | CLS token 对 patch 的 attention | 特征向量的 L2 范数 |
| 视觉编码器 | CLIP ViT | Qwen3VL ViT |
| 拦截层 | 倒数第二层 (layer 22/23) | 倒数第二层 (block 22/23) |
| 剪枝粒度 | 单个 patch token | 2×2 spatial merge block |
| 特征维度 | 1024 | 1024（raw），映射到 2560（merged） |

### 实现细节

```
pixel_values → Qwen3VLVisionModel.forward()
    ├── Block 0..21: 正常前向
    ├── Block 22:    前向 → forward hook 捕获 _scope_features [total_raw, 1024]
    ├── Block 23:    正常前向
    └── Merger:      2×2 merge → merged tokens [total_merged, 2560]
                          ↓
Qwen3_5Model.forward() (patched)
    ├── get_image_features() → merged tokens
    ├── masked_scatter → inputs_embeds
    └── _prune_scope():
        ├── 按图像提取 raw features → 分组为 block (4 token → 1 block)
        ├── SCOPE_L2 选取 (1-r) 比例的 block
        ├── 映射到 merged token 位置
        └── 从 inputs_embeds 删除未选中 token
```

## 文件结构

```
scripts/
├── prune.py              # 剪枝核心（l2/k2/divprune/scope）
├── prune_inference.py    # 独立推理脚本
└── smoke_test_viz.py     # 冒烟测试（pruner + 可视化，无需模型/GPU）

prune/
├── base.py               # VisualTokenPruner 基类（统一 pruner 接口）
├── l2norm.py             # L2NormPruner（L2 范数 top-k）
├── viz.py                # 可视化：把 keep/prune 掩码叠加回原图
├── scope.py              # SCOPE-L2 核心算法 + 剪枝函数
├── README.md             # 本文档
└── output/               # 实验结果
```

### 可视化（prune/viz.py）

把任意 `VisualTokenPruner` 的剪枝结果映射回原图（2×2 spatial merge 块级）：

```python
from prune.l2norm import L2NormPruner
from prune.viz import prune_to_grid_mask, visualize_pruning

# features: [N, D] 视觉 token 特征；grid_h/w: merge 后网格
mask = prune_to_grid_mask(L2NormPruner(), features, grid_h, grid_w, keep_ratio=0.5)
visualize_pruning(image, mask, title="L2Norm keep 50%", save_path="viz.png")
```

冒烟测试（无需模型/GPU）：
```bash
python scripts/smoke_test_viz.py
```

## 效果（R=0.5，保留 50% 图像 token）

| 数据集 | 指标 | Baseline | l2 (R=0.5) |
|------|------|------|------|
| VRSBench VQA | Acc | 52.31% | 52.18% |
| MME | Acc | 35.73% | 38.22% |
| XLRS | Acc | 35.55% | 36.10% |
| LEVIR-CC | BLEU-4 | 1.72% | 1.79% |

> SCOPE 方法的效果待补充。建议使用 `--prune_r 0.5` 起步测试，后续可探索 `0.3`（保留 70%）等不同比例。

## 断点续跑

脚本每 200 条自动保存 `.ckpt.json` 断点。超时后重新运行同一命令即可从断点续跑。
