# Token Pruning for Qwen VL

三种免训练的视觉 token 剪枝方法，plug-and-play，通过 `--prune_method` 参数控制，不传则完全不启用。

## 用法

```bash
# 不剪枝（默认，与原版行为完全一致）
python -m evaluation.main_qwen35vl \
    --model_path /path/to/Qwen3.5-4B --datasets all

# L2-Norm 剪枝
python -m evaluation.main_qwen35vl \
    --model_path /path/to/Qwen3.5-4B --datasets all \
    --prune_method l2 --prune_r 0.5

# DivPrune 多样性剪枝
python -m evaluation.main_qwen35vl \
    --model_path /path/to/Qwen3.5-4B --datasets all \
    --prune_method divprune --prune_r 0.5

# FastV-K 剪枝（K=2）
python -m evaluation.main_qwen35vl \
    --model_path /path/to/Qwen3.5-4B --datasets all \
    --prune_method k2 --prune_r 0.5
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--prune_method` | str | `None` | `l2` / `k2` / `divprune` |
| `--prune_r` | float | `0.5` | 剪枝比例，0.5 = 保留 50% |

## 三种方法

| 方法 | 打分准则 | 剪枝位置 | 真实加速 |
|------|------|------|------|
| **l2** | L2 范数 top-k | 进入 LLM 前 | ✅ |
| **divprune** | 余弦距离 max-min 贪心 | 进入 LLM 前 | ✅ |
| **k2** | 第 1 层后 L2-norm，第 2 层起 mask | LLM 内部 | ❌ |

## 实现原理

```
Qwen3_5Model.forward (原版)
├── 图像编码 → scatter → inputs_embeds   ← 原版
├── 计算 position_ids                     ← 原版
├── ★ 剪枝（新增，无 pruning 时完全跳过）  ← 仅此 hook
│     └── visual_pos_masks 定位图像 token
│     └── L2-norm / DivPrune 打分
│     └── 低分 token 从 inputs_embeds/position_ids/attention_mask 删除
└── self.language_model(...)              ← 收到的序列已缩短
```

底层 monkey-patch 自动检测模型类型，支持 Qwen3.5 / Qwen3-VL / Qwen2.5-VL。

## 文件

```
scripts/prune.py                    # 全部剪枝逻辑
evaluation/adapters/qwen35vl.py     # 适配器（+13 行）
evaluation/main_qwen35vl.py         # CLI（+9 行）
```

## 效果（pre-LLM l2, R=0.5, 保留 50% 图像 token）

| 数据集 | 指标 | Baseline | pruned |
|------|------|------|------|
| VRSBench VQA | Acc | 52.31% | 52.18% |
| MME | Acc | 35.73% | 38.22% |
| XLRS | Acc | 35.55% | 36.10% |
| LEVIR-CC | BLEU-4 | 1.72% | 1.79% |
