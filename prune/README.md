# Token Pruning for Qwen3.5-4B

三种免训练的视觉 token 剪枝方法，均通过 `evaluation/main_qwen35vl.py` 的 `--prune_method` 参数调用。

## 用法

```bash
# 不剪枝（默认）
python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets all

# L2-Norm 剪枝（保留 R% 的图像 token）
python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets all \
    --prune_method l2 --prune_r 0.5

# DivPrune 多样性剪枝
python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets all \
    --prune_method divprune --prune_r 0.5

# FastV-K 剪枝（K=2，前2层全量，第2层起 mask）
python -m evaluation.main_qwen35vl \
    --model_path /home/u2024311149/models/Qwen3.5-4B \
    --datasets all \
    --prune_method k2 --prune_r 0.5
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--prune_method` | str | `None` | `l2` / `k2` / `divprune`（不指定则不剪枝） |
| `--prune_r` | float | `0.5` | 剪枝比例，0.5 = 保留 50% |

## 三种方法对比

| 方法 | 原理 | 位置 | 真实加速 |
|------|------|------|------|
| **l2** | L2 范数 top-k，选"最强"token | 进入 LLM 前 | ✅ 直接删除 |
| **divprune** | 余弦距离 max-min 贪心，选"最互补"token | 进入 LLM 前 | ✅ 直接删除 |
| **k2** | 第 1 层后 L2-norm 打分，第 2 层起 mask | LLM 内部 | ❌ mask 模式 |

## 实现位置

- `evaluation/_patches/prune.py`：全部剪枝逻辑
- `evaluation/adapters/qwen35vl.py`：适配器集成
- `evaluation/main_qwen35vl.py`：CLI 入口

## 效果（pre-LLM, R=0.5）

| 数据集 | 指标 | Baseline | l2 (R=0.5) |
|------|------|------|------|
| VRSBench VQA | Acc | 52.31% | 52.18% |
| MME | Acc | 35.73% | 38.22% |
| XLRS | Acc | 35.55% | 36.10% |
| LEVIR-CC | BLEU-4 | 1.72% | 1.79% |
