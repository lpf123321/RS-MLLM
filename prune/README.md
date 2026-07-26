# Token Pruning for Qwen VL

三种免训练的视觉 token 剪枝方法，直接作用于模型 forward，无需微调。

## 快速开始

```bash
# 批量推理（推荐，速度更快）
python scripts/prune_inference.py \
    --model_path /path/to/Qwen3.5-4B \
    --data_path datasets/shared_datasets/VRSBench/vrsbench_eval.jsonl \
    --prune_method l2 --prune_r 0.5 \
    --batch_size 4 \
    --max_samples 100

# 单条推理风格（兼容 test_inference.py 格式）
python scripts/prune_inference.py \
    --model_path /path/to/Qwen3.5-4B \
    --data_path datasets/shared_datasets/LEVIR-CC/levircc_test.jsonl \
    --prune_method l2 --prune_r 0.5 \
    --batch_size 1
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--model_path` | str | 必填 | 模型路径 |
| `--data_path` | str | 必填 | JSONL 数据集路径 |
| `--prune_method` | str | `None` | `l2` / `k2` / `divprune` |
| `--prune_r` | float | `0.5` | 剪枝比例 |
| `--batch_size` | int | `4` | 批量推理大小 |
| `--max_samples` | int | `0` | 最大样本数（0=全部） |
| `--start_offset` | int | `0` | 跳过前 N 条 |
| `--max_new_tokens` | int | `256` | 最大生成长度 |
| `--load_in_4bit` | flag | — | 4bit 量化（与剪枝兼容） |
| `--load_in_8bit` | flag | — | 8bit 量化（与剪枝兼容） |
| `--output` | str | `None` | 结果保存路径 |

## 三种方法

| 方法 | 打分准则 | 剪枝位置 | 量化兼容 |
|------|------|------|------|
| **l2** | L2 范数 top-k | 进入 LLM 前 | ✅ |
| **divprune** | 余弦距离 max-min 贪心 | 进入 LLM 前 | ✅ |
| **k2** | 第 1 层后 L2-norm，第 2 层起 mask | LLM 内部 | ✅ |

## 文件结构

```
scripts/
├── prune.py              # 剪枝核心（自动检测 Qwen3.5/3-VL/2.5-VL）
└── prune_inference.py    # 独立推理脚本（不依赖 evaluation 框架）

evaluation/
├── adapters/qwen35vl.py  # 适配器集成（+13 行）
└── main_qwen35vl.py      # CLI 参数（+9 行）
```

## 效果（R=0.5，保留 50% 图像 token）

| 数据集 | 指标 | Baseline | l2 (R=0.5) |
|------|------|------|------|
| VRSBench VQA | Acc | 52.31% | 52.18% |
| MME | Acc | 35.73% | 38.22% |
| XLRS | Acc | 35.55% | 36.10% |
| LEVIR-CC | BLEU-4 | 1.72% | 1.79% |

## 断点续跑

脚本每 200 条自动保存 `.ckpt.json` 断点。超时后重新运行同一命令即可从断点续跑。
