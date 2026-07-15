# VRSBench 微调与评估

基于 [Qwen-VL-Series-Finetune](https://github.com/2UQ3E260/Qwen-VL-Series-Finetune) 框架，对 Qwen3-VL-2B-Instruct 在 VRSBench 遥感图像数据集上进行 LoRA 微调，并对比微调前后的 NLG 指标。

## 目录

```
finetune_framework/VRSbench/
├── finetune_test.sh               # LoRA 微调训练脚本 (SFT)
├── finetune_grpo_referring.sh     # GRPO 强化学习微调脚本 (Referring)
├── merge_lora.sh                  # LoRA 权重合并脚本
├── prepare_referring_grpo_data.py # 从混合数据提取 referring 专精数据
├── eval_metrics.py                # 基座 vs 微调模型评估脚本
├── compare_results.py             # 评估结果对比表格生成工具
├── VRSBench_train.json            # 训练数据（Qwen-VL 对话格式）
├── VRSBench_referring_grpo.json   # Referring-only GRPO 训练数据
└── output/
    ├── finetune_test/             # SFT 训练输出（checkpoint / adapter / TensorBoard）
    ├── merged_model/              # SFT LoRA 合并后的完整模型
    ├── grpo_referring/            # GRPO 训练输出
    ├── eval_results.json          # 评估结果 JSON
    └── eval_results_compare.md    # 自动生成的对比报告
```

## 功能概览

| 脚本 | 功能 | 类型 |
|------|------|------|
| `finetune_test.sh` | 使用 DeepSpeed ZeRO-2 + LoRA 进行单卡 SFT 微调 | 训练 |
| `finetune_grpo_referring.sh` | 在 SFT 模型基础上用 IoU reward 进行 GRPO 强化学习微调 | 训练 |
| `prepare_referring_grpo_data.py` | 从 142K 混合数据中提取 36K referring 专精 GRPO 数据 | 数据处理 |
| `merge_lora.sh` | 将 LoRA adapter 合并回基座模型，生成完整权重 | 后处理 |
| `eval_metrics.py` | 加载基座/微调模型，在 VRSBench 验证集上计算指标 | 评估 |
| `compare_results.py` | 读取 `eval_results.json`，生成 终端表格 / Markdown 对比报告 | 可视化 |

## 1. 微调训练

### 1.1 前置条件

- **训练数据**：`VRSBench_train.json`（Qwen-VL 对话格式，包含 caption / VQA / referring 三类任务）
- **图片目录**：`datasets/shared_datasets/VRSBench/images/Images_train`
- **基座模型**：`models/Qwen3-VL-2B-Instruct`
- **环境**：conda 环境需安装 deepspeed 及 Qwen-VL-Series-Finetune 依赖

### 1.2 训练配置

```bash
conda activate rs_mllm
bash finetune_framework/VRSbench/finetune_test.sh
```

关键超参（可在脚本中修改）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `lora_rank` | 32 | LoRA 低秩维度 |
| `lora_alpha` | 64 | LoRA 缩放因子 |
| `learning_rate` | 1e-4 | 学习率 |
| `num_train_epochs` | 1 | 训练轮数 |
| `global_batch_size` | 32 | 全局 batch size |
| `freeze_llm` | True | 冻结 LLM 主干，仅训视觉+merger |
| `freeze_vision_tower` | False | 视觉编码器参与训练 |
| `deepspeed` | zero2.json | DeepSpeed ZeRO-2 |

训练产物保存在 `output/finetune_test/`，包含 adapter 权重和定期 checkpoint。

## 2. LoRA 权重合并

训练完成后，将 LoRA adapter 合并回基座模型：

```bash
# 合并最终 checkpoint 到默认路径
bash finetune_framework/VRSbench/merge_lora.sh

# 合并指定 checkpoint
bash finetune_framework/VRSbench/merge_lora.sh checkpoint-200

# 指定输出路径
bash finetune_framework/VRSbench/merge_lora.sh checkpoint-200 /path/to/merged_model
```

合并后的模型可直接用于推理部署，不再依赖 adapter 文件。

## 3. 评估模型

### 3.1 评估脚本

`eval_metrics.py` 支持三种评估模式：

```bash
# 完整对比：基座 vs 微调
python eval_metrics.py --task all --max_samples 500

# 仅评估基座
python eval_metrics.py --task caption --base_only --max_samples 500

# 仅评估微调
python eval_metrics.py --task caption --ft_only --max_samples 500

# 评估全部数据（不采样，耗时长）
python eval_metrics.py --task caption
```

### 3.2 参数说明

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--task` | `caption`, `vqa`, `referring`, `all` | 评估任务 |
| `--max_samples` | 整数，0=全部 | 每任务采样数 |
| `--base_only` | flag | 仅评估基座模型 |
| `--ft_only` | flag | 仅评估微调模型 |
| `--output` | 路径 | 保存 JSON 结果 |
| `--base_model` | 路径 | 基座模型路径 |
| `--lora_path` | 路径 | LoRA adapter 路径 |

### 3.3 计算的指标

**Caption / VQA 任务** — 标准 NLG 指标：

| 指标 | 说明 |
|------|------|
| BLEU-1 ~ BLEU-4 | 基于 n-gram 精确匹配的机器翻译指标 |
| ROUGE-L | 最长公共子序列召回率 |
| METEOR | 考虑同义词和词形变化的匹配指标 |
| CIDEr-D | 图像描述专用指标，带 TF-IDF 加权和长度惩罚 |

**Referring 任务** — 边界框定位指标（遵循原论文）：

| 指标 | 说明 |
|------|------|
| mean_iou | 预测框与真实框的平均 IoU |
| Acc@0.25 | IoU ≥ 0.25 的预测占比 |
| Acc@0.5 | IoU ≥ 0.5 的预测占比（标准宽松阈值） |
| Acc@0.7 | IoU ≥ 0.7 的预测占比（标准严格阈值） |

从模型输出中解析 `{<x1><y1><x2><y2>}` 格式坐标后计算，无法解析的视为无效预测。

### 3.4 依赖安装

```bash
pip install nltk rouge_score pycocoevalcap
python -m nltk.downloader wordnet punkt punkt_tab
```

## 4. 生成对比报告

评估完成后，使用 `compare_results.py` 将 `eval_results.json` 转为可读表格：

```bash
# 输出到自动命名的 Markdown 文件
python finetune_framework/VRSbench/compare_results.py output/eval_results.json
# → output/eval_results_compare.md

# 指定输出路径
python finetune_framework/VRSbench/compare_results.py output/eval_results.json -o my_report.md

# 打印终端表格（box-drawing 风格）
python finetune_framework/VRSbench/compare_results.py output/eval_results.json -t

# 打印 Markdown 到终端
python finetune_framework/VRSbench/compare_results.py output/eval_results.json -t --md
```

生成的 Markdown 报告包含：
- **页头信息**：基座模型路径、LoRA 权重路径、任务列表、样本数
- **总览表**：所有任务 × 所有指标合并为一个大表，方便横向对比
- **各任务明细**：每个任务独立的详细对比表

## 5. 典型工作流

```bash
# Step 1: 微调
conda activate rs_mllm
bash finetune_framework/VRSbench/finetune_test.sh

# Step 2: 评估（基座 vs 微调对比）
python finetune_framework/VRSbench/eval_metrics.py \
    --task all --max_samples 500 \
    --output finetune_framework/VRSbench/output/eval_results.json

# Step 3: 生成对比报告
python finetune_framework/VRSbench/compare_results.py \
    finetune_framework/VRSbench/output/eval_results.json

# (可选) 合并权重用于部署
bash finetune_framework/VRSbench/merge_lora.sh
```

## 6. 训练数据格式

训练数据 `VRSBench_train.json` 采用 Qwen-VL 标准对话格式：

```json
{
  "id": "dataset/v1.0",
  "image": "00002_0000.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n[caption] Could you describe the contents of this image for me?"
    },
    {
      "from": "gpt",
      "value": "The image, sourced from GoogleEarth, shows a rural area with..."
    }
  ]
}
```

三类任务的 prompt 标识：
- **Caption**: `[caption] Could you describe the contents of this image for me?`
- **VQA**: `[vqa] {question}`
- **Referring**: `[refer] could you tell me the location for <p>{question}</p>?`

## 7. 依赖关系

### SFT 流程

```
VRSBench_train.json ──→ finetune_test.sh ──→ output/finetune_test/
                                                 │
                    ┌────────────────────────────┘
                    ▼
              eval_metrics.py ──→ output/eval_results.json
                                        │
                    ┌───────────────────┘
                    ▼
            compare_results.py ──→ output/eval_results_compare.md
```

### GRPO 流程

```
VRSBench_train.json ──→ prepare_referring_grpo_data.py ──→ VRSBench_referring_grpo.json
                                                                     │
output/finetune_test/ ──→ merge_lora.sh ──→ output/merged_model/     │
                                                    │                │
                                                    ▼                ▼
                                         finetune_grpo_referring.sh ──→ output/grpo_referring/
```

## 8. GRPO 强化学习微调（Referring 任务）

在 SFT 模型基础上，使用 GRPO (Group Relative Policy Optimization) 结合 IoU reward 进行强化学习微调，进一步提升 referring 任务的边界框定位精度。

### 8.1 流程概览

```
SFT LoRA → Merge → GRPO (IoU Reward) → 评估
```

1. **合并 SFT LoRA**：将第一阶段 SFT 的 LoRA adapter 合并为完整模型权重
2. **准备数据**：从 142K 混合数据中提取 36K referring 专精数据
3. **GRPO 训练**：在合并模型基础上用 IoU reward 进行强化学习

### 8.2 快速开始

```bash
conda activate rs_mllm
# 完整流程（合并 + 数据准备 + GRPO 训练）
bash finetune_framework/VRSbench/finetune_grpo_referring.sh

# 跳过已完成的步骤
bash finetune_framework/VRSbench/finetune_grpo_referring.sh --skip_merge --skip_prepare

# 自定义超参
bash finetune_framework/VRSbench/finetune_grpo_referring.sh --lr 1e-5 --num_gen 8 --beta 0.04
```

关键超参（可在脚本中修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LR` | 5e-6 | GRPO 学习率（比 SFT 低） |
| `NUM_GENERATIONS` | 4 | 每个 prompt 生成的候选数 |
| `BETA` | 0.04 | KL 惩罚系数，防止偏离 SFT 模型过远 |
| `MAX_COMPLETION_LENGTH` | 64 | 生成的最大 token 数（坐标输出很短） |
| `LORA_RANK` | 16 | GRPO 阶段使用较小的 rank |
| `freeze_llm` | True | 冻结 LLM 主干 |
| `freeze_vision_tower` | False | 视觉编码器参与训练 |

### 8.3 Reward 函数 — 框架修改记录

为支持 referring 任务的 GRPO 训练，对 Qwen-VL-Series-Finetune 框架做了以下修改：

**修改文件**：`finetune_framework/Qwen-VL-Series-Finetune/src/train/reward_funcs.py`

**新增内容**：

1. **bbox 解析工具函数**（模块级私有函数）：
   - `_BBOX_PATTERN` — 匹配 `{<x1><y1><x2><y2>}` 格式的正则表达式
   - `_parse_bbox(text)` — 从文本中提取坐标，返回 `(x1, y1, x2, y2)` 或 `None`
   - `_compute_iou(box_a, box_b)` — 计算两个边界框的 IoU（支持乱序坐标）
   - `_extract_text(completion)` — 将 completion（str / list / 其他）统一转为纯文本

2. **两个 reward 函数**（GRPO 框架自动发现 `*_reward` 结尾的公开函数）：

| 函数 | 作用 | Reward 范围 |
|------|------|-------------|
| `iou_reward(completions, assistant, **kwargs)` | 解析输出中的 `{<x1><y1><x2><y2>}` 坐标，计算与 GT 的 IoU | [0, 1] |
| `iou_format_reward(completions, assistant, **kwargs)` | 检查输出是否包含合法的 bbox 格式 | {0, 1} |

**使用方式**：

两个 reward 函数可以组合使用，通过 `--reward_weights` 控制权重（详见 `finetune_grpo_referring.sh`）。脚本中通过 Python 验证加载：

```python
from train.reward_funcs import iou_reward, iou_format_reward
```

- `iou_reward` 提供连续的定位精度信号，解析失败返回 0.0
- `iou_format_reward` 鼓励模型保持正确的输出格式，防止 reward hacking

### 8.4 结果评估

```bash
# 评估 GRPO 模型
python finetune_framework/VRSbench/eval_metrics.py \
    --task referring \
    --base_model output/merged_model \
    --lora_path output/grpo_referring

# 对比 SFT 基线
python finetune_framework/VRSbench/eval_metrics.py \
    --task referring \
    --base_model models/Qwen3-VL-2B-Instruct \
    --lora_path output/finetune_test
```
