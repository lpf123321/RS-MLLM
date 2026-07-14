# VRSBench 微调与评估

基于 [Qwen-VL-Series-Finetune](https://github.com/2UQ3E260/Qwen-VL-Series-Finetune) 框架，对 Qwen3-VL-2B-Instruct 在 VRSBench 遥感图像数据集上进行 LoRA 微调，并对比微调前后的 NLG 指标。

## 目录

```
finetune_framework/VRSbench/
├── finetune_test.sh          # LoRA 微调训练脚本
├── merge_lora.sh             # LoRA 权重合并脚本
├── eval_metrics.py           # 基座 vs 微调模型评估脚本
├── compare_results.py        # 评估结果对比表格生成工具
├── VRSBench_train.json       # 训练数据（Qwen-VL 对话格式）
└── output/
    ├── finetune_test/        # 训练输出（checkpoint / adapter / TensorBoard）
    ├── eval_results.json     # 评估结果 JSON
    └── eval_results_compare.md  # 自动生成的对比报告
```

## 功能概览

| 脚本 | 功能 | 类型 |
|------|------|------|
| `finetune_test.sh` | 使用 DeepSpeed ZeRO-2 + LoRA 进行单卡微调 | 训练 |
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
