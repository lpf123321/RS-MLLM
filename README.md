# 多模态遥感大模型

> 项目结构与技术报告章节对照说明见 [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)。
> 顶层保留 `evaluation` / `prune` / `token_compression` / `scripts` 等 Python 包路径以维持内部 import，运行入口集中在 `slurm_scripts/` 与 `evaluation/main.py`。
> 训练蒸馏/自进化（`training/`）、模型量化（`quantization/`）、星载容错（`tolerance/`）为待上传模块占位。

## 环境配置

```bash
# 克隆仓库
git clone git@github.com:lpf123321/RS-MLLM.git
cd RS-MLLM

# 一键安装环境（自动建 conda 环境 + 装所有依赖）
bash setup.sh
```

## 模型权重

模型权重和数据集存放在共享区，需要手动建立软链接：

```bash
# 模型
mkdir -p models
ln -s /users/u2024311136/shared/shared_models/Qwen3-VL-2B-Instruct models/Qwen3-VL-2B-Instruct

# 数据集
mkdir -p datasets
ln -s /users/u2024311136/shared/shared_datasets datasets
```

## 数据预处理

### VRSBench（已完成预处理）

VRSBench 训练和评测数据已预处理为 Qwen3-VL 的 `messages` 格式，位于共享数据集目录中：

```
datasets/shared_datasets/VRSBench/
├── vrsbench_train.jsonl    # 训练数据，142,390 条
└── vrsbench_eval.jsonl     # 评测数据，62,918 条
```

建立好 `datasets` 软链接后可直接读取，无需再运行预处理脚本。

**格式说明**（每条数据一个 JSON 对象，每行一条）：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/absolute/path/to/image.png"},
        {"type": "text", "text": "[VQA] 指令文本"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "答案文本"}
      ]
    }
  ]
}
```

**任务前缀说明**：

| 前缀 | 任务 | 示例输出 |
|------|------|---------|
| `[VQA]` | 视觉问答 | `"expressway-toll-station"` |
| `[CAP]` | 图像描述 | `"The image shows..."` |
| `[REF]` | 指代表达定位 | `"{<45><45><59><59>}"` |

**使用方式**：微调时直接读取 `datasets/shared_datasets/VRSBench/vrsbench_train.jsonl` 和 `datasets/shared_datasets/VRSBench/vrsbench_eval.jsonl` 即可。图像路径为绝对路径，需确保软链接 `datasets/shared_datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始标注重新生成，可运行 `python scripts/preprocess_vrsbench.py`。

### LEVIR-CC（已完成预处理）

LEVIR-CC 是遥感变化描述数据集，每对图像包含 5 句描述。已预处理为 Qwen3-VL 的 `messages` 格式，位于共享数据集目录中：

```
datasets/shared_datasets/LEVIR-CC/
├── levircc_train.jsonl    # 训练数据，34,075 条（每 caption 独立一条）
├── levircc_val.jsonl      # 验证数据，1,333 条（分组，5 条 reference / 条）
└── levircc_test.jsonl     # 评测数据，1,929 条（分组，5 条 reference / 条）
```

建立好 `datasets` 软链接后可直接读取，无需再运行预处理脚本。

**格式说明**：

- **train**：每句 caption 展开为 1 条独立样本（同 VRSBench messages 格式）
- **val / test**：每个 image pair 合为 1 条，保留 5 条 reference 用于多参考评测

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/absolute/path/to/train/A/image.png"},
        {"type": "image", "image": "/absolute/path/to/train/B/image.png"},
        {"type": "text", "text": "[CD] Describe the changes between these two images."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "first caption"}
      ]
    }
  ],
  "references": [
    "first caption",
    "second caption",
    "third caption",
    "fourth caption",
    "fifth caption"
  ]
}
```

**与 VRSBench 格式差异**：

| 维度 | VRSBench | LEVIR-CC |
|------|---------|---------|
| image 数量 | 1 张 | 2 张（A 变化前，B 变化后） |
| 任务前缀 | `[CAP]` `[VQA]` `[REF]` | `[CD]` |
| 指令文本 | 来自原对话 | 固定 `"Describe the changes between these two images."` |

**使用方式**：微调时读取 `datasets/shared_datasets/LEVIR-CC/levircc_train.jsonl`，评测时读取 `levircc_val.jsonl` / `levircc_test.jsonl` 并使用 `references` 字段计算多参考指标。图像路径为绝对路径，需确保软链接 `datasets/shared_datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始标注重新生成，可运行 `python scripts/preprocess_levircc.py`。

### XLRS-Bench-lite（已完成预处理）

XLRS-Bench-lite 是遥感多选问答（MCQ）评测集，共 **3,080 条** QA 对，覆盖 **13 个子任务**（目标分类、颜色、计数、空间关系、复杂推理等）。所有图像嵌入在 HuggingFace Datasets Arrow 格式中（3080 条对 800 张唯一图片，每张图约 3.85 个问题）。

已预处理为 Qwen3-VL `messages` 格式，位于共享数据集目录：

```
datasets/shared_datasets/XLRS-Bench-lite/
├── xlrs.jsonl                    # 3,080 行
└── images_resized/xlrs_*.png     # 800 张 resize 后的图片（最长边 ≤ 1024）
```

**格式说明：**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/users/u2024311136/shared/shared_datasets/XLRS-Bench-lite/images_resized/xlrs_00000.png"},
        {"type": "text", "text": "[MCQ] The width and calmness of the river in the picture suggest?\n(A) It is a vital...\n(B) This area...\n(C) This area...\n(D) It may serve..."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "D. (D) It may serve as a vital natural water source, crucial for the surrounding ecosystem."}
      ]
    }
  ]
}
```

| 维度 | 说明 |
|------|------|
| 任务前缀 | `[MCQ]` |
| 指令 | question + 4 个选项（`\n` 拼接） |
| 答案格式 | `"X. (X) 完整选项文本"` |
| 图像数量 | 1 张 |
| 分辨率 | resize 至最长边 ≤ 1024（BICUBIC，保持宽高比） |

**使用方式**：微调时直接读取 `datasets/shared_datasets/XLRS-Bench-lite/xlrs.jsonl` 即可。图像路径为绝对路径，需确保软链接 `datasets/shared_datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始 Arrow 文件重新生成，可运行 `python scripts/preprocess_xlrs.py`。

### MME-RealWorld-RS（已完成预处理）

MME-RealWorld-RS 是 MME-RealWorld 基准的遥感子集，共 **3,736 条** MCQ，覆盖 **3 个类别**（color / count / position），涉及 **1,264 张**唯一遥感图像（来自 DOTA-v2 和 Toronto 数据集）。

已预处理为 Qwen3-VL `messages` 格式，位于共享数据集目录：

```
datasets/shared_datasets/MME-RealWorld-RS/
├── mme_rs.jsonl                  # 3,736 行
└── images_resized/mme_*.png      # 1,264 张 resize 后的图片（最长边 ≤ 1024）
```

**格式说明**（与 XLRS-Bench-lite 完全一致）：：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/users/u2024311136/shared/shared_datasets/MME-RealWorld-RS/images_resized/mme_03553_Toronto.png"},
        {"type": "text", "text": "[MCQ] What color is the roof of the square building in the lower right area of the picture?\n(A) Yellow\n(B) Blue\n(C) Gray\n(D) White\n(E) The image does not feature the color."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "D. (D) White"}
      ]
    }
  ]
}
```

**与 XLRS-Bench-lite 的格式一致性：**

| 维度 | XLRS-Bench-lite | MME-RealWorld-RS |
|------|----------------|-----------------|
| 任务前缀 | `[MCQ]` | `[MCQ]` |
| 指令 | question + 选项 | question + 选项 |
| 答案 | `"X. (X) text"` | `"X. (X) text"` |
| 图像数量 | 1 张 | 1 张 |
| 分辨率处理 | resize ≤ 1024 | resize ≤ 1024 |

两个数据集的 `jsonl` 可直接合并，作为同一类 MCQ 任务参与等比例采样。

**使用方式**：微调时直接读取 `datasets/shared_datasets/MME-RealWorld-RS/mme_rs.jsonl` 即可。图像路径为绝对路径，需确保软链接 `datasets/shared_datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始 JSON + PNG 重新生成，可运行 `python scripts/preprocess_mme.py`。

## finetune_framework 说明

微调目录已按公共后端、数据集 recipe、运行产物分层：

```text
finetune_framework/
├── backends/qwen_vl/       # 公共 Qwen-VL 训练后端
├── recipes/vrsbench/       # VRSBench SFT / GRPO
├── recipes/levir_cc/       # LEVIR-CC 数据准备 / SFT
├── outputs/vrsbench/       # VRSBench checkpoint 与模型
└── outputs/levir_cc/       # LEVIR-CC checkpoint 与模型
```

从仓库根目录启动：

```bash
conda activate rs_mllm

# VRSBench
bash finetune_framework/recipes/vrsbench/train_sft.sh
bash finetune_framework/recipes/vrsbench/merge_lora.sh
bash finetune_framework/recipes/vrsbench/train_grpo_referring.sh --skip_merge --skip_prepare

# LEVIR-CC
python finetune_framework/recipes/levir_cc/prepare_data.py
bash finetune_framework/recipes/levir_cc/train_sft.sh
bash finetune_framework/recipes/levir_cc/merge_lora.sh
```

目录职责、完整路径、数据规模、GRPO reward 行为、旧路径映射及验证方式见 [`finetune_framework/README.md`](finetune_framework/README.md)。任务细节分别见 [`recipes/vrsbench/README.md`](finetune_framework/recipes/vrsbench/README.md) 和 [`recipes/levir_cc/README.md`](finetune_framework/recipes/levir_cc/README.md)。

## Visual Token Pruning

The Qwen3.5 evaluation path supports the following visual-token compression methods:

| Method | Adapter | Description |
|--------|---------|-------------|
| `uniform` | `qwen35_pruned` | Retains evenly spaced visual tokens. |
| `random` | `qwen35_pruned` | Retains a deterministic random subset controlled by `--pruner_seed`. |
| `mmtok` | `qwen35_pruned` | Greedy cosine-similarity coverage selection. |
| `l2norm` | `qwen35_pruned` | Retains tokens with the largest feature L2 norms. |
| `scope_l2` | `qwen35_pruned` | Combines L2 saliency with cosine-similarity coverage. |
| DivPrune | `qwen35_divprune` | Greedily retains a max-min diverse subset by cosine distance. |
| Fourier | `qwen35_fourier` | Retains low-frequency 2D DCT components of each visual feature grid. |

For compressed inference, use `--eval_batch_size 1`. For example, run L2-norm pruning at a 50% token keep ratio with:

```bash
python -m evaluation.main \
  --model_path /path/to/Qwen3.5-checkpoint \
  --adapter qwen35_pruned \
  --pruner l2norm \
  --keep_ratio 0.5 \
  --eval_batch_size 1 \
  --datasets mme
```

For consistent XLRS memory comparisons, use the committed fixed subset at
`evaluation/data/xlrs_memory_1000.jsonl`. It contains the exact 1,000 records
used for the existing memory measurements; its source checksum and selection
indices are stored in `evaluation/data/xlrs_memory_1000.metadata.json`.

```bash
sbatch evaluation/run_xlrs_memory.slurm
```

The script uses this fixed subset by default. To regenerate it from the
canonical XLRS JSONL, run `evaluation/tools/create_xlrs_memory_subset.py` with
the paths in the metadata file.

DivPrune and Fourier use dedicated adapters:

```bash
python -m evaluation.main --model_path /path/to/Qwen3.5-checkpoint \
  --adapter qwen35_divprune --keep_ratio 0.5 --eval_batch_size 1 --datasets mme
python -m evaluation.main --model_path /path/to/Qwen3.5-checkpoint \
  --adapter qwen35_fourier --keep_ratio 0.5 --eval_batch_size 1 --datasets mme
```

To measure GPU memory for a method in a fresh process, use `evaluation.benchmark_memory`:

```bash
python -m evaluation.benchmark_memory \
  --model_path /path/to/Qwen3.5-checkpoint \
  --dataset mme \
  --data_path /path/to/mme_rs.jsonl \
  --method scope \
  --keep_ratio 0.5 \
  --sample_count 20 \
  --output evaluation/results/memory_scope_r50.json
```

## 提交代码

```bash
# 1. 拉取最新代码
git checkout main && git pull

# 2. 创建自己的分支
git checkout -b feat/你的功能名

# 3. 提交并推送
git add .
git commit -m "feat: 描述你的改动"
git push origin feat/你的功能名

# 4. 在 GitHub/GitLab 上创建 Pull Request
```
