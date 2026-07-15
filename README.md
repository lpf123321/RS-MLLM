# 多模态遥感大模型

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
datasets/VRSBench/
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

**使用方式**：微调时直接读取 `datasets/VRSBench/vrsbench_train.jsonl` 和 `datasets/VRSBench/vrsbench_eval.jsonl` 即可。图像路径为绝对路径，需确保软链接 `datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始标注重新生成，可运行 `python scripts/preprocess_vrsbench.py`。

### LEVIR-CC（已完成预处理）

LEVIR-CC 是遥感变化描述数据集，每对图像包含 5 句描述。已预处理为 Qwen3-VL 的 `messages` 格式，位于共享数据集目录中：

```
datasets/LEVIR-CC/
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

**使用方式**：微调时读取 `datasets/LEVIR-CC/levircc_train.jsonl`，评测时读取 `levircc_val.jsonl` / `levircc_test.jsonl` 并使用 `references` 字段计算多参考指标。图像路径为绝对路径，需确保软链接 `datasets → /users/u2024311136/shared/shared_datasets` 已建立。

> 如需从原始标注重新生成，可运行 `python scripts/preprocess_levircc.py`。

### XLRS-Bench-lite（已完成预处理）

XLRS-Bench-lite 是遥感多选问答（MCQ）评测集，共 **3,080 条** QA 对，覆盖 **13 个子任务**（目标分类、颜色、计数、空间关系、复杂推理等）。所有图像嵌入在 HuggingFace Datasets Arrow 格式中（3080 条对 800 张唯一图片，每张图约 3.85 个问题）。

已预处理为 Qwen3-VL `messages` 格式，位于：

```
output/
├── xlrs.jsonl              # 3,080 行
└── images/xlrs_*.png       # 800 张 resize 后的图片（最长边 ≤ 1024）
```

**格式说明：**

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/absolute/path/to/xlrs_00000.png"},
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

> 如需从原始 Arrow 文件重新生成，可运行 `python scripts/preprocess_xlrs.py`。

### MME-RealWorld-RS（已完成预处理）

MME-RealWorld-RS 是 MME-RealWorld 基准的遥感子集，共 **3,736 条** MCQ，覆盖 **3 个类别**（color / count / position），涉及 **1,264 张**唯一遥感图像（来自 DOTA-v2 和 Toronto 数据集）。

已预处理为 Qwen3-VL `messages` 格式，位于：

```
output/
├── mme_rs.jsonl            # 3,736 行
└── images/mme_*.png        # 1,264 张 resize 后的图片（最长边 ≤ 1024）
```

**格式说明**（与 XLRS-Bench-lite 完全一致）：：

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/absolute/path/to/mme_03553_Toronto.png"},
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

> 如需从原始 JSON + PNG 重新生成，可运行 `python scripts/preprocess_mme.py`。

## finetune_framework 说明
### 1、VRSbench
#### 目录结构

```
finetune_framework/
├── Qwen-VL-Series-Finetune/       # Qwen 官方微调框架
└── VRSbench/                      # VRSBench 微调工作区
    ├── finetune_test.sh           # LoRA 微调脚本
    ├── merge_lora.sh              # LoRA 权重合并脚本
    ├── VRSBench_train.json        # 训练数据（LLaVA 格式，142,383 条）
    └── output/                    # 训练输出（checkpoint + LoRA 权重 + TensorBoard）
```

#### 快速开始

```bash
# 1. 激活环境
conda activate rs_mllm
cd finetune_framework/VRSbench

# 2. 运行微调
bash finetune_test.sh

# 3. 查看结果：终端输出 loss / grad_norm / lr，或查看 trainer_state.json
```

#### 数据格式

训练代码（`SupervisedDataset`）使用 **LLaVA 格式**。VRSBench 原始标注已预处理并过滤后放置在 `finetune_framework/VRSbench/VRSBench_train.json`（从原始 142,390 条中移除了 7 条损坏图片的条目，剩余 142,383 条）：内部通过llava_to_openai() 转换为 openai喂给qwen3-vl

```json
[
  {
    "image": "00002_0000.png",
    "conversations": [
      {"from": "human", "value": "<image>\n[VQA] 问题文本"},
      {"from": "gpt",   "value": "答案文本"}
    ]
  }
]
```

`image` 字段为文件名，运行时通过 `--image_folder` 拼接为完整路径。三种任务通过 `[VQA]`/`[CAP]`/`[REF]` 前缀区分，统一由 cross-entropy loss（仅计算 assistant 回复部分）优化。


#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | 本地 Qwen3-VL-2B | 模型路径或 HuggingFace ID |
| `--data_path` | `VRSBench_train.json` | 训练数据，LLaVA 格式 JSON 数组 |
| `--image_folder` | `Images_train/` | 图片所在目录 |
| `--lora_enable` | `True` | LoRA 微调（单卡必需） |
| `--lora_rank` | `32` | LoRA 秩，越大适配能力越强但参数越多 |
| `--freeze_llm` | `True` | 冻结 LLM 主干，仅训练 LoRA adapter |
| `--freeze_vision_tower` | `False` | 是否冻结视觉编码器 |
| `--learning_rate` | `1e-4` | LoRA 学习率（全量微调用 `1e-5`） |
| `--deepspeed` | `zero2.json` | ZeRO-2 单卡可用；ZeRO-3 offload 需 GLIBC ≥ 2.29 |
| `--per_device_train_batch_size` | `2` | 每卡 batch size |
| `--gradient_accumulation_steps` | `16` | 梯度累积步数，有效 batch = bs × acc |
| `--image_min_pixels` | `512×32²` | 图片最小像素数（Qwen3-VL 须为 32² 的倍数） |
| `--image_max_pixels` | `1280×32²` | 图片最大像素数 |

脚本调用qwen微调框架中的train_sft.py进行微调
#### 训练输出

```
output/finetune_test/
├── runs/                       # TensorBoard 事件文件
└── checkpoint-*/               # DeepSpeed 断点，每个 checkpoint 内有：
    ├── adapter_model.safetensors   # 该 checkpoint 的 LoRA 权重（~200 MiB）
    ├── adapter_config.json         # LoRA 超参（rank/alpha/dropout）
    ├── non_lora_state_dict.bin     # 非 LLM 的可训练参数（vision tower / merger）
    ├── trainer_state.json          # 训练日志（loss / grad_norm / lr / step）
    └── ...                      # tokenizer / scheduler / training_args 等
```

训练过程中每隔 `--save_steps` 步保存一个 checkpoint，`adapter_model.safetensors` 等文件位于每个 checkpoint 目录内。训练正常结束后，最优模型的权重也会输出到根目录。如训练中断，可使用任意 checkpoint 直接合并或续训。

> 再次运行脚本时，如 `checkpoint-*` 存在会自动从断点续训。如需从头开始，先 `rm -rf output/finetune_test`。

#### LoRA 权重合并

LoRA 微调只训练 adapter 旁路矩阵（A、B），**基座模型权重未被修改**。合并操作将 LoRA 权重融入原模型（`W_new = W + BA`），得到一个完整的微调后模型。


**使用方法：**

```bash
cd finetune_framework/VRSbench

# 合并最终模型（需训练完成，根目录有 adapter_model.safetensors）
bash merge_lora.sh

# 合并指定 checkpoint（训练中断也能用）
bash merge_lora.sh checkpoint-200

# 指定输出路径
bash merge_lora.sh checkpoint-600 output/merged_model_ckpt600
```

合并后的完整模型默认输出到 `output/merged_model/`，包含模型权重 + tokenizer + processor，可直接用于推理或部署。




## Qwen3.5-4B 遥感 LoRA 微调

该流程使用 `uv` 锁定 ms-swift / PyTorch 环境，训练数据由 VRSBench caption、VRSBench VQA 和 LEVIR-CC 三个来源组成。训练/验证集按图像分组隔离，避免同一遥感图像跨集合泄漏；XLRS-Bench-lite 与 MME-RealWorld-RS 只用于最终评测，不参与微调。

模型放在 `.ms_cache/models/Qwen--Qwen3.5-4B/snapshots/master`。启动脚本会按 `model_manifests/qwen35-4b.sha256` 校验配置和权重分片，防止缓存中的基座模型静默变化。

```bash
# 登录节点：创建锁定环境与无泄漏训练/验证集
uv sync --project environments/qwen35-finetune
uv run --project environments/qwen35-finetune ./prepare_qwen35_rs_data.py
mkdir -p logs results

# 提交单卡 A100 LoRA：1 epoch，batch size 1，梯度累积 8
sbatch --export=ALL,MODE=pilot scripts/run_qwen35_rs_lora_slurm.sh
```

训练完成后，在相同验证样本上分别运行基座和 LoRA，并做配对统计：

```bash
OUTPUT_PARENT=$(cat results/qwen35_rs_lora_latest_pilot.txt)
CHECKPOINT=$(find "$OUTPUT_PARENT" -type d -name 'checkpoint-*' -print | sort -V | tail -1)

sbatch --export=ALL,MODE=base,SAMPLES=100 scripts/benchmark_qwen35_rs_slurm.sh
sbatch --export=ALL,MODE=adapter,SAMPLES=100,ADAPTER_DIR="$CHECKPOINT" \
  scripts/benchmark_qwen35_rs_slurm.sh

# 将 JOB_ID 替换为两个作业的实际编号
uv run --project environments/qwen35-finetune ./evaluate_swift_finetune.py \
  --base results/qwen35_rs_benchmark_base_100_BASE_JOB_ID.jsonl \
  --adapter results/qwen35_rs_benchmark_adapter_100_ADAPTER_JOB_ID.jsonl \
  --out-dir results/qwen35_rs_comparison_100
```

`evaluate_swift_finetune.py` 会先核对提示词、图像路径与参考答案完全一致，再报告 VQA exact match、caption / change-caption 的 token-F1 和 ROUGE-L。caption 指标是词面代理指标，不能替代 CIDEr、METEOR 或语义评审。

### 2026-07-15 无泄漏 pilot 结果

- 数据：3000 条训练、300 条图像隔离验证，训练/验证图像交集为 0；
- 训练：A100 40GB，1 epoch / 375 step，17 分 3 秒，峰值显存 11.3 GiB；
- 参数：Qwen3.5-4B，LoRA `r=16`、`alpha=32`，仅注入语言层 `q/k/v/o_proj`，可训练参数 3.1457M（0.0693%）；
- 收敛：验证损失 `1.0708 → 0.9734 → 0.9575`，最终验证 token accuracy 为 `0.6245`；
- 对照：从无泄漏验证集中固定抽取 100 条，基座与 LoRA 使用完全相同的提示词、图像和参考答案。

| 任务 | 指标 | n | 基座 | LoRA | 差值 |
|---|---:|---:|---:|---:|---:|
| VRSBench caption | ROUGE-L F1 | 28 | 0.1821 | 0.3482 | +0.1662 |
| VRSBench caption | token-F1 | 28 | 0.2900 | 0.4539 | +0.1639 |
| LEVIR-CC change-caption | ROUGE-L F1 | 36 | 0.0512 | 0.6292 | +0.5781 |
| LEVIR-CC change-caption | token-F1 | 36 | 0.0653 | 0.6606 | +0.5953 |
| VRSBench VQA | exact match | 36 | 0.3056 | 0.6944 | +0.3889 |
| VRSBench VQA | token-F1 | 36 | 0.3489 | 0.7315 | +0.3826 |

这些是内部 pilot 验证结果，不等同于 VRSBench / LEVIR-CC 官方测试集成绩，也不能外推到未参与微调的 XLRS-Bench-lite 与 MME-RealWorld-RS。

## 候选模型评测

候选模型统一维护在 `candidate_registry.py`。下载只在登录节点执行，GPU 作业只做加载和推理。

```bash
# 登录节点：确认已缓存模型，不占 GPU
python3 download_models.py --group cached

# 登录节点：只解析候选模型缓存和 VRSBench 样本图片，不加载模型
python3 eval_candidates.py --candidate minicpm_v46 --dataset vrsbench --task vqa --resolve-only

# GPU 作业内：跑一个最小 fit smoke
python3 eval_candidates.py --candidate minicpm_v46 --dataset vrsbench --task vqa --stage fit --max-samples 3

# 汇总 registry、缓存状态和评测结果
python3 summarize_candidates.py
```

汇总文件输出到 `results/candidate_eval_summary.md`，原始预测和每次运行 summary 输出到 `results/candidate_*.jsonl` / `results/candidate_*_summary.json`。

### 2026 专项候选冒烟

- `UHR-BAT` 已接入统一候选评测器；它需要 `environments/uhrbat` 中锁定的 Transformers 4.43.4 环境，以及 `.ms_cache/models/openai--clip-vit-large-patch14-336` 下的本地视觉塔。
- `Satellite Omni LISA` 同时覆盖文本与分割输出，使用 Linux 计算节点上的 `environments/satellite-omni` 环境运行 `scripts/smoke_satellite_omni.py`。
- `Count Anything` 是数量题专家，使用 Linux 计算节点上的 `environments/count-anything` 环境运行 `scripts/smoke_count_anything.py`；推理只需要发布的 `count_anything.pt`，不需要受限访问的 SAM3 初始权重。

这些专项模型与通用 VLM 的接口和输出形式不同，先做独立冒烟与资源测量，再决定是否接入路由系统；不能把单张样本结果当作正式精度。

## 结果评测与统计

baseline 推理完成后，使用 `evaluate_results.py` 汇总 JSONL 结果，输出 accuracy、文本代理指标、bootstrap 置信区间和可复现实验记录。

```bash
python3 evaluate_results.py \
  --run qwen3_vl2b:results/baseline_vrsbench_full_1335.jsonl \
  --run qwen3_vl2b:results/baseline_xlrs_lite_full_1335.jsonl \
  --run qwen3_vl2b:results/baseline_mme_realworld_full_1336.jsonl \
  --run qwen3_vl2b:results/baseline_levir_cc_full_merged.jsonl \
  --out-dir results/eval_reports/qwen3_vl2b_baseline \
  --bootstrap 5000 \
  --seed 20260708
```

输出文件：

- `summary.csv`：适合复制到表格或飞书周报；
- `summary.md`：适合放入技术报告；
- `summary.json`：适合程序继续处理；
- `unit_scores.jsonl`：逐样本/逐统计单元分数，用于 bootstrap 和 paired comparison；
- `run_manifest.json`：记录输入文件、行数、参数和运行环境，保证可复现；
- `across_run_summary.csv`：当同一任务有多次 run 时，统计 observed 指标的跨运行均值、标准差、最小值和最大值。

说明：caption / change-caption 当前使用不依赖第三方库的 lexical proxy metrics；官方 METEOR / CIDEr / LLM-as-Judge 不在第一版脚本中。VRSBench Refer 会同时报告 `mean_iou`、`accuracy_at_0_5`、`accuracy_at_0_7` 和仅用于诊断输出格式的 `format_validity`；周会与模型对比应使用 IoU/Accuracy 指标，不能把 `format_validity` 当作定位准确率。

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
