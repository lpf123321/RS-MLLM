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
├── levircc_train.jsonl    # 训练数据，34,075 条
├── levircc_val.jsonl      # 验证数据，6,665 条
└── levircc_test.jsonl     # 评测数据，9,645 条
```

建立好 `datasets` 软链接后可直接读取，无需再运行预处理脚本。

**格式说明**（每条数据一个 JSON 对象，每行一条）：

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
        {"type": "text", "text": "caption text"}
      ]
    }
  ]
}
```

**与 VRSBench 格式差异**：

| 维度 | VRSBench | LEVIR-CC |
|------|---------|---------|
| image 数量 | 1 张 | 2 张（A 变化前，B 变化后） |
| 任务前缀 | `[CAP]` `[VQA]` `[REF]` | `[CD]` |
| 指令文本 | 来自原对话 | 固定 `"Describe the changes between these two images."` |

**使用方式**：微调时直接读取 `datasets/LEVIR-CC/levircc_train.jsonl` 等文件即可。图像路径为绝对路径，需确保软链接 `datasets → /users/u2024311136/shared/shared_datasets` 已建立。两个数据集可合并到同一 JSONL 中混合训练（LoRA 对多图输入原生支持）。

> 如需从原始标注重新生成，可运行 `python scripts/preprocess_levircc.py`。

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
