# General / Grounding Expert LoRA 续训练

本目录保存 General 与 Grounding 两条续训练链路。训练使用 Qwen3.5-VL 原生模型、
answer-only loss，并固定关闭 thinking（`enable_thinking=False`）。

## 统一目录

```text
datasets/
├── MME-RealWorld-RS/                 # 上游原数据，可为软链
├── VRSBench/
├── XLRS-Bench-lite/
├── XLRS-Bench_visual_grounding_en/
└── training35/                       # 我们改编后的训练数据
    ├── expert_general_lora.json
    ├── expert_ground_lora_bootstrap.json
    ├── expert_ground_lora.json
    ├── assets/                       # 五阶段内容哈希图片
    └── images/                       # General/Grounding 内容哈希图片

models/
├── Qwen3.5-4B
├── expert_general
├── expert_ground
├── expert_change
├── expert_caption
├── expert_general_lora
├── expert_ground_lora
├── expert_general_full
├── expert_ground_full
└── training35/
    ├── deltas/                       # expert_general.pt 等 raw delta
    └── runs/<run-id>/                # 本次训练 LoRA 与 merged model
```

`expert_general` 和 `expert_ground` 是 base + full-rank delta，不含 LoRA；
`*_lora` 是 PEFT adapter；`*_full` 是 base + delta + LoRA。

数据 staging 只建立经过校验的软链，不复制原图或大 JSON：

```bash
python scripts/stage_training35_data.py
python scripts/preflight_training35.py
```

## 训练

General 最终 adapter 固定命名为 `expert_general_lora`：

```bash
ARTIFACT_ROOT="$PWD/models/training35/runs/training35-short"
bash training/train_general_expert.sh \
  --gpus 1 --max-updates 30 --output-root "$ARTIFACT_ROOT"
```

Grounding 先执行 942 条 bootstrap，再续训最终 adapter。bootstrap 仅是内部初始化，
最终对外名称仍为 `expert_ground_lora`：

```bash
bash training/train_grounding_expert.sh \
  --stage bootstrap --gpus 1 --output-root "$ARTIFACT_ROOT"

bash training/train_grounding_expert.sh \
  --stage final --gpus 1 --max-updates 30 \
  --init-lora "$ARTIFACT_ROOT/bootstrap/expert_ground_lora" \
  --output-root "$ARTIFACT_ROOT"
```

单卡会自动调整 gradient accumulation，保持有效 batch size 32。完整训练使用相同
入口、去掉 `--max-updates` 并使用下表卡数：

| 输出名称 | 起点 | GPU | updates |
|---|---|---:|---:|
| `expert_general_lora` | `expert_general` | 4 | 484 |
| bootstrap（内部步骤） | `expert_ground` | 4 | 30 |
| `expert_ground_lora` | bootstrap adapter | 8 | 1370 |

三组均为 1 epoch、LoRA rank 32、alpha 64、dropout 0.05、lr 1e-4、
seed 20260823、最大 3,211,264 pixels。

## 数据来源

改编 JSON 发布于 ModelScope `Uchitachi/RS-MLLM-Distillation-Data`，图片不重复发布。
`IMAGE_INDEX.jsonl` 记录上游来源、字节数和 SHA-256；官方图片下载脚本固定 revision：

```bash
bash training/distillation/expert_lora/scripts/download_official_datasets.sh
```

General 数据为 MME 3,736 + XLRS 3,080（loss weight 2）以及 VRSBench VQA
8,664（loss weight 1），共 15,480 条。Grounding bootstrap 为 942 条，最终
Grounding 数据为 43,838 条。
