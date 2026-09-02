# General / Grounding Expert LoRA 续训练

本目录保存报告 5.2 中实际保留的 General Exp7 与 Grounding bootstrap/Exp5
训练实现。两条链路均使用 Qwen3.5-VL 原生模型、answer-only loss，并显式设置
`enable_thinking=False`。入口会先校验数据 schema、记录数、相对图片路径和图片存在性，
拒绝覆盖非空输出目录。

## 标准运行时布局

训练只从仓库根目录的标准名称解析起点：

```text
models/
├── Qwen3.5-4B
├── expert_general
├── expert_ground
├── expert_change
├── expert_caption
├── expert_general_lora
├── expert_ground_lora
├── expert_general_full
└── expert_ground_full
```

`expert_general` 与 `expert_ground` 是 `Qwen3.5-4B + full-rank delta`，
不包含本目录续训练产生的 PEFT LoRA。运行仓库根目录的 staging 工具可在 ignored
cache 中生成这两个标准 HF 模型并建立软链：

```bash
source scripts/training/env.sh
activate_conda
python scripts/stage_training35_models.py --build-experts
```

staging 幂等执行，拒绝替换冲突路径，并在
`.models/training35/staging_manifest.json` 记录所有权重分片的大小和 SHA-256。

## 数据

已发布的便携数据仓库为
`Uchitachi/RS-MLLM-Distillation-Data`。其中 General Exp7 包含：

- MME 3,736 + XLRS 3,080，loss weight 2；
- VRSBench VQA 8,664，loss weight 1；
- 合计 15,480 条。

源历史 JSON SHA-256 为
`8711391dec337dd5117602eb122b9bdcf8994831458202979d699662b47104cb`；
便携 JSON SHA-256 为
`3f599530b7431cd2280410a5c8590a4cb90241e828206956b2950a5fcaaa1153`。
Grounding bootstrap 为 942 条；Exp5 为 VRS 37,227 + XLRS 6,611，共 43,838 条。

数据仓库不重复发布上游图片。先下载 JSON/index，再用官方数据根目录按
`IMAGE_INDEX.jsonl` 的字节数和 SHA-256 物化内容寻址图片。Pillow 必须使用
`requirements.txt` 固定的 12.3.0，避免 JPEG 重放漂移。

## 可运行入口

所有命令均从仓库根目录执行。`--gpus` 表示 GPU 数量；单卡会自动把 gradient
accumulation 调整到 32，保持历史有效 batch size 32。

General Exp7 30-update 验收：

```bash
RUN_ID=general-exp7-short
bash training/train_general_expert.sh \
  --config general_exp7 \
  --gpus 1 \
  --max-updates 30 \
  --output-root "outputs/training35/$RUN_ID"
```

Grounding 必须先生成新的 bootstrap LoRA，再把它显式传给 Exp5：

```bash
RUN_ID=grounding-short
OUT="outputs/training35/$RUN_ID"

bash training/train_grounding_expert.sh \
  --config grounding_bootstrap_942 \
  --gpus 1 \
  --output-root "$OUT"

bash training/train_grounding_expert.sh \
  --config grounding_exp5 \
  --gpus 1 \
  --init-lora "$OUT/grounding_bootstrap_942" \
  --max-updates 30 \
  --output-root "$OUT"
```

`training_metrics.json` 记录实际 world size、有效 batch、更新数、数据 SHA、起始
adapter SHA、输出 adapter SHA、平均 loss 和运行时间。

## 完整历史配置

| 配置 | 起点 | 卡数 | 累积 | 更新数 | 历史结果 |
|---|---|---:|---:|---:|---|
| `general_exp7` | `expert_general` | 4 | 8 | 484 | MME 0.702623；XLRS 0.490584；VRS VQA 0.705713 |
| `grounding_bootstrap_942` | `expert_ground` | 4 | 8 | 30 | VRS Acc@0.5 0.690204 |
| `grounding_exp5` | 新 bootstrap LoRA | 8 | 4 | 1370 | VRS Acc@0.5 0.771644；XLRS Acc@0.5 0.319651 |

三组均为 1 epoch、LoRA rank 32、alpha 64、dropout 0.05、lr 1e-4、
seed 20260823、最大 3,211,264 pixels。完整训练只需使用上面的入口和对应卡数，
去掉 `--max-updates`。

## CPU 检查

```bash
PYTHONPATH=training/distillation/expert_lora/src \
  python -m unittest discover \
  -s training/distillation/expert_lora/tests -v

python -m compileall -q \
  training/distillation/expert_lora/src \
  training/distillation/expert_lora/scripts
```

这些检查不代替 GPU 验收；GPU 产物还必须通过 PEFT 重新加载和固定 ID 趋势评测。
