# General / Grounding Expert 有效 LoRA 实验

这里提交的是本项目实际产生保留 checkpoint 的 Qwen3.5-VL 原生 LoRA 训练路径，
不是只展示命令的空壳。它包含：

- Qwen3.5-4B 基座与 General/Grounding full-rank delta 的严格合并；
- answer-only loss、micro batch 1、LoRA 和 DDP 训练实现；
- General Exp3、Grounding bootstrap/Exp1/Exp4/Exp5 的冻结参数；
- ModelScope 下载、portable 数据路径校验和防覆盖检查。

## 保留实验

| 配置 | 数据 | 卡数 / 累积 | 报告中的结果 |
|---|---:|---:|---|
| `general_exp3` | MME 3,736 + XLRS 3,080 | 4 / 8 | MME 0.704497；XLRS 0.464610 |
| `grounding_bootstrap_942` | VRS-new 942 | 4 / 8 | VRS Acc@0.5 0.690204 |
| `grounding_exp1` | washed VRS 36,285 | 8 / 4 | VRS Acc@0.5 0.772573 |
| `grounding_exp4` | XLRS official 6,308 + new 303 | 8 / 4 | XLRS Acc@0.5 0.327734 |
| `grounding_exp5` | VRS 37,227 + XLRS 6,611 | 8 / 4 | VRS 0.771644；XLRS 0.319651 |

所有配置均为 1 epoch、有效 batch 32、LoRA rank 32、alpha 64、dropout 0.05、
lr 1e-4、seed 20260823、运行时图像上限 3,211,264 pixels。配置中的
`reported_eval` 是历史成功运行结果，不会被训练脚本伪造为新结果。

## 安装

```bash
cd training/distillation/expert_lora
python -m pip install -e .
```

保留环境版本为 PyTorch 2.10.0 + CUDA 12.8、Transformers 5.5.0、PEFT
0.18.0 和 qwen-vl-utils 0.0.14。GPU/CUDA 对应的 PyTorch wheel 应按机器环境安装。

## ModelScope 资产

发布完成后设置：

```bash
export RS_MLLM_DISTILLATION_DATASET_ID=YOUR_NAMESPACE/RS-MLLM-Distillation-Data
export RS_MLLM_EXPERT_MODEL_ID=YOUR_NAMESPACE/RS-MLLM-Expert-Checkpoints
```

运行入口会先调用 `ms-hub download`。当前数据仓库为私有，复现者需要先执行
`ms-hub login`。下载布局固定为：

```text
.artifacts/
├── base/Qwen3.5-4B/
├── datasets/
│   ├── ASSET_MANIFEST.json
│   ├── general/
│   ├── grounding/
│   ├── IMAGE_INDEX.jsonl
│   └── images/...  # 由官方仓库下载后按 SHA-256 物化
└── expert_models/
    ├── ASSET_MANIFEST.json
    ├── deltas/general/delta_model.pt
    ├── deltas/grounding/delta_model.pt
    └── adapters/grounding_bootstrap_942/...
```

Qwen3.5-4B 默认从官方 `Qwen/Qwen3.5-4B` 下载。ModelScope 数据仓库不重复分发
MME、VRSBench、XLRS 原图；`download_assets.sh` 会调用官方 Hugging Face 仓库并按
`IMAGE_INDEX.jsonl` 的 SHA-256 重建 `images/`。首次下载 MME 前需阅读其学术用途与
禁止再分发条款，并显式设置 `RS_MLLM_ACCEPT_MME_TERMS=1`。数据 JSON 中只允许相对
图像路径；本机 `/home/...`、`/data2/...` 路径会在训练前被拒绝。

## 运行

General：

```bash
bash scripts/run_experiment.sh general_exp3 --gpus 0,1,2,3
```

Grounding 三组配置都从相同的 942 bootstrap adapter 独立初始化。可以先本地训练
bootstrap，也可以直接下载已发布的 bootstrap：

```bash
bash scripts/run_experiment.sh grounding_bootstrap_942 --gpus 0,1,2,3
bash scripts/run_experiment.sh grounding_exp1 --gpus 0,1,2,3,4,5,6,7
bash scripts/run_experiment.sh grounding_exp4 --gpus 0,1,2,3,4,5,6,7
bash scripts/run_experiment.sh grounding_exp5 --gpus 0,1,2,3,4,5,6,7
```

脚本拒绝覆盖已有 adapter。若已经手动准备好 `.artifacts`，可设置
`RS_MLLM_SKIP_DOWNLOAD=1`。`--prepare-only` 只校验数据并生成 base+delta runtime
模型；`--dry-run` 只打印将执行的命令。

## 单独合并权重

只合并 expert delta：

```bash
rs-mllm-expert-compose \
  --base /path/to/Qwen3.5-4B \
  --delta /path/to/delta_model.pt \
  --output /path/to/general_expert_runtime
```

生成用于部署的 `base + delta + LoRA` 完整权重时，再追加
`--adapter /path/to/adapter`。合并器校验未知 key 和 shape mismatch，并写出
`composition_manifest.json`；不会覆盖非空目录。

## 数据路径规范化与检查

ModelScope staging 时可显式改写旧绝对路径：

```bash
rs-mllm-expert-data normalize \
  --input legacy.json \
  --output portable.json \
  --map /old/MME=images/mme \
  --map /old/XLRS=images/xlrs \
  --manifest portable.manifest.json

rs-mllm-expert-data validate \
  --data portable.json \
  --schema mcq \
  --image-root /path/to/downloaded/datasets \
  --expected-records 6816
```

## CPU 级检查

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src
```

这些检查不替代 GPU 训练；它们验证配置、schema、坐标语法、路径可移植性和
防覆盖约束。
