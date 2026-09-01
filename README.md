
## 目录

- [一、基本信息](#一基本信息)
  - [1.1 摘要](#11-摘要)
  - [1.2 主要工作](#12-主要工作)
  - [1.3 项目分工](#13-项目分工)
- [二、项目概述](#二项目概述)
  - [2.1 整体架构图](#21-整体架构图)
- [三、快速开始](#三快速开始)
  - [3.1 硬件要求](#31-硬件要求)
  - [3.2 环境配置](#32-环境配置)
  - [3.3 模型推理](#33-模型推理)
  - [3.4 模型评测](#34-模型评测)
  - [3.5 模型训练](#35-模型训练)
- [四、测试结果](#四测试结果)
- [五、目录索引](#五目录索引)
- [六、Acknowledgment](#六acknowledgment)

---

## 一、基本信息

### 1.1 摘要

随着多模态大模型（MLLMs）的变革性发展与太空智算科技的兴起，运用其强大的理解能力来克服卫星图像处理中面临的高分辨率解析困难与泛化能力弱等挑战已成为当前的重要课题。然而，现有的通用大模型难以直接适应超高分辨率的遥感图像，且卫星平台严苛的算力与内存瓶颈进一步限制了其端侧部署。

为突破通用模型能力的限制，本团队针对星载任务的难度大、资源极度受限的特点，设计了一套高效的、出色的多模态大模型轻量化部署方案。整个方案按序完成了数据集处理——基座模型选型——模型训练——模型轻量化——容错与故障恢复机制的设计链条，成功基于轻量的模型达到了优秀的表现效果。在基座选型上，本方案对比了目前先进的Qwen、LLaVA、InternVL通用多模态大模型系列及部分遥感专家模型，充分比较了模型的表现与实际应用场景，最终选定 **Qwen3.5-4B** 作为方案的模型基座。鉴于所有的通用模型基座在赛题数据集，尤其是高分辨率的两个数据集上的表现存在如输出格式错误和指令遵循能力差的明显缺陷，本方案深入调研了当前有效的模型训练方法，并引入了本团队先前在ICML26中提出的CVSearch框架。基于四个指定的竞赛数据集，本方案协同应用了多专家有监督微调与在线策略蒸馏的训练方法，显著增强了模型对复杂遥感任务的适应性，使模型基座的性能完成了质变。接下来，为了丰富训练数据，团队利用GPT模型生成了数千条额外数据，并依此进行深化的微调工作，进一步优化了模型在任务中的表现效果。

在模型轻量化的方面，我们结合了视觉Token剪枝与模型量化策略。我们通过消融实验发现，不同的数据集和任务类型对于视觉Token剪枝的敏感性和方法要求是有显著差异的。基于这个发现，我们基于免训练的视觉Token剪枝方法，设计了任务适应的Token剪枝路由机制。同时，我们使用了$\mathrm{W8A8}$/$\mathrm{W4A16}$ 训练后量化方法。该方法大幅减少了计算开销，将压缩存储开销控制在8G以内。随后，为了增强模型在太空辐射环境下的鲁棒性，我们实现了针对单粒子翻转的容错机制，保证了卫星侧推理的稳定性。实验表明，本方案在满足星载端侧严苛资源约束的同时，仍保持了优异的遥感理解与推理性能。

### 1.2 主要工作

1. 我们充分调研了赛题的具体需求、提供的数据集和当前的遥感多模态大模型工作成果，构建了从基座选型到星载端侧部署的完整技术链路；
2. 我们创新性地提出了一套多阶段协同训练方案。结合多专家有监督微调（SFT）、在线策略蒸馏（OPD）以及 CVSearch 框架，并通过大模型合成数据深化微调，我们成功大幅跨越了通用大模型在复杂高分辨遥感任务上的性能鸿沟；
3. 我们设计了任务适应的高效模型轻量化策略。我们提出了一种免训练的视觉 Token 剪枝路由机制，在精准削减冗余 视觉 Token 的同时保留了核心视觉特征，并配合 W8A8/W4A16 训练后量化，实现与星载硬件资源环境的适配；
4. 最终，我们实现了高可靠的星载容错机制。针对太空辐射环境，我们设计了应对单粒子翻转的底层容错与故障恢复策略，保障了极端严苛资源约束下的系统推理稳定性。

### 1.3 项目分工
<!-- 表格容器 -->
<div style="overflow-x: auto;">
  <table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; min-width: 1000px; table-layout: fixed; width: 100%;">
    <thead>
      <tr>
        <th style="width: 120px; white-space: nowrap;">姓名</th>
        <th style="width: 260px;">负责模块</th>
        <th>工作内容</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>
          <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>赖培丰　</strong></div>
        </td>
        <td>...</td>
        <td>
1. ...<br>
2. ...
        </td>
      </tr>
      <tr>
        <td>
          <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>曹鹏宇</strong></div>
        </td>
        <td>...</td>
        <td>
1. ...<br>
2. ...
        </td>
      </tr>
      <tr>
        <td>
          <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>王翰　</strong></div>
        </td>
        <td>...</td>
        <td>
1. ...<br>
2. ...
        </td>
      </tr>
      <tr>
        <td>
          <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>曹忠博　</strong></div>
        </td>
        <td>...</td>
        <td>
1. ...<br>
2. ...
        </td>
      </tr>
      <tr>
        <td>
          <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>孙昊　</strong></div>
        </td>
        <td>...</td>
        <td>
1. ...<br>
2. ...
        </td>
      </tr>
      <tr>
        <td>
          <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"><strong>李刘鹏　</strong></div>
        </td>
        <td>...</td>
        <td>
1. ...<br>
2. ...
        </td>
      </tr>
    </tbody>
  </table>
</div>

---

## 二、项目概述

### 2.1 整体架构图

![overall](assets/overall.png)

---

## 三、 快速开始

### 3.1 硬件要求

- **显卡**：NVIDIA GPU，显存 ≥ 12 GB，CUDA 计算能力 ≥ 8.0（Ampere 级及更高）
- **CPU**：建议 x86_64
- **内存**：建议 ≥ 32 GB
- **存储**：建议 ≥ 100 GB
- 系统：建议 Ubuntu 22.04

> **备注**：实际资源消耗量远低于此规格。本项目未在其他硬件配置上运行过，不保证完全兼容。

### 3.2 环境配置

**方式一：uv （推荐）**

```bash
uv sync --locked            # 按 uv.lock 精确安装 121 个依赖（CUDA 12.8 + torch 2.8.0）
source .venv/bin/activate
```

**方式二：Docker 容器（推荐**

```bash
docker build -t rs-mllm .
docker run --gpus all -it -v $(pwd):/workspace rs-mllm bash
```

**方式三：一键配置（`setup.sh` 自动识别 uv/conda 并自动进入环境）**

```bash
bash setup.sh                                              # 训练/推理环境(torch 2.8 cu128)
bash evaluation/vllm_eval/setup_env.sh                     # vLLM 评测环境(vllm 0.26 cu129)
```

> 评测器依赖 vllm 0.26，要用 `evaluation/vllm_eval` 的评测环境，不能用 3.2 的训练环境。
>
> 首次安装需下载约 4GB（torch/nvidia/vllm wheel）。若下载卡住无进度，
> 设置代理后重跑：`export HTTPS_PROXY=http://<代理>:<端口> HTTP_PROXY=http://<代理>:<端口>`。

### 3.3 模型推理

**推荐：交互式控制台（问答式选择任务，模型首次使用自动从 ModelScope 拉取）**

```bash
./bin/rsmllm            # 进入控制台（或 python -m rsmllm.console）
```

菜单：`[2] 推理服务` → 选模型别名 → 自动下载并启动推理。

**非交互（脚本）**：

```bash
# 推理服务需要 vLLM，用评测环境运行：
cd evaluation/vllm_eval
PYTHONPATH=<仓库根> .venv/bin/python -m rsmllm.serve --model w8a8 --port 8001
```

支持的模型别名见 `rsmllm/config.py` 的 `MODEL_REGISTRY`（如 `base`、`mmerestore_bf16`、
`w8a8`、`gptq`、`expert_general`、`expert_general_w8a8` 等 15 个），
或直接给 ModelScope id / 本地路径。模型按需下载缓存在 `.models/`（
`RSMLLM_MODEL_CACHE` 可覆盖），离线时 `MODELSCOPE_OFFLINE=1` 强制本地命中。

### 3.4 模型评测

评测走 **vLLM 0.26 离线批量推理**（与报告一致：greedy + bf16 + 像素 200,704–2,097,152 + max_len 16,384），
统一入口（首次建环境，之后直接评测，无需手动 export）：

```bash
# 1) 一次性建环境(vllm 0.26 + torch 2.11 cu129, 独立于训练环境):
bash evaluation/vllm_eval/setup_env.sh

# 2) vLLM 评测器(环境变量自动设置, 三pass + clean_correct 计分):
cd evaluation/vllm_eval
.venv/bin/python vision_opd_vllm_eval.py \
  --manifest /path/to/testset.jsonl \
  --model /path/to/model_dir \
  --model-profile mmerestore_bf16 \
  --min-pixels 200704 --max-pixels 2097152 --batch-size 128

```

> 说明：`--model-profile` 选 `model_policy.py` 中的可信 profile（如 `mmerestore_bf16`），
> 或改用 `--derived-profile /path/to/profile.json`（二选一）。
> `--output-dir` 省略时默认为 `results/<manifest文件名>_<profile>`。
> 评分结果自动写入输出目录：`clean_summary.json` / `official_summary.json` / `unit_scores.jsonl`。

**转录评测（Transformers 路径，不依赖 vLLM）**：

```bash
./bin/rsmllm          # 菜单 [1] 评测，交互选择 子集(950/770/590/400/full) 与数据集
# 或脚本直接调用:
python -m evaluation.main --model_path w8a8 --datasets vrsbench mme xlrs levircc \
  --eval_batch_size 64 --image_min_pixels 200704 --image_max_pixels 2097152 \
  --data_path_overrides '{"vrsbench":"/path/950/vrsbench_eval.jsonl","mme":"/path/950/mme_rs.jsonl","xlrs":"/path/950/xlrs.jsonl","levircc":"/path/950/levircc_test.jsonl"}' \
  --output results/950.json
```

评测子集（`950` 量化部署 / `770` 量化对比 / `590` 运行时配对 / `400` 离散精度）
通过 `--data_path_overrides` 把清单路径改到对应子集文件实现（`--data_root` 是
结果的输出目录，默认 `output/`），与报告附录的说明一致；
像素上下限与报告评测设置一致（200,704 - 2,097,152）。

**量化转换**（与报告同链路）：

```bash
python -m rsmllm.quantize --method w8a8-int8 --model mmerestore_bf16 \
  --calibration calibration_512.jsonl --output quantized_models/out
```

**容错探针 / TTFT / 并发扫描 / 剪枝**（实验级入口）：

```bash
python scripts/quant_tol_probe.py \
  --model bf16 --model-dir /path/to/bf16 \
  --model w8a8 --model-dir /path/to/w8a8 \
  --model gptq --model-dir /path/to/gptq \
  --out-dir results/quant_tol_probe     # 容错探针: 位翻转 + 哈希检测率 + 输出漂移
python scripts/ttft_serve_probe.py      # vLLM serve /metrics TTFT
python scripts/run_batch_scan.py        # 6 档并发吞吐/时延扫描
python evaluation/run_prune_sweep.py    # Token 剪枝方法扫描
```

### 3.5 模型训练

模型训练采用 **统一 SFT 主干 + 多专家（四专家）LoRA 微调** 的流程，全部训练脚本位于
`scripts/training/`，并可通过 `scripts/train.sh` 一键启动。

**训练流程一览**：

| step | 内容 | 脚本 | 说明 |
|------|------|------|------|
| 1 | 统一 SFT 主干 | `scripts/train.sh stage1_clean` | Qwen3.5-4B 起步（clean 数据），得到统一主干 |
| 2 | General 专家 | `scripts/train.sh expert_general` | from-base 微调 VQA/Caption/MCQ，得到 v2 |
| 3 | Grounding 专家续训 | `scripts/train.sh a1_grounding` | 统一主干续训 + XLRS 域对齐（2048） |
| 4 | Change 专家续训 | `scripts/train.sh a2b_change` | 统一主干续训 + 防遗忘 |
| 5 | Caption 专家 | `scripts/train.sh caption` | GA2 起点 + 双域（VRS 短/XLRS 长）数据 |
| 6 | 生成 Delta 权重 | `python scripts/gen_expert_deltas.py --verify` | 输出各专家相对基座的 `delta_model.pt` |

**一键启动示例**：

```bash
cd RS-MLLM

# 1) 查看某步将要提交的命令（不实际提交）
bash scripts/train.sh dry expert_general

# 2) 提交单个训练步骤（SLURM 集群）
bash scripts/train.sh stage1_clean      # 统一 SFT 主干
bash scripts/train.sh ga2_general       # General 续训（防遗忘）
bash scripts/train.sh a1_grounding      # Grounding 域对齐续训
bash scripts/train.sh a2b_change        # Change 防遗忘续训
bash scripts/train.sh caption           # Caption 双域训练

# 3) 提交完整流水线（顺序: stage1_clean → ga2 → a1 → a2b → caption）
bash scripts/train.sh all

# 4) 数据构建与 delta 生成
python scripts/build_caption_expert_data.py      # 构建 Caption 双域数据
python scripts/generate_mcq_data.py              # 合成 MCQ 样本
python scripts/gen_expert_deltas.py --verify     # 生成四专家 delta 并验证 W0+Δ==merged
```

**训练配置要点**：所有专家统一采用全参冻结的 LoRA（rank 32 / alpha 64 / dropout 0.05）、
lr=1e-4、1 epoch、bf16、DeepSpeed ZeRO-2；图像分辨率 262,144 ~ 1,048,576 像素
（Grounding 域对齐为 2048 分辨率）。训练数据的构建脚本（`split_expert_data.py`、
`build_caption_expert_data.py`）也在 `scripts/` 下。

> 注：训练数据 json/jsonl 体积较大，不随仓库分发（已 gitignore），
> 运行训练前请确保数据位于 `finetune_framework/VRSbench/`（可参照 `scripts/` 下构建脚本生成）。

---

## 四、测试结果

## 五、目录索引

## 六、Acknowledgment
