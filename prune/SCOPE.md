# SCOPE-L2 实验报告

## 方法

**SCOPE** (Saliency-Coverage Oriented Token Pruning) 用于 Qwen3.5-4B 视觉 token 剪枝。

| 方面 | 原版 SCOPE | SCOPE-L2 (本实现) |
|------|-----------|-----------------|
| 显著性来源 | CLS token 对 patch 的 attention 分数 | 特征向量 L2 范数 |
| 视觉编码器 | CLIP ViT | Qwen3VL ViT |
| 拦截层 | 倒数第二层 | Block 22 (depth=24) |
| 剪枝粒度 | 单个 patch token | 2×2 spatial merge block |
| 选取算法 | 贪心边际覆盖增益 × 显著性^alpha | 完全一致 |
| Alpha | 1.0 | 1.0 (可通过 ALPHA 环境变量调整) |
| 组合方式 | multi (gains × saliency) | multi (可通过 COMBINED 切换) |

## 实现

```
pixel_values → Qwen3VLVisionModel.forward()
    ├── Block 0..21 → 正常前向
    ├── Block 22    → forward hook 捕获 _scope_features [N_raw, 1024]
    ├── Block 23    → 正常前向
    └── Merger      → 2×2 merge → merged tokens [N_merged, 2560]

Qwen3_5Model.forward (patched)
    ├── get_image_features() → merged tokens
    ├── masked_scatter → inputs_embeds
    ├── _prune_scope():
    │   ├── 按 4 token = 1 block 做 mean pooling
    │   ├── SCOPE_L2 贪心选取 (1-R) 比例 block
    │   ├── 混合尺寸检测 → fallback (跳过剪枝)
    │   └── 删除未选中 visual token + padding 对齐
    └── self.language_model()
```

### 混合尺寸 Fallback

当同一 batch 内图片尺寸不一致时（如 1024×1024 和 812×1024），视觉编码器的 RoPE 位置编码存在跨图片差异，导致 layer 22 的 raw features 产生偏移。SCOPE 基于这些 features 选择 block，偏移会改变选择结果，产生垃圾输出。

**双层修复：**
1. `inference.py`：数据加载后按图片尺寸排序，同尺寸聚在同一 batch
2. `prune/scope.py`：边界 batch 检测到 `len(set(merged_per_image)) > 1` 时跳过剪枝

## 配置

- **模型**: Qwen3.5-4B + sft_stage1_lora
- **剪枝**: SCOPE, R=0.2 (保留 80% merged token)
- **推理**: scripts/inference.py, batch_size=32
- **指标**: prune/eval_metrics.py（修复版答案提取：支持 (X) 格式、A-E 选项）

## 结果

### MME (3,736 samples)

| 指标 | SCOPE R=0.2 | 对比 |
|------|-----------|------|
| **MCQ Accuracy** | **52.46%** | baseline 35.73%* (+16.7%) |
| 无字母率 | 0.1% | |
| 错误字母率 | 47.5% | |

分析：SCOPE 在 MME 上显著优于 baseline。LoRA 微调 + SCOPE 剪枝的组合有效。

### XLRS (3,080 samples)

| 指标 | SCOPE R=0.2 | 对比 |
|------|-----------|------|
| **MCQ Accuracy** | **31.81%** | baseline 35.55%* (-3.7%) |
| 无字母率 | 0.0% | |
| 错误字母率 | 68.2% | |

分析：SCOPE 在 XLRS 上略低于 baseline。XLRS 问题多要求细粒度空间推理（角落位置、小物体计数），block 级别的剪枝可能丢失关键空间信息。

### LEVIR-CC (1,929 samples)

| 指标 | SCOPE R=0.2 |
|------|-----------|
| BLEU-1 | 84.3% |
| BLEU-2 | 74.5% |
| BLEU-3 | 67.6% |
| **BLEU-4** | **62.3%** |
| ROUGE-L | 76.8% |
| CIDEr | 1.616 |

分析：LEVIR-CC 变化描述任务受 SCOPE 影响较小。双图场景下，SCOPE 对两张图各选 80% block，保留的变化信息足够生成高质量描述。

### VRSBench (62,918 samples, 5 chunks)

| Chunk | 进度 | 状态 |
|-------|------|------|
| 0 | offset=0 | 🔄 评估中 |
| 1-3 | offset=12600-50400 | ⏳ 排队 |

> VRSBench 含 VQA + Caption + Referring 三项子任务，每 chunk ~12,600 samples，预计 ~2h/chunk。

## 关键技术问题及解决

### 问题 1：答案提取正则不匹配

**现象：** ~50% 样本判定为"无法提取字母"
**原因：** MCQAccuracy 正则 `([A-D])\s*[.)]` 不支持 (X) 括号格式和 E 选项
**修复：** `prune/eval_metrics.py` 新增多层提取：(X) → X.) → 独立字母 → A-E

### 问题 2：B>1 批量推理产生垃圾

**现象：** batch_size>1 时，第一个 batch 正常，后续 batch 产生 "." "c" ">" 等输出
**原因：** 同一 batch 内图片尺寸不同（如 1024×1024 vs 812×1024），RoPE 位置编码偏移导致 SCOPE 选择错误
**修复：** 图片尺寸排序 + mixed-size fallback

### 问题 3：VRSBench system prompt 缺失

**现象：** VQA accuracy ~18%（远低于预期 52%）
**原因：** `prune_inference.py` 未加载 `Obey the task prefix...` system prompt
**修复：** 统一使用 `inference.py`（自带 SYSTEM_PROMPTS）

## 代码文件

```
prune/
├── scope.py              # SCOPE-L2 核心算法 (SCOPE_L2, _prune_scope)
├── eval_metrics.py       # 独立指标计算 (修复版 MCQ 答案提取)
├── __init__.py
├── README.md
├── SCOPE.md              # 本报告
└── output/               # 可视化 + 结果文件

scripts/
├── prune.py              # 剪枝集成 (第4种方法: scope)
├── inference.py          # 推理引擎 (图片尺寸排序)
├── viz_scope2.py         # 剪枝可视化

slurm_scripts/
├── run_scope_mme.sh
├── run_scope_xlrs.sh
├── run_scope_levircc.sh
├── run_scope_vrs0..4.sh
```

## 使用

```bash
# 全量评估 (4数据集)
sbatch slurm_scripts/run_scope_mme.sh
sbatch slurm_scripts/run_scope_xlrs.sh
sbatch slurm_scripts/run_scope_levircc.sh
sbatch slurm_scripts/run_scope_vrs0.sh

# 计算指标
python3 prune/eval_metrics.py --predictions outputs/inference_results/predictions_mme.json

# 调整 alpha
ALPHA=2.0 sbatch slurm_scripts/run_scope_mme.sh
```
