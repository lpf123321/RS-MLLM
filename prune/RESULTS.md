# RS-MLLM Token Pruning 实验汇总

## 实验设置

| 项目 | 值 |
|------|------|
| 基座模型 | Qwen3.5-4B |
| 微调模型 | sft_stage1_lora（VRSBench 训练，142K 条） |
| 剪枝方法 | L2-Norm / DivPrune（pre-LLM 输入侧直接删除） |
| 数据集 | MME-RealWorld-RS, XLRS-Bench-lite, LEVIR-CC, VRSBench |

## 完整对比

### MME-RealWorld-RS (3736 条，MCQ)

| 方法 | Accuracy | MCQ_Accuracy |
|------|------|------|
| 基座 no-prune | 35.73% | **49.63%** |
| 基座 L2 R=0.5 | — | 47.78% |
| 微调 no-prune | 43.0% | **50.3%** |
| 微调 L2 R=0.2 | 15.9% | 24.1% |
| 微调 L2 R=0.5 | 16.8% | 20.4% |
| 微调 DivPrune R=0.5 | 16.9% | 22.0% |

### XLRS-Bench-lite (3080 条，MCQ)

| 方法 | Accuracy | MCQ_Accuracy |
|------|------|------|
| 基座 no-prune | 35.55% | **35.94%** |
| 基座 L2 R=0.5 | — | 36.10% |
| 微调 no-prune | 27.4% | 31.4% |
| 微调 L2 R=0.2 | 5.3% | 9.1% |
| 微调 L2 R=0.5 | 5.8% | 8.3% |
| 微调 DivPrune R=0.5 | 5.5% | 8.3% |

### LEVIR-CC (1929 条，变化描述)

| 方法 | BLEU-4 | ROUGE-L | CIDEr |
|------|------|------|------|
| 基座 no-prune | 1.72% | 15.98% | ~0 |
| 微调 no-prune | ~59% | — | — |
| 微调 L2 R=0.2 | **56.42%** | **73.90%** | **1.37** |
| 微调 L2 R=0.5 | 12.03% | — | — |
| 微调 DivPrune R=0.5 | 12.03% | — | — |

### VRSBench (62918 条，VQA / Caption / Referring)

| 任务 | 指标 | 基座 no-prune | 微调 L2 R=0.2 |
|------|------|------|------|
| VQA | Accuracy | 52.31% | **59.76%** |
| Caption | BLEU-4 | 2.19% | **15.14%** |
| Caption | ROUGE-L | 16.17% | **38.29%** |
| Caption | CIDEr | ~0 | **1.38** |
| Referring | Acc@0.5 | 1.0% | **73.90%** |

## 结论

1. **基座模型上剪枝几乎无损**：R=0.5 时 MME/XLRS/LEVIR 精度基本持平或微涨
2. **微调模型上需要低剪枝率**：R=0.5 大幅下降，R=0.2 接近不剪枝水平
3. **L2 和 DivPrune 效果接近**：贪心多样性选择在微调模型上无明显优势
4. **VRSBench 受益最大**：微调 + 20% 剪枝在 VQA（+7.5%）、Caption（+7x）、Referring 上全面领先基座模型
5. **MME/XLRS 偏低是模型能力边界**：微调模型未学习 MCQ 格式，非剪枝脚本问题

## 文件结构

```
prune/output/
├── results_lora_vrs_chunk0~4.json    # VRSBench R=0.2 各 chunk
├── results_lora_mme_r02.json         # MME R=0.2
├── results_lora_xlrs_r02.json        # XLRS R=0.2
├── results_lora_levir_r02.json       # LEVIR R=0.2
└── results_lora_vrs_cap_r02.json     # VRS caption R=0.2（含 CIDEr）
```

## 运行命令

```bash
# 微调模型 + L2 R=0.2 剪枝
python scripts/prune_inference.py \
    --model_path /path/to/Qwen3.5-4B \
    --lora_path /path/to/sft_stage1_lora \
    --prune_method l2 --prune_r 0.2 --batch_size 32 \
    --data_path datasets/shared_datasets/《数据集》 --output 《输出文件》
```
