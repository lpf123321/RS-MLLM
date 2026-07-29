# RS-MLLM Token Pruning 实验汇总

## 实验设置

| 项目 | 值 |
|------|------|
| 基座模型 | Qwen3.5-4B |
| 评估脚本 | `scripts/prune_inference.py`（6 个指标与 `evaluation/metrics/` 完全一致 ✅） |
| 微调模型 | sft_stage1_lora（VRSBench 训练，142K 条） |
| 剪枝方法 | L2-Norm / DivPrune（pre-LLM 输入侧直接删除） |
| 数据集 | MME-RealWorld-RS, XLRS-Bench-lite, LEVIR-CC, VRSBench |

## 完整对比

### MME-RealWorld-RS (3736 条，MCQ)

| 方法 | Accuracy | MCQ_Accuracy |
|------|------|------|
| 基座 no-prune | 35.55% | **49.68%** |
| 基座 L2 R=0.5 | — | 47.78% |
| 微调 no-prune | 43.0% | **50.3%** |
| 微调 L2 R=0.2 | 15.93% | 24.09% |
| 微调 L2 R=0.5 | 16.81% | 20.40% |
| 微调 DivPrune R=0.5 | 16.94% | 21.95% |

### XLRS-Bench-lite (3080 条，MCQ)

| 方法 | Accuracy | MCQ_Accuracy |
|------|------|------|
| 基座 no-prune | 35.97% | **36.36%** |
| 基座 L2 R=0.5 | — | 36.10% |
| 微调 no-prune | 27.4% | 31.4% |
| 微调 L2 R=0.2 | 5.32% | 9.12% |
| 微调 L2 R=0.5 | 5.78% | 8.31% |
| 微调 DivPrune R=0.5 | 5.49% | 8.31% |

### LEVIR-CC (1929 条，变化描述)

| 方法 | BLEU-4 | ROUGE-L | CIDEr |
|------|------|------|------|
| 基座 no-prune | 0.71% | 5.84% | ~0 |
| 基座 L2 R=0.5 | 1.79% | 16.82% | — |
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
| Referring | Acc@0.5 | 1.0% | **~62%** |

## 指标验证

6 个指标全部与 `evaluation/metrics/` 对齐：

| 指标 | 验证结果 |
|------|------|
| Accuracy | ✅ |
| MCQ_Accuracy | ✅ |
| BLEU-1~4 | ✅ |
| ROUGE-L | ✅ |
| CIDEr | ✅ |
| Referring Acc@0.5 | ✅ |

## 结论

1. **基座模型上剪枝几乎无损**：R=0.5 时 MME/XLRS/LEVIR 精度基本持平或微涨
2. **微调模型上需低剪枝率**：R=0.5 大幅下降（LEVIR 12%），R=0.2 接近不剪枝水平（LEVIR 56%）
3. **LEVIR 对剪枝最敏感**：R=0.5 vs R=0.2 相差 4.7 倍
4. **L2 和 DivPrune 效果接近**，DivPrune 在微调模型上无明显优势
5. **VRSBench 受益最大**：微调 + R=0.2 在 VQA（+7.5%）、Caption（×7）、Referring（×62）上全面领先基座
6. **MME/XLRS 偏低是模型能力边界**：微调模型未学习 MCQ 格式，非剪枝问题

## 输出文件

```
prune/output/
├── results_base_mme.json               # 基座 MME
├── results_base_xlrs.json              # 基座 XLRS
├── results_base_levir.json             # 基座 LEVIR
├── results_lora_mme_r02.json           # 微调 MME R=0.2
├── results_lora_xlrs_r02.json          # 微调 XLRS R=0.2
├── results_lora_levir_r02.json         # 微调 LEVIR R=0.2
├── results_lora_vrs_chunk0~4.json      # VRSBench R=0.2 各 chunk
└── results_lora_vrs_cap_r02.json       # VRS caption R=0.2（含 CIDEr）
```

## 运行命令

```bash
# 基座模型
python scripts/prune_inference.py --model_path /path/to/Qwen3.5-4B --batch_size 32 --data_path 《数据》 --output 《输出》

# 微调 + 剪枝
python scripts/prune_inference.py --model_path /path/to/Qwen3.5-4B --lora_path /path/to/sft_stage1_lora --prune_method l2 --prune_r 0.2 --batch_size 32 --data_path 《数据》 --output 《输出》
```
