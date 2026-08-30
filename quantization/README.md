# 模型量化（报告 §6.2）

## 方法

| 方法 | 类型 | 精度变化 | 部署体积 | 时延变化 |
|---|---|---|---|---|
| W8A8-INT8（PTQ） | 离线（LLM Compressor） | +1.00 pp | 5.54 GB（-39.2%）| **-13.1%** |
| W4A16-GPTQ（PTQ） | 离线（LLM Compressor） | +1.75 pp | 3.81 GB（-58.2%）| -7.6% |
| INT8（bitsandbytes 运行时） | 加载时 | CI ±1.19 pp（非劣效通过） | ≈4.6 GB | +6.7% |
| NF4（bitsandbytes 运行时） | 加载时 | CI [-2.03, +1.86]（未通过） | ≈2.3 GB | 未实测 |

- **部署默认：W8A8-INT8**（时延收益最大、无精度退化、5.54 GB 落入 8 GB 预算）；**W4A16-GPTQ 为存储更小备选**。
- 评测口径：770 条量化对比子集 + 590 条运行时配对子集（A-B-B-A 交替），`clean_correct` 官方计分。
- 依赖版本：LLM Compressor、bitsandbytes（报告附录「模型与版本」）。

## 转换（LLM Compressor CLI）

```bash
# W8A8-INT8（权重/激活 8 位，SmoothQuant 激活平滑）
llmcompressor.transform --model <bf16_model> --recipe recipes/w8a8_int8.yaml --output-dir <out>

# GPTQ W4A16（权重 4 位，激活 16 位）
llmcompressor.transform --model <bf16_model> --recipe recipes/gptq_w4a16.yaml --output-dir <out>
```

## 评测

```bash
python quantization/quant_pair_eval.py --model <quantized_model> --profile w8a8|gptq --outdir results/<name>
# 批量评测入口见 slurm_scripts/quant_eval_950.slurm
```

## 两点结论（实测）

1. **W8A8 时延收益最大**：A100 INT8 Tensor Core（真 INT8 GEMM 路径）实测 -13.1%；GPTQ/Marlin 为反量化+BF16 计算，主要收益在存储。
2. **bitsandbytes 运行时 INT8 +6.7%（更慢）**：`LLM.int8()` 混合分解（outlier 保留 FP16）+无融合核——是工具实现限制，不代表 INT8 无收益。
