# 模型量化（报告 §6.2）

## 方法对比（固定清单实测，详见报告 §6.2）

| 方法 | 类型 | 精度变化 | 部署体积 | 时延变化 |
|---|---|---|---|---|
| W8A8-INT8（PTQ，LLM Compressor） | 离线 | +1.00 pp（无统计学意义） | 5.54 GB（-39.2%） | -13.1%（未达 15% 门槛，辅助参考） |
| W4A16-GPTQ（PTQ，LLM Compressor） | 离线 | +1.75 pp（无统计学意义） | 3.81 GB（-58.2%） | -7.6% |
| INT8（bitsandbytes 运行时） | 加载时 | CI ±1.19 pp（仅显存候选） | ≈4.6 GB | +6.7%（更慢） |
| NF4（bitsandbytes 运行时） | 加载时 | CI [-2.03, +1.86]（有统计学意义下降，排除） | ≈2.3 GB | 未实测 |

- **部署默认：W8A8-INT8**（时延收益最大、无精度退化、5.54 GB 落入 8 GB 预算）；**W4A16-GPTQ 为存储更小备选**。
- 评测口径：量化对比子集（770）+ 运行时配对子集（590，A-B-B-A 交替），`clean_correct` 官方计分；详见报告附录「评测校验条件」。
- 数字来源均为报告 §6.2 固定清单实测——**不应用其他来源数字替换**。

## 转换（vLLM 0.26 已验证支持 Qwen3.5 架构）
```bash
# 仓库入口（rsmllm 菜单 [4] 或直接调用）：
python -m rsmllm.quantize --method w8a8-int8 --model mmerestore_bf16 \
  --calibration calibration_512.jsonl --output quantized_models/out

# 底层 LLM Compressor 转换脚本：
python scripts/quantize_qwen35_vlm.py \
  --method w8a8-int8            # 或 w4a16-gptq
  --model <bf16_model_dir> \
  --calibration <calib.jsonl> \
  --calibration-meta <meta.json> \
  --output <out_dir> [--workers 8]
```
> 依赖：LLM Compressor（转换）+ vLLM 0.26 官方 +cu129 wheel（`Qwen3_5ForConditionalGeneration` 已注册）。
> 运行时依赖链：torchcodec/ffmpeg symlink + `VLLM_USE_FLASHINFER_SAMPLER=0`（详见仓库 smoke 记录）。

## 评测
```bash
# 量化配对评测（报告 770/590 子集 A-B-B-A 交替）：
sbatch quantization/quant_pair_eval.sbatch

# 独立模型评测入口（evaluation.main，像素/口径与报告一致）：
python -m evaluation.main --model_path <quantized_model> --datasets vrsbench mme xlrs levircc \
  --eval_batch_size 64 --image_min_pixels 200704 --image_max_pixels 2097152 \
  --data_path_overrides '<json: 子集清单路径>' --output results/<name>.json
```
> 注意：`--data_root` 是**输出目录**（默认 `output/`），不是清单根；切换评测子集用 `--data_path_overrides`。

## 云托管（已上传 ModelScope）
- 8 个量化专家（general/ground/change/caption × w8a8-int8/w4a16-gptq）已上传：
  `Fun10165/rs-mllm-expert-{general,ground,change,caption}-{w8a8-int8,w4a16-gptq}`
- mmerestore 三件套（bf16 / w8a8-int8 / w4a16-gptq）亦已托管。
- 上传脚本：`scripts/upload_to_modelscope.py`（env `MODELSCOPE_API_TOKEN` 不落盘；`--dry-run` 预检）。

## 两点结论（实测）
1. **W8A8 时延收益最大**：A100 INT8 Tensor Core（真 INT8 GEMM 路径）实测 -13.1%；GPTQ/Marlin 为反量化+BF16 计算，主要收益在存储。
2. **bitsandbytes 运行时 INT8 +6.7%（更慢）**：`LLM.int8()` 混合分解（outlier 保留 FP16）+ 无融合核——工具实现限制，不代表 INT8 无收益。
