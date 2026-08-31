# vLLM 离线评测器（RS-MLLM 报告口径）

本目录是 RS-MLLM 四数据集评测使用的 **vLLM 批量评测器**（拆包为目录脚本，
无包结构、`python xxx.py` 直接运行）。

## 包含模块

| 文件 | 作用 |
|---|---|
| `vision_opd_vllm_eval.py` | **主评测器**（vLLM 离线批量推理 + 官方计分） |
| `vision_opd_eval.py` | 串行 Transformers 评测器（同一协议/计分） |
| `model_policy.py` | 可信模型 profile（sha256/required_files 校验） |
| `model.py` | Qwen 适配器 |
| `run_eval.py` | 评测协议（max_new_tokens 派生） |
| `schema.py` / `scoring.py` | 样例结构与计分（含 clean_correct 口径） |
| `vision_opd_tools.py` / `_v2` / `_v3` | 工具（sha256、percentile 等） |
| `vision_opd_profile.py` | 注册 vision_opd_9b profile |
| `prompts.py` / `sampling.py` / `caption_metrics.py` / `validate_run.py` | 提示词、采样、caption 代理指标、验证 |

## 运行方式（标准）

```bash
cd evaluation/vllm_eval
python vision_opd_vllm_eval.py \
  --manifest <testset.jsonl> \
  --model <model_dir> \
  --model-profile <key>   # 或 --derived-profile <manifest.json> \
  --output-dir <outdir> \
  --min-pixels 200704 --max-pixels 2097152 \
  --batch-size 128
```

- `--model-profile` 取 `model_policy.py` 中可信 profile（如 `mmerestore_bf16`）；
- 新模型用 `--derived-profile`（结构见 `docs/EXPERIMENT_MAP.md` ⑧）；
- 三 pass 策略（token 倍数 1x/2x/4x），输出 `prediction_attempts.jsonl` + `run_config.json`。

## 运行环境（vllm 0.26 + torch 2.11 cu129）

评测器依赖 **vLLM 0.26**，用本项目独立环境（不与 rs-mllm 主项目共用）。
**标准可复现**（uv 项目 + uv.lock 精确锁定，已验证干净环境加载 Qwen3.5 成功）：

```bash
# 一键: bash evaluation/vllm_eval/setup_env.sh
cd evaluation/vllm_eval
uv sync --locked          # 依 uv.lock 精确还原(vllm 0.26 + torch 2.11 cu129 + torchcodec 0.15 cu129)
```

运行前置（Qwen3.5 加载）：
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0          # flashinfer 0.6.14 与 nvcc12.4 不兼容
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

> **注意**：`VLLM_USE_FLASHINFER_SAMPLER=0` 为必需（否则 flashinfer 0.6.14 编译失败）。
> 图片评测**不需要** `LD_LIBRARY_PATH`（uv.lock 已含 torchcodec cu129 + nvidia libs）。
> vllm wheel 由 `setup_env.sh` 下载；GitHub 慢时可用 ghproxy 加速。

## 数据/模型

- 清单：`--manifest` 指向 testset.jsonl（pilot_500 / 950 / full 等）；
- 模型：预训练权重/量化导出目录（bf16、W8A8、GPTQ）。
