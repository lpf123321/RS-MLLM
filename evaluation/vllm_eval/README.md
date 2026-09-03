# vLLM 离线评测器（RS-MLLM 报告配置）

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
| `schema.py` / `scoring.py` | 样例结构与计分（含 clean_correct 计分） |
| `vision_opd_tools.py` / `_v2` / `_v3` | 工具（sha256、percentile 等） |
| `vision_opd_profile.py` | 注册 vision_opd_9b profile |
| `prompts.py` / `sampling.py` / `caption_metrics.py` / `validate_run.py` | 提示词、采样、报告口径 Caption 指标、验证 |

## 运行方式

```bash
# 首次: 建环境(一次性)
bash evaluation/vllm_eval/setup_env.sh

# 每次评测(无需手动 export, 环境变量评测器自动设置):
cd evaluation/vllm_eval
.venv/bin/python vision_opd_vllm_eval.py \
  --manifest <testset.jsonl> \
  --model <model_dir> \
  --model-profile <key> \
  --min-pixels 200704 --max-pixels 2097152 \
  --batch-size 128
```

- `--model-profile` 取 `model_policy.py` 中可信 profile（如 `mmerestore_bf16`），
  或 `--derived-profile /path/to/profile.json`（二选一，不能同时省略）；
- `--output-dir` 省略时默认仓库根下的 `results/<manifest文件名>_<profile>`；
  相对输出目录也按仓库根解析，不随启动时的 cwd 改变；
- 三 pass 策略（token 倍数 1x/2x/4x），输出 `prediction_attempts.jsonl` + `run_config.json`。
- 评测器自包含: VLLM_USE_FLASHINFER_SAMPLER/LD_LIBRARY_PATH/WORKER_MULTIPROC 自动设置。
- 统一路由入口使用 `batch_size=128`（request window）与 `max_num_seqs=64`（GPU
  并发上限）；每个 window 以有界 4-worker CPU 池解码 RGB 图像，保持原始顺序与像素
  策略不变。可用 `--image-load-workers` 显式覆盖 worker 数。
- `--enforce-eager` 仍是路由默认稳定配置。关闭 eager 可减少稳态 kernel launch
  开销，但 vLLM 0.26 的首次编译/graph capture 可能超过 3 分钟，不能把冷启动时间
  当作稳态吞吐。

协议例外：XLRS Caption 使用训练时九宫格长提示和 550-token 长度约束；XLRS
Grounding 为复现 Exp5 的实际运行条件，使用 4096 导出、16,777,216 像素上限、
32K 上下文和显式 `thinking -> response` assistant 前缀。历史脚本中声称的
1,048,576 上限写到了 processor 的无效属性，不能代表 31.97%结果的真实条件。

## 运行环境（vllm 0.26 + torch 2.11 cu129）

评测器依赖 **vLLM 0.26**，用本项目独立环境（不与 rs-mllm 主项目共用）。
评测数据清单的尺寸探测也必须使用该环境：Pillow 是显式依赖，清单构建器在
Pillow 缺失、图片路径不存在或尺寸无效时会直接失败，不会生成或覆盖不安全的清单。
**标准可复现**（uv 项目 + uv.lock 精确锁定，已验证干净环境加载 Qwen3.5 成功）：

```bash
# 一键: bash evaluation/vllm_eval/setup_env.sh
cd evaluation/vllm_eval && uv sync --locked   # 依 uv.lock 还原(vllm 0.26 + torch 2.11 cu129)
```

## 数据/模型

- 清单：`--manifest` 指向 testset.jsonl（pilot_500 / 950 / full 等）；
- 通过 `./rsmllm.sh` 评测时，程序会在启动 vLLM 前验证基础清单和子任务清单；
  若发现旧清单含 `0x0` 尺寸或失效路径，会先重建并在构建失败时阻止评测；
- 模型：预训练权重/量化导出目录（bf16、W8A8、GPTQ）。
