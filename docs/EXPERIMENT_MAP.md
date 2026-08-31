# 报告实验 ↔ 仓库脚本映射（复现链）

本文件将《技术报告》中提到的实验与仓库各目录脚本一一对应，并按复现顺序串成一条链。
每条记录含：报告位置、关键结论/数字、对应脚本、运行命令、状态。

## ① 数据准备与评测环境（报告 §3 实验条件与附录）

| 报告位置 | 内容 | 仓库对应 | 运行 |
|---|---|---|---|
| §3.2 数据集 | 四个基准清单 | `datasets_data/{vrsbench_eval,mme_rs,xlrs,levircc_test}.jsonl` | 数据预处理入口：`rsmllm` 菜单 [5] |
| 附录·评测校验条件 | 像素 200,704–2,097,152、greedy、批基准 | `rsmllm/config.py` `REPORT_CONF` | 该配置为报告口径唯一来源 |
| 附录·评测校验条件 | 清单子集切换 | `--data_path_overrides`（覆盖清单路径） | 见 `evaluation/main.py` 参数；`--data_root` 为**输出目录** |

## ② 训练（报告 §5 多专家 LoRA + OPSD）

| 报告位置 | 实验 | 仓库脚本 | 命令 |
|---|---|---|---|
| §5 多专家 LoRA | 统一 SFT 主干 + 四专家 | `scripts/train.sh` | `bash scripts/train.sh stage1_clean` → `expert_general` → `a1_grounding` → `a2b_change` → `caption` |
| §5 数据构建 | Caption 双域 / MCQ 合成 | `scripts/build_caption_expert_data.py`、`scripts/generate_mcq_data.py` | 见 `scripts/train.sh` 第 4 步 |
| §5 专家 delta | 生成 delta 并验证 | `scripts/gen_expert_deltas.py --verify` | `python scripts/gen_expert_deltas.py --verify` |
| §5.3 OPSD 蒸馏 | 在线策略蒸馏 | `training/distillation/` | 见 `training/distillation/README.md` |
| §5 参数 | rank 32 / alpha 64 / dropout 0.05 | `scripts/train.sh` 配置要点 | 与报告一致 |

## ③ 模型量化（报告 §6.2）

| 报告位置 | 实验 | 仓库脚本 | 命令 |
|---|---|---|---|
| §6.2 量化方法对比 | W8A8-INT8 / W4A16-GPTQ PTQ 转换 | `scripts/quantize_qwen35_vlm.py` | `python scripts/quantize_qwen35_vlm.py --method w8a8-int8 --model <bf16> --calibration <calib.jsonl> --calibration-meta <meta.json> --output <out>`（`--method` ∈ {w8a8-int8, w4a16-gptq}） |
| §6.2 量化配对评测 | A-B-B-A 交替、770/590 子集 | `quantization/quant_pair_eval.sbatch`（e0 配对路径）、评测器 `evaluation/main.py` | `sbatch quantization/quant_pair_eval.sbatch` |
| §6.2 结论 | W8A8: +1.00pp / 5.54GB(-39.2%) / -13.1%；GPTQ: +1.75pp / 3.81GB(-58.2%) / -7.6%；bitsandbytes INT8 +6.7%、NF4 未通过 | 表格 `tables/quant_compare.tex` | 数字来源为报告 §6.2 固定清单实测 |
| §6.2 部署 | 默认 W8A8-INT8 | `rsmllm` 菜单 [4] / `rsmllm/quantize.py` | `python -m rsmllm.quantize --method w8a8-int8 ...` |
| 云托管 | 8 个量化专家已上传 ModelScope | `scripts/upload_to_modelscope.py`（env `MODELSCOPE_API_TOKEN`） | `python scripts/upload_to_modelscope.py [--dry-run]` |

> 运行时量化（bitsandbytes INT8/NF4）为对照候选，未用于部署；结论见 §6.2"实测结论"。

## ④ 推理优化（报告 §6.3）

| 报告位置 | 实验 | 仓库脚本 | 命令 |
|---|---|---|---|
| §6.3 框架迁移 | 逐样本 → vLLM 批量引擎（约 150×） | 评测迁移：`evaluation/main.py`（批处理）+ slurm 侧 `vision_opd_vllm_eval.py` | 报告附录"评测校验条件"记录迁移前后对比 |
| §6.3 批处理扫描 | B=1/4/16/64/128/192 六档 | `scripts/run_batch_scan.py` | `python scripts/run_batch_scan.py` |
| §6.3 TTFT | vLLM server /metrics `time_to_first_token_seconds` | `scripts/ttft_serve_probe.py` | `SERVER_MODEL=<model_id> python scripts/ttft_serve_probe.py` |
| §6.3 推理配置 | batch 64 / max-num-seqs 64 / 16384 | `rsmllm/config.py` `REPORT_CONF` | 与全量评测一致 |

## ⑤ 视觉 Token 压缩（报告 §6.1）

| 报告位置 | 实验 | 仓库脚本 | 命令 |
|---|---|---|---|
| §6.1 Token 剪枝方法 | L2 / Scope / divprune / fourier / uniform / random | `token_compression/`（每方法一模块） | 见 `token_compression/README.md` |
| §6.1 剪枝扫描 | R-sweep 精度-时延 | `evaluation/run_prune_sweep.py` | `python evaluation/run_prune_sweep.py` |
| §6.1 router 剪枝 | 任务自适应路由 | `evaluation/run_delta_task_router.py`、`evaluation/run_delta_prune_eval.py` | 见各文件 `--help` |
| §6.1 结论 | 非结构化剪枝仅存储收益、不降计算 | `tables/prune_tradeoff.tex` | 报告 §6.3 精度-效率权衡 |

## ⑥ 星载容错（报告 §9）

| 报告位置 | 实验 | 仓库脚本 | 命令 |
|---|---|---|---|
| §9 权重位翻转 | bf16/W8A8/GPTQ 翻转敏感性（哈希检测率、输出漂移、失效-安全） | `scripts/quant_tol_probe.py` | `python scripts/quant_tol_probe.py`（固定种子 20260826、20 位翻转/模型） |
| §9 状态机回放 | 故障事件回放验证 | `tolerance/`（见 `tolerance/README.md`） | 状态机逻辑 + 任务级重放 |
| §9 结论 | 文件层 100% 拦截；张量故障约 2/3 超出逐次校验；量化模型翻转后失效-安全 | 报告 §9 测量 | 软件级结论，硬件 ECC 待实测 |

> 注：`scripts/quant_tol_probe.py` 的模型路径为开发机绝对路径（`/home/u2024311009/tiled_rag/...`），
> 迁移环境时同步替换 `BASE`/`MODELS`/`OUT` 变量。

## ⑦ 最终方案评测（报告 §3/§7）

| 报告位置 | 实验 | 仓库脚本 | 命令 |
|---|---|---|---|
| §3 最终方案性能 | 四专家路由全量评测（71,631） | `evaluation/main.py`（适配器 `router`）——评测入口；**vLLM 评测器见 `evaluation/vllm_eval/`**（拆包目录脚本，可直跑） | 报告 §3 数字（VRS-VQA 0.775 / CIDEr 0.284 / MME 0.541 / XLRS 0.359 / LEVIR 1.344） |

## 链上一览（复现顺序）

```
数据准备(①) → 训练多专家(②) → 量化(③) → 推理优化(④) → Token 剪枝(⑤) → 容错(⑥) → 全量评测(⑦)
```

## ⑧ 最终推理链路（报告 §5.3 + §6.3，仓库对照）

**模型切换策略（报告 §5.3）**：单一 Qwen3.5-4B 基座 + 四个 LoRA 专家适配器
（General/Grounding/Change/Caption），轻量级**规则路由器**按
`任务前缀 → 关键词 → 默认 General` 顺序匹配，Adapter Manager 动态加载并
仅激活**一个**适配器（基座常驻显存）。

| 推理参数 | 报告口径 | 仓库实现 | 一致 |
|---|---|---|---|
| 推理引擎 | vLLM 0.26 批量调度 | `rsmllm/serve.py`；评测 `evaluation/main.py` | ✅ |
| 像素区间 | 200,704 – 2,097,152 | `config.REPORT_CONF` + serve `mm_processor_kwargs` | ✅ |
| 解码 | greedy（temperature=0） | `SamplingParams(temperature=0)` | ✅ |
| 最大序列 | 16,384 | `--max-model-len`（REPORT_CONF） | ✅ |
| 批量/并发 | batch 64、max-num-seqs 64（全量评测） | `--eval_batch_size 64`、`REPORT_CONF` | ✅ |
| 前缀缓存 | **活动版报告未列出**（旧版实测 850.97→908.96 tok/s、+6.8%，仅共享前缀负载） | vLLM 默认 `enable_prefix_caching=True` | ⚠️ 报告未收录 |
| KV Cache 量化 | 可选优化，不用于部署 | 未启用 | ✅（一致） |
| 投机解码 MTP | 弃用（实测 0.45–0.58× 净亏损） | 未启用 | ✅ |
| 多专家路由规则 | 前缀[REF]/[CAP]/[VQA]/[MCQ]、关键词、默认 General | `evaluation/run_delta_task_router.py`（VQA/MCQ→general、referring→grounding、change→change、caption→caption） | ✅ |
| 服务侧路由 | 报告 §5.3 Adapter Manager | serve.py 为单模型入口（路由在评测链路），部署侧按报告规则实现 | ⚠️ 见上 |

所有"报告位置"编号以《技术报告》当前版本为准；本文件与 `quantization/README.md`、`tolerance/README.md`、`token_compression/README.md`、`training/README.md` 互为索引。
