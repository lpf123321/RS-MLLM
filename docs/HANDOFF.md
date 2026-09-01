# RS-MLLM 仓库约定与操作手册（Handoff）

> 给队友的同步文档：仓库当前状态、关键约定、评测/推理/训练怎么跑、已知坑。
> 最后更新：2026-09-01

## 0. 仓库定位

- 独立交付仓库（GitHub: `lpf123321/RS-MLLM`，main），**评测/推理不依赖 slurm**（slurm 脚本已归档 `archive/slurm_experiments/`，仅证据）
- 模型/数据托管 ModelScope（`Fun10165/*`，已设公开），HF 官方源作 fallback

## 1. 环境（两套 uv 项目，别混）

| 环境 | 位置 | 内容 | 安装 |
|---|---|---|---|
| 训练 | 根 `.venv/` | torch 2.8 cu128 + deepspeed + trl + peft | `bash setup.sh` |
| 评测 | `evaluation/vllm_eval/.venv/` | torch 2.11 cu129 + vllm 0.26 + modelscope + gradio | `bash evaluation/vllm_eval/setup_env.sh` |
| 量化 | `quantization/.venv/` | llmcompressor 0.13（**勿加入评测环境**，compressed-tensors 冲突）| `bash quantization/setup_env.sh` |

**注意**：评测环境 pyproject 不要加 llmcompressor（vllm 0.26 锁 compressed-tensors 0.17.0，llmcompressor 要 0.17.1/0.18 冲突）。量化放 quantization/。

## 2. 模型（ModelScope，get_model 按需下载缓存到 .models/）

**专家架构**（队友训练口径）：
- `general` / `grounding` = base + delta + **LoRA(PEFT)** → 完整版 = **`expert_general_full`** / `expert_ground_full`
- `change` / `caption` = base + delta（**无 LoRA**，架构如此）

**完整模型矩阵**（12 个，全部已上传 ModelScope + profile 注册）：

| 专家 | bf16 | w8a8 | gptq |
|---|---|---|---|
| general（含 LoRA）| `expert_general_full` | `expert_general_full_w8a8` | `expert_general_full_gptq` |
| ground（含 LoRA）| `expert_ground_full` | `expert_ground_full_w8a8` | `expert_ground_full_gptq` |
| change（无 LoRA）| `expert_change` | `expert_change_w8a8` | `expert_change_gptq` |
| caption（无 LoRA）| `expert_caption` | `expert_caption_w8a8` | `expert_caption_gptq` |

- LoRA adapter 单独托管：`expert_general_lora` / `expert_ground_lora`（合并脚本 `scripts/merge_lora_to_model.py`）
- `_full` 模型 **profile 已注册**（`evaluation/vllm_eval/expert_profiles.py`，sha 实测）——评测器加载会校验
- 别名注册在 `rsmllm/config.py::MODEL_REGISTRY`

## 3. 数据

- **评测图片**：ModelScope 数据集 `Fun10165/rs-mllm-benchmark-data-clean`（6 个 tar，~8.8GB，公开）
  - `python scripts/fetch_benchmark_data.py --all`（默认 ModelScope）
  - `--source hf` 走 HF 官方（XLRS 系列全量巨大，仅复现用）
  - 目录契约：`datasets/shared_datasets/<VRSBench|MME-RealWorld-RS|XLRS-Bench-lite|XLRS-Bench_caption_en|XLRS-Bench_visual_grounding_en|LEVIR-CC>/`
- **评测清单**（Sample schema）：`evaluation/vllm_eval/manifests/`（gitignore，由 `scripts/build_sample_manifest.py` 从 `datasets_data/` 转换，`--images-root` 重写路径）
- **训练数据**：`bash scripts/fetch_training_data.sh`（默认 `yasumi/rs-mllm-datasets`，队友仓库）
- datasets_data/*.jsonl 是 **Git LFS**（clone 后 `git lfs pull`）

## 4. 评测（唯一入口）

```bash
# 一键路由专家评测(只选量化方式, 其余自动分配):
python -m rsmllm.router_eval --quant bf16     # 或 w8a8 / gptq
# 映射: general→vqa/mme/xlrs-lite; ground→referring/grounding-en; change→levir; caption→caption×2
# 模型: general/ground 用 full(含LoRA), change/caption 用 delta-only

# 单模型评测:
./rsmllm.sh → [1] 评测 → route/single

# 结果: evaluation/vllm_eval/results/<manifest>_<profile>_<时间戳>/  (每次独立, 互不覆盖)
#   clean_summary.json(正式分) / predictions.jsonl(原始) / unit_scores.jsonl
```

**评测器约定**（`evaluation/vllm_eval/vision_opd_vllm_eval.py`）：
- **关 thinking**（`enable_thinking=False`，与 qwen35vl 默认一致；多模态评测不需思考）
- system prompt 按任务前缀（[VQA]/[CAP]/[REF]/[CD]/[MCQ]，报告口径）
- manifest 保留任务前缀 + MCQ **保留选项行**（模型需看到 A/B/C/D）
- `--enforce-eager`（**必须**：观测到 vllm graph 模式下 grounding/bbox 任务启动阶段挂起（无输出、CPU 0%）；eager 规避后跑通。**机制未确认**，router_eval 已内置）
- 时间戳输出目录 + resume 保护（code_sha256 校验）

**known issues**：
1. vllm graph 模式下 grounding/bbox 任务启动挂起（现象，机制未确认；已用 --enforce-eager 规避）
2. MCQ 无选项行 → 模型答语义 0 分（已修：保留选项行）
3. `xlrs` 官方无 test split（HF 只有 train 74 分片）——我们用其 index 顺序前 3,080 条；`xlrs_caption_en`(934) / `xlrs_grounding_test`(6,310) 是官方 test
4. 评测环境勿加 llmcompressor（依赖冲突）

## 5. 推理（Router，报告 §5.3）

```bash
bash scripts/start_router.sh      # 4 专家 vLLM 实例: general=8001 ground=8002 change=8003 caption=8004
python -m rsmllm.router --chat    # 规则路由对话
# 用完: pkill -f "vllm.entrypoints"  (释放显存, 不影响评测)
```

## 6. 训练

```bash
bash scripts/train.sh stage1_clean   # 本地前台(默认); --slurm 走集群
bash scripts/fetch_training_data.sh  # 训练数据(默认 yasumi/rs-mllm-datasets)
```

## 7. 量化

```bash
bash quantization/setup_env.sh
python scripts/quantize_qwen35_vlm.py --method w8a8-int8 --model <完整模型> \
  --calibration <calib>.jsonl --calibration-meta <meta>.json --output <out>
```
- 量化前先合并 LoRA（`scripts/merge_lora_to_model.py`）得到完整模型
- 量化产物含 conversion_manifest.json（记录来源 sha）

## 8. 报告口径速查

- 报告 §3：VRS-VQA 0.775 / CIDEr 0.284 / MME 0.541 / XLRS 0.359 / LEVIR 1.344
- 评测配置：greedy + bf16 + 像素 200,704–2,097,152 + max_len 16,384
- xlrs 只用 en（zh 未评测）
