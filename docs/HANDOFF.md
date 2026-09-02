# RS-MLLM 仓库约定与操作手册（Handoff）

> 给队友的同步文档：仓库当前状态、关键约定、评测/推理/训练怎么跑、已知坑。
> 最后更新：2026-09-02

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
- `general` / `grounding` 的 canonical 快照已经是 base + delta + **LoRA(PEFT)**
  的一次合并结果：`expert_general` / `expert_ground`
- `change` / `caption` = base + delta（**无 LoRA**，架构如此）

**完整模型矩阵**（12 个，全部已上传 ModelScope + profile 注册）：

| 专家 | bf16 | w8a8 | gptq |
|---|---|---|---|
| general（含 LoRA）| `expert_general` | `expert_general_w8a8` | `expert_general_gptq` |
| ground（含 LoRA）| `expert_ground` | `expert_ground_w8a8` | `expert_ground_gptq` |
| change（无 LoRA）| `expert_change` | `expert_change_w8a8` | `expert_change_gptq` |
| caption（无 LoRA）| `expert_caption` | `expert_caption_w8a8` | `expert_caption_gptq` |

- LoRA adapter 单独托管：`expert_general_lora` / `expert_ground_lora`，作为
  canonical 快照的来源记录；不要再次叠加。
- 旧的 `*_full` 模型及其量化版是二次合并产物，**不用于新评测**；对应 profile
  仅保留历史结果兼容性，`get_model` 已拒绝这些旧别名。
- canonical 模型 profile 已注册（`evaluation/vllm_eval/expert_profiles.py`，sha 实测）——评测器加载会校验
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
# 模型: general/ground 使用已合并一次的 canonical 快照; change/caption 使用 delta-only

# 单模型评测:
./rsmllm.sh → [1] 评测 → route/single

# 结果: results/<manifest>_<profile>_<时间戳>/  (每次独立, 互不覆盖)
#   clean_summary.json(正式分) / predictions.jsonl(原始) / unit_scores.jsonl
```

**评测器约定**（`evaluation/vllm_eval/vision_opd_vllm_eval.py`）：
- **关 thinking**（`enable_thinking=False`，与 qwen35vl 默认一致；多模态评测不需思考）
- system prompt 按任务前缀（[VQA]/[CAP]/[REF]/[CD]/[MCQ]，报告口径）
- manifest 保留任务前缀 + MCQ **保留选项行**（模型需看到 A/B/C/D）
- `--quantization` 由一键入口透传为 `bf16` / `w8a8` / `gptq`，同时写入
  `run_config.json`/`run_manifest.json` 的 engine/model 元数据；实际量化由模型目录配置决定
- 不在离线 vLLM 评测中动态挂 LoRA：vLLM 0.26 的 offline `LLM` 没有经过验证的多模态
  `add_lora_adapter` 路径；canonical 模型已完成一次合并，禁止二次应用
- `--enforce-eager`（router_eval 当前稳定配置；graph 模式的启动与编译可能耗时，不能仅凭中途无输出判定挂起）
- 统一入口默认 `batch_size=128` 是一次提交给 vLLM offline engine 的 request window；
  `max_num_seqs=64` 仍是 GPU 调度并发上限。每个 window 使用有界 4-worker CPU 池解码
  RGB 图像，并在解码后关闭文件句柄；不改变图像字节、像素策略或请求顺序。
- GPU 利用率出现 0%--高值交替是当前路径的预期特征：`Rendering prompts`/首轮 Triton
  JIT 属于 CPU/预热阶段，GPU 随后以短批量 prefill/decode burst 工作。不要用一次
  `ps`/`nvidia-smi` 快照判定卡死；应结合完整日志和明确的 GPU 进程状态。
- vLLM 错误中的 `cuda:0` 是进程内可见设备序号；`CUDA_VISIBLE_DEVICES=1` 时它对应物理 GPU 1，须用 `nvidia-smi` 的 GPU index/UUID 核对，不能据此断言落在物理 GPU 0。
- 评测器在创建 `LLM` 前用 NVML 解析物理 GPU UUID、做显存预检并持有进程锁；同一物理 GPU 的重复评测会快速失败，不会再启动冲突的 EngineCore；不同物理 GPU 仍可并行。
- `router_eval` 保留原有断点续跑语义：记录子评测失败后继续其余任务，最终以非零退出；GPU 资源冲突在 `LLM` 前由 guard 快速拒绝，不会启动冲突的 EngineCore。
- 时间戳输出目录 + resume 保护（code_sha256 校验）

**known issues**：
1. vllm graph 模式下 grounding/bbox 的启动与编译可能超过 3 分钟；本次干净的
   caption/grounding smoke 均 exit 0，但冷启动开销显著，且尚未用完整任务矩阵证明
   稳定收益。`router_eval` 仍保留 `--enforce-eager` 作为当前稳定配置；不要用单次
   CPU/GPU 快照判定卡死。
2. MCQ 无选项行 → 模型答语义 0 分（已修：保留选项行）
3. `xlrs` 官方无 test split（HF 只有 train 74 分片）——我们用其 index 顺序前 3,080 条；`xlrs_caption_en`(934) / `xlrs_grounding_test`(6,310) 是官方 test
4. 评测环境勿加 llmcompressor（依赖冲突）
5. `results/` 中带 `expert_*_full` 的旧目录来自二次 LoRA 合并，不能作为 canonical
   指标；新结果应使用 `expert_general`/`expert_ground` 及其 canonical 量化 profile。
6. 报告后续 PEFT LoRA 数字与当前 vLLM 结果必须注明计分/清单：canonical BF16
   VRSBench VQA 当前 exact-match 为 70.91%（37,409 条），与报告
   `+ LoRA (1 stage)` 的 70.43% 接近；报告多专家行的 77.5% 使用 GPT-4 语义判定，
   不能直接与 exact-match 相减。当前 Caption/LEVIR 结果中的 `caption_smoke_metrics`
   是 lexical proxy，不是报告的官方 CIDEr；当前 XLRS Grounding 的 17.59%（6,310
   条、清单中的 resized 图像）也不能直接替代报告后续 PEFT 域对齐实验的 31.57%→
   32.77%。

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
- canonical `expert_general` / `expert_ground` 已完成一次 delta+LoRA 合并，量化直接以
  这些快照为输入；`scripts/merge_lora_to_model.py` 仅用于显式指定的 delta-only 自定义目录，
  对已有 `merge_manifest.json` 的模型会拒绝二次合并
- 量化产物含 conversion_manifest.json（记录来源 sha）

## 8. 报告口径速查

- 报告 §3：VRS-VQA 0.775 / CIDEr 0.284 / MME 0.541 / XLRS 0.359 / LEVIR 1.344
- 评测配置：greedy + bf16 + 像素 200,704–2,097,152 + max_len 16,384
- xlrs 只用 en（zh 未评测）
- 运行时优化仅改变 request window/CPU 图像解码，不改变上述 PEFT 模型权重；
  比较精度时优先看同一清单、同一评分实现，报告数字作为标注过协议的参考基线。
