# Delta 专家模型剪枝实验总览

## 1. 实验目标

在新版 Delta 多专家模型上评估视觉 token 剪枝方法，并根据任务敏感性设计任务级 Router。

模型结构：

```text
base_model + general Delta + grounding Delta + change Delta + caption Delta
```

专家权重目录：

```text
/users/u2024311136/shared/shared_models/lora_expert/lora/
```

## 2. 实验阶段

### 阶段 A：非 Caption 固定比例方法对比

结果目录：

```text
prune/output/delta_prune_ablation_batch32/
```

配置：

- 方法：Uniform、Random、MMTok、L2Norm、DivPrune、ScopeL2
- R：`0.50`、`0.25`
- 数据：VRSBench（VQA + Referring）、MME、XLRS、LEVIR-CC
- Caption：排除
- 结果：`48/48`

总结：`prune/output/delta_prune_ablation_batch32/SUMMARY.md`

### 阶段 B：非 Caption 完整 R 消融

结果目录：

```text
prune/output/delta_prune_r_sweep/
```

配置：

- 方法：6 种
- R（keep ratio）：`0.10, 0.25, 0.35, 0.50, 0.65, 0.75, 0.90, 1.00`
- 实际剪枝率：`90%, 75%, 65%, 50%, 35%, 25%, 10%, 0%`
- 数据：VRSBench（VQA + Referring）、MME、XLRS、LEVIR-CC
- Caption：排除
- 结果：`192/192`

总结：`prune/output/delta_prune_r_sweep/SUMMARY.md`

### 阶段 C：Caption expert 固定比例实验

结果目录：

```text
prune/output/delta_caption_ablation_batch32/
```

配置：

- Caption expert：`lora/caption/delta_model.pt`
- 方法：6 种
- R：`0.50`、`0.25`
- 数据：VRSBench Caption、XLRS Caption
- 结果：`24/24`

执行脚本：

```text
slurm_scripts/run_delta_caption_ablation.slurm
```

### 阶段 D：Caption R 消融

脚本已经准备：

```text
slurm_scripts/run_delta_caption_r_sweep.slurm
```

计划配置：

- 方法：6 种
- R：8 个点，与阶段 B 一致
- 数据：VRSBench Caption、XLRS Caption
- 总配置：`6 × 8 × 2 = 96`

状态：尚未提交，等待 GPU/QOS 提交限制释放。

## 3. 非 Caption 主要结论

| 任务 | 推荐方法 | 推荐 R | 依据 |
|---|---|---:|---|
| VQA | L2Norm | 0.50 | R=0.50 时 VQA=0.741 |
| Referring | ScopeL2 | 0.75 | Referring=0.609，低 R 时下降最明显 |
| MME | ScopeL2 | 0.50 | MCQ=0.576 |
| XLRS | L2Norm | 0.25 | MCQ=0.377 |
| Change | L2Norm | 0.50-0.75 | LEVIR CIDEr=1.322-1.364 |

当前第一版 Router 配置建议：

```python
{
    "referring": {"method": "scope_l2", "keep_ratio": 0.75},
    "vqa": {"method": "l2norm", "keep_ratio": 0.50},
    "mcq": {"method": "scope_l2", "keep_ratio": 0.50},
    "change": {"method": "l2norm", "keep_ratio": 0.50},
}
```

按 $\epsilon=0.05$ 的严格保留率阈值，默认配置为：VQA/MCQ 使用
$L_2$ Norm、$R=0.25$；Change 使用 $L_2$ Norm、$R=0.50$；Referring
使用 ScopeL2、$R=0.75$。若需进一步压缩而允许 Referring 性能下降，
可将 Referring 降至 ScopeL2、$R=0.50$。

ScopeL2 的 XLRS MCQ R 曲线：

| R | MCQ Accuracy |
|---:|---:|
| 0.10 | 0.366 |
| 0.25 | 0.371 |
| 0.35 | 0.375 |
| 0.50 | 0.375 |
| 0.65 | 0.366 |
| 0.75 | 0.364 |
| 0.90 | 0.361 |
| 1.00 | 0.361 |

ScopeL2 在 XLRS 上没有表现出相对 L2Norm 的稳定优势；R=0.25/0.50 时 L2Norm 分别为 `0.377/0.372`，因此 XLRS 仍优先采用 L2Norm。

## 4. Caption 固定比例结果

> 以下是旧 prompt/指标口径的阶段性结果，仅保留作历史对照。新版 Caption 结果应使用官方 user-instruction，并只报告 BLEU-1/2/3/4、METEOR、ROUGE-L。

主要指标格式：`BLEU-4 / ROUGE-L / CIDEr`。

| 方法 | R | VRSBench Caption | XLRS Caption |
|---|---:|---:|---:|
| Uniform | 0.50 | 0.152 / 0.378 / 0.377 | 0.045 / 0.190 / 0.000 |
| Random | 0.50 | 0.148 / 0.373 / 0.386 | 0.046 / 0.191 / 0.000 |
| MMTok | 0.50 | 0.154 / 0.375 / 0.363 | 0.045 / 0.189 / 0.000 |
| L2Norm | 0.50 | 0.157 / 0.376 / 0.374 | 0.044 / 0.189 / 0.000 |
| DivPrune | 0.50 | 0.154 / 0.377 / 0.372 | 0.045 / 0.190 / 0.000 |
| ScopeL2 | 0.50 | **0.158** / 0.376 / 0.367 | 0.044 / 0.189 / 0.000 |
| Uniform | 0.25 | 0.140 / 0.369 / 0.333 | 0.042 / 0.190 / 0.000 |
| Random | 0.25 | 0.130 / 0.361 / 0.326 | 0.041 / 0.190 / 0.000 |
| MMTok | 0.25 | 0.141 / 0.370 / 0.326 | 0.039 / 0.190 / 0.000 |
| L2Norm | 0.25 | 0.135 / 0.363 / 0.318 | 0.039 / 0.189 / 0.000 |
| DivPrune | 0.25 | 0.141 / 0.369 / 0.337 | 0.040 / 0.189 / 0.000 |
| ScopeL2 | 0.25 | **0.146** / 0.373 / 0.356 | 0.040 / 0.189 / 0.000 |

阶段性观察：

- VRSBench Caption 在 R=0.50 下整体优于 R=0.25。
- VRSBench Caption 的 BLEU-4 当前由 ScopeL2 略高，但 CIDEr 由 Random 略高。
- XLRS Caption 的 CIDEr 全部为 0，需要在报告中单独核查官方 reference/prompt 与 CIDEr 评测口径，暂不据此比较方法优劣。

## 6. Router 实现状态

## 7. General + exp7 最新 R 消融

新版 General Expert 已按以下顺序生成：

```text
raw Qwen3.5-4B + General Delta + exp7 PEFT LoRA
```

转换后的 Delta：

```text
prune/output/new_experts/general_exp7_delta.pt
```

新版 R sweep 已完成 `192/192`，结果目录：

```text
prune/output/delta_exp7_r_sweep/
```

总结：`prune/output/delta_exp7_r_sweep/SUMMARY.md`

关键变化：exp7 使 MME baseline 从旧版约 `0.566` 提升到 `0.702`，XLRS 从约 `0.361` 提升到 `0.477`；VQA、Referring、LEVIR-CC 基本保持稳定。

完整实验条件、权重路径、抽样、prompt、batch、剪枝和复现命令记录在：

```text
prune/output/delta_exp7_r_sweep/SUMMARY.md
```

## 8. Task Router Smoke

候选任务路由配置：

```python
{
    "vqa": {"method": "l2norm", "keep_ratio": 0.50},
    "mcq": {"method": "l2norm", "keep_ratio": 0.50},
    "change": {"method": "l2norm", "keep_ratio": 0.50},
    "referring": {"method": "scope_l2", "keep_ratio": 0.75},
    "caption": {"method": "l2norm", "keep_ratio": 0.50},
}
```

Smoke Job `7122` 已通过，覆盖 VQA、Caption、Referring、MCQ、Change 和 XLRS Grounding：

- General + exp7 正常加载；
- Caption 正确路由到 Caption expert；
- Referring 正确路由到 Grounding expert；
- Change 正确路由到 Change expert；
- bbox 输出可解析；
- 无 OOM、Traceback 或 CUDA 错误。

Smoke 脚本：`scripts/smoke_test_delta_task_router.py`

## 5.1 R=1 完整集口径复核

为核对论文表格，Job `6852` 使用完整数据集、`eval_batch_size=4`、think 预填充和 R=1 重新评测。结果为：

| 数据集/任务 | 当前严格指标 | 表格指标 | 结论 |
|---|---:|---:|---|
| VRSBench VQA | 0.705 | 0.775 | 表格使用 GPT-4o-mini semantic accuracy，不是严格 Accuracy |
| VRSBench Referring | 0.667 | 0.667 | 对齐 |
| MME MCQ | 0.541 | 0.541 | 对齐 |
| XLRS MCQ | 0.358 | 0.359 | 基本对齐 |
| LEVIR-CC CIDEr | 1.346 | 1.344 | 基本对齐 |

证据文件：

```text
outputs/gpt_judge/vrsbench_vqa_judged_gpt-4o-mini.json
```

其中记录了：

```text
v2_router semantic_acc = 0.7753
v2_router strict_acc   = 0.7043
```

因此当前 Delta 权重、专家路由和指标实现没有发现导致整体性能异常的错误。后续表格对比必须明确区分 `strict Accuracy` 与 `semantic Accuracy`。

已完成：

- `evaluation/router/rules.py` 支持 `[CAP] -> caption`
- `RouterAdapter` 支持 `caption_lora`
- `DeltaPrunedAdapter` 支持 Caption expert
- `task_prune_config` 支持按任务同时切换剪枝方法和 R

相关实现：

```text
evaluation/router/rules.py
```

待完成：

- Caption R 消融
- 非 Caption + Caption 的统一 Router 正式评测
- Router 与统一 baseline 的最终对照表
