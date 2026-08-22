## 当前情况
已经测试完几种剪枝方法在R=0.5和R=0.25的效果，确认目前L2 Norm剪枝方法最优；
当前的脚本来自 czb 同学。为了保持统一，wh & czb 在后续的实验中将保持脚本统一。

## 实验计划
### 1. 补充实验
补充对相对连续的R的测试，R=[0.1 0.25 0.35 0.5 0.65 0.75 0.9 1.0]，对L2 Norm、DivPrune进行重新测试；
以此绘制折线图，增强视觉效果，丰富报告内容；

### 2. Pre-LLM 和 In-LLM协同实验
由于测试的很多方法都是Pre-LLM的剪枝方法，考虑可以和In-LLM进行结合。目前可以考虑的In-LLM方法：
PDrop, CLSE, (仿制的)SA-GEM, Sparse-VLM, 简单的T2V, Clip（直接在第k层剪去所有Visual Token）

### 3. 简单Router设计

由于不同任务对于剪枝率的**敏感性*是不一样的，比如REF类型任务对于剪枝率非常敏感，因此我们可以设计简单Router对剪枝率进行调控。

### 4. 最新的SA-GEM方法测试
8/15 时，针对遥感的剪枝方法SA-GEM在Arxiv上发布。其剪枝方法有一定的参考价值，但是没有开源。因此我们可以
根据他的思路进行“仿制”，观察其效果。其具体的方法见SAE.md。

### 5. (TBD) 重测
因为微调处新增了多专家微调等方法，原先的实验均基于二阶段LoRA进行，可能已经过时了；因此我们可能需要对剪枝方法进行重新测试！由于随机的数据集数量不大，我们可以很快跑完。

###注：实验条件：
权重采用二阶段LoRA，两个 stage2 high_merged 都是从 Qwen3.5-4B 继续二阶段 LoRA微调后、再合并得到的完整 mo用相同样本和 batch 设置。
lel.safetensors 权重。跑分包含 baseline (不裁剪)、L2Norm 保留 50%、L2Norm 保留 25%:三组均使
日志中可核验各集样本数与 batch: logs/ful1-base-mme-5197.out:4、logs/fu11-base-xlrs-5:
98.out:4, logs/ful1-base-Vrsbench-5199.out:37424, logs/ful1-base-levircc-5200.out:4.

### 6. R 折线重测 —— 实验条件（多专家模型，已定稿）

- 模型：多专家 = `shared_models/lora_expert/base_model`（sft_stage1_clean merged）+ 3 expert LoRA（general/grounding/change，r=32/alpha=64）
- 剪枝方法：L2Norm（`pruner l2norm`）+ DivPrune（官方 max-min 余弦多样性删除，新增 `evaluation/pruners/divprune.py`）
- R（保留比例）：[0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]，R=1.0 为不剪枝 baseline
- 数据：DATA_ROOT 全量 4 数据集（vrsbench / mme / xlrs / levircc），每集随机抽 1000 条（`--random_samples 1000 --sample_seed 2026`）
  - 各 R 值用同 seed 抽样，保证样本一致，折线图对比公平
  - vrsbench 全量混有 VQA/Caption/Referring，抽 1000 自然含 referring（路由到 grounding 专家）
- batch_size = 32（剪枝逐样本 → 按专家分组 → 组内左 pad 批量生成）
- 结果输出：prune/output/prune_sweep/
- 脚本：evaluation/adapters/router_pruned.py（RouterPrunedAdapter）、evaluation/run_prune_sweep.py、slurm_scripts/run_prune_sweep_router.slurm、scripts/plot_prune_sweep.py

Smoke test 抽样分布（seed=2026）：
- vrsbench: 1000 = vqa 587 / caption 148 / referring 265（general 735 / grounding 265）
- mme: 1000（全 vqa / general）
- xlrs: 1000（全 vqa / general）
- levircc: 1000（全 caption / change）

### 7. 新 XLRS 任务接入（caption + visual grounding，官方口径）

XLRS 数据集新增了 caption（en/zh）与 visual grounding（en/zh）子集（8/13~8/15），仿照 VRSBench
的官方 prompt/reference 口径接入，保留最小任务前缀用于路由。

- 数据源：lora_expert 的 `split_evals`（同学已从 arrow 预处理好的扁平 jsonl）
  - caption: `xlrs_caption_en.jsonl`（934 条）
  - grounding: `xlrs_grounding_test_4096.jsonl`（6310 条，4096 高分辨率，test split）
- 处理：`scripts/preprocess_xlrs_caption.py` / `preprocess_xlrs_grounding.py` → `evaluation/data/` 下 messages 格式 jsonl
  - caption → `[CAP]` 前缀 + 文本 reference（BLEU/ROUGE/CIDEr）；**三段式 prompt 作为 system prompt**（`evaluation/prompts/xlrs_caption_en.txt`，对齐官方 CAPTION_PROMPT_EN）
  - grounding → `[REF]` 前缀（仅路由）+ **官方 question + 0-1 浮点 bbox `[xmin,ymin,xmax,ymax]`**（不 round，指标用 `GroundingIoU` 归一化到 0-1）
- 新增 evalset：`evaluation/evalsets/xlrs_caption.py`、`xlrs_grounding.py`
- 新增指标：`evaluation/metrics/grounding.py`（GroundingIoU）
- 注册：`evaluation/main.py`、`run_prune_sweep.py`、`run_task_adaptive.py` 的 `DATASETS` + `SYSTEM_PROMPTS`
- 路由自动复用：`[CAP]`→general 专家、`[REF]`→grounding 专家
- 2048 版（`datasets_data/`）：`scripts/fix_xlrs_2048.py` 从 split_evals 重新生成（本地 2048 图 + 0-1 bbox + 官方 question）
- 状态：脚本已写完、预处理 jsonl 已生成；**等同学确定最终多专家权重后再跑评测**

注：放弃"仿 VRSBench 0-100 整数 bbox"方案——它会 (a) 让 caption 短 prompt 与三段式 reference 的 n-gram 重叠趋近 0，(b) 给 grounding 小目标引入 8-16% 量化误差。改为对齐 XLRS 官方口径。

### 8. Delta 专家模型切换后的全方法重测

共享目录中的多专家模型已从旧版 `base_model + adapter_model.safetensors` LoRA
切换为新版 `W_expert = W0 + delta_model.pt`：基座为原始 Qwen3.5-4B，
`general/grounding/change` 各自提供完整 Delta。Delta 路由冒烟测试已通过。

- 旧 `prune_ablation.tex` 的所有方法结果均不能直接复用，需在新版专家权重上重新测试：Baseline、Uniform、Random、MMTok、L2Norm、SCOPE、DivPrune、FourierCompressor。
- 评测必须统一使用新版 Delta 加载、system prompt、输出清理和相同抽样/batch 条件。
- 先重新验证 Baseline 与 Router，再按原 R=0.5/0.25 补齐所有方法，最后重做完整 R 序列与 task-adaptive Router。
