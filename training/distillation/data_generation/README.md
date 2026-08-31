# Train-only 数据生成（Codex 登录态）

本目录用于复现 `MME-new`、`XLRS-new`、`VRSBench-new` 和
`XLRS-grounding-new` 的语言生成/审核阶段，并包含后两套 Grounding 数据的本地
SAM3 train-only 端到端管线。Codex 部分调用本机 `codex login` 的 ChatGPT 登录态，
不读取 OpenAI API key，也不会自动创建 key。

```bash
python -m pip install -r requirements.txt
```

## 安全边界

- 只允许 `train` split；不得把公开测试标签用于生成。
- General SFT job 只携带图像和官方训练集的**聚合风格 profile**，不携带原问题、答案或框。
- Grounding 必须先由本地 SAM3 完成检测、分数过滤、无效框过滤、NMS 和同类官方框 IoU
  过滤；语言阶段只有固定 candidate id 和只读框上下文，输出 schema 没有坐标字段。
- 全量运行前必须先生成并人工查看 20 条 smoke；结果被改动后原批准自动失效。
- 每条结果记录 Codex 版本、模型、job payload 和图像 SHA256。

## 1. 登录检查

```bash
codex login
codex login status
```

若 `codex login status` 显示 API-key 登录，`codex_runner.py` 会拒绝运行。

## 2. General SFT jobs

```bash
python prepare_general_jobs.py \
  --source-manifest assets/source_manifest.jsonl \
  --question-manifest assets/question_manifest.jsonl \
  --style-profile assets/style_profile.json \
  --split train \
  --output runs/general/jobs.jsonl
```

## 3. Grounding 固定框语言 jobs

SAM3 后处理产物必须包含 `candidate_id`、`image`、`class`、`bbox_xyxy_px`、
`mask_path` 和 `overlay_path`。

```bash
python prepare_grounding_jobs.py \
  --candidates assets/postprocessed_candidates.jsonl \
  --image-root assets/images \
  --official-train-records assets/official_train.jsonl \
  --dataset-profile xlrs \
  --split train \
  --boxes-are-frozen-postprocessed \
  --output runs/xlrs_grounding/jobs.jsonl
```

VRSBench 将 `--dataset-profile` 改成 `vrsbench`。

## 4. Smoke、人工批准和全量运行

以 General 为例：

先做完全离线的输入检查；该命令不检查登录，也不会调用 Codex：

```bash
python codex_runner.py validate \
  --jobs runs/general/jobs.jsonl \
  --schema schemas/general_sft_annotation.schema.json \
  --run-root runs/general \
  --stage smoke
```

```bash
python codex_runner.py run \
  --jobs runs/general/jobs.jsonl \
  --schema schemas/general_sft_annotation.schema.json \
  --run-root runs/general \
  --stage smoke
```

人工查看 `runs/general/smoke/results/` 的全部 20 条及对应图像后：

```bash
python codex_runner.py approve-smoke \
  --run-root runs/general \
  --reviewer YOUR_NAME \
  --confirmation 'I reviewed the smoke outputs and approve the full run'

python codex_runner.py run \
  --jobs runs/general/jobs.jsonl \
  --schema schemas/general_sft_annotation.schema.json \
  --run-root runs/general \
  --stage full
```

默认模型是 `gpt-5.6-sol`，可通过 `--model` 或 `CODEX_DATA_MODEL` 显式覆盖。运行使用
`--ephemeral --ignore-user-config --sandbox read-only`。

## 可复现边界

这组脚本当前封装的是 **Codex 语言生成/审核层**，不是四个数据集从官方原始文件开始的
完整构建器：

- `MME-new`、`XLRS-new` 需要先提供 train-only `source_manifest.jsonl`、
  `question_manifest.jsonl` 和冻结的 `style_profile.json`；
- `VRSBench-new`、`XLRS-grounding-new` 需要先完成 SAM3 推理及固定框后处理，再把
  candidates 交给本目录；语言层不会生成或修改框；
- Grounding 收集结果是 `question-decisions`，还必须交给与候选框对应的 finalize
  实现才能得到训练 JSON；仓库内实现和完整命令见
  [`sam3_pipeline/README.md`](sam3_pipeline/README.md)；
- 离线 `validate`、fixture 测试和结构一致并不能证明新一轮模型生成内容与历史内容逐条
  相同。模型版本、登录账户和视觉判断都会影响语义结果，必须完成 20 条人工 smoke
  审核后才能讨论效果一致性。

输入行若声明了 `split`，非 `train` 会被拒绝。老 manifest 没有 `split` 字段时，调用者
仍须通过来源清单和哈希证明其训练来源；命令行的 `--split train` 不能把未知来源自动变成
train 数据。

## 历史效果一致性审计

离线 fixture 验证的是输入契约、确定性过滤、坐标/格式和数据泄漏门禁，不等于重新调用
模型后会得到与历史逐字相同的输出。当前可核对的历史证据如下：

| 流程 | 历史保留产物 | 当前可验证范围 | 尚缺的效果验证 |
|---|---|---|---|
| MME-new | 有效训练分量实际来自 `old_mme_sft_3736`：3,736 条、1,264 图；另一个 `new_mme_sft_2550` 历史效果较差 | 最终训练 JSON 的数量、格式和选项分布 | 3,736 条版本的 source/question/style/accepted/audit 中间件已不在原路径，不能逐条重建历史审核链 |
| XLRS-new | 800 图、3,080 条，source/question/style/audit 均保留 | 当前 builder 可重放 3,080 jobs；13 类任务数量与历史逐项一致；Codex-login 真实 smoke 20/20 通过 schema、冻结身份和最终收集器 | 用户人工批准 smoke 及重新训练评测 |
| VRSBench-new | 5,000 图 → 7,552 SAM3 candidates → 942 accepted | train 选择、固定词表分类、SAM3、NMS/官方框去重、native finalize 的离线 fixture | 真实 SAM3/Codex smoke，以及历史 raw-base/strict semantic 第二独立视觉门禁 |
| XLRS-grounding-new | 500 图 → 1,975 candidates → 303 accepted | train Arrow-only 选择、SAM3、过滤、关系描述 finalize、fixed-five 归一化 fixture | 真实 SAM3/Codex smoke、第二独立视觉复核及最终 LoRA 评测 |

因此，在完成上述 smoke、独立复核和训练评测前，只能声明**工程链路/格式一致**，不能声明
新一轮生成集合、语义质量或最终模型指标与历史完全一致。项目文档中的 `MME-new=3736`
是保留有效实验的命名约定，不是历史文件名。

## 5. 完整性收集

```bash
python collect_results.py \
  --jobs runs/general/jobs.jsonl \
  --results-dir runs/general/full/results \
  --kind general \
  --output runs/general/general_training.jsonl
```

Grounding 使用 `--kind grounding`，输出可直接作为既有 finalize 脚本的
`question-decisions` 输入。收集器要求每个 job 恰好一个结果，拒绝缺失、额外文件、
身份字段变化、非原生格式或语言 meta 泄漏。

## 6. ModelScope 生成展示包

历史运行目录中可能包含本机绝对路径和受上游许可证约束的 review 图。发布时使用
`stage_generation_audit.py` 只导出 train-only 结构化审计，所有本机路径都会改写，原图
仍由官方仓库下载。VRSBench SFT 会分别生成 `vqa.jsonl`（1,241 条）、
`referring.jsonl`（526 条）和 `caption.jsonl`（296 条）；两套 Grounding 会保留候选、
过滤、接受/拒绝及最终样本记录。视觉 gallery 可在下载官方图片后通过
`sam3_pipeline` 的 finalize 命令重新渲染。

```bash
python stage_generation_audit.py \
  --vrsbench-sft-parquet /path/to/vrsbench_sft.parquet \
  --xlrs-general-root /path/to/xlrs_new_run \
  --vrsbench-grounding-root /path/to/vrsbench_new_run \
  --xlrs-grounding-root /path/to/xlrs_grounding_new_run \
  --output /path/to/staged_distillation_data
```

MME 3,736 条有效版本的历史中间审计已经缺失，因此只能发布最终冻结训练 JSON，不能伪造
source/question/style/audit 展示。OPSD 的生成轨迹与 mask 审计放在
`RS-MLLM-Self-Evolution-Data`，避免在两个 ModelScope 仓库重复存储。
