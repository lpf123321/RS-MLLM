# CVSearch × SAM3-LoRA × Vision-OPD 自进化框架

> 2026-08-18 修正版。修改前的 README 已原样归档为
> [`docs/archive/README_legacy_gt_bbox_2026-08-18.md`](docs/archive/README_legacy_gt_bbox_2026-08-18.md)，
> 归档时两者 SHA-256 均为 `1ae55a3dfda96e8352a71e62bf8dbe189aa701883e894a2492014841dc58871a`。
> 本文后半部保留的 2026-08-17 结果属于“使用数据集 GT bbox”的旧 VOPD 特殊对照，不能代表无 bbox 自进化方法。

## 2026-08-18 无答案 bbox 修正版

### 目标与输入

第一版只处理有标准答案的 A/B/C/D 选择题。标准 JSONL 每行必须提供：

```json
{"image_path":"images/1.jpg","question":"...","options":["...","...","...","..."],"answer":"A"}
```

`bbox` 可选，只能写入 `optional_gt_bbox_*` 诊断字段，不参与轨迹筛选、SAM 训练或正式晋升。旧 VOPD 的 `problem/original_images/answer/bbox` 格式仍兼容，但其中 bbox 同样只作可选诊断。无 bbox 时按照 `question_family` 做确定性分层；若数据 source ID 与既有 VOPD 完全一致，则继续精确复用冻结的 `4993/624/624` source-index split。

当前修正版配置为 `configs/vopd_qwen3vl_2b_bbox_free.yaml`，独立 workspace 为：

`/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_bbox_free`

它不会读取旧 `qwen3vl_2b_one_fifth` 的 state、`.done` 或 candidate 权重。

999/624 正式单轮使用 `configs/vopd_qwen3vl_2b_bbox_free_one_fifth.yaml` 与另一个独立 workspace；`data.train_limit=999` 表示精确截取冻结 train source-index 顺序的前 999 条，valid 仍为完整 624，test 仍为 624 且封存。

该正式配置将 SAM LoRA 设为单卡、`num_workers=0` 运行。首次 4 卡尝试在模型加载后发生 DDP 同步死锁；随后单卡的 multiprocessing DataLoader 也在首批卡住（worker 等待队列、主线程 futex、没有 step/checkpoint，均不是 OOM）。因此保留已完成轨迹和 verified manifest，改用主进程直接加载数据恢复。此调整只影响数据加载方式，不改变样本、损失或优化器。

### CVSearch 路由与结构化轨迹

根节点仍计算 Qwen 回答置信度，但配置固定为：

- `answering_confidence_threshold_lower=0`
- `fast_threshold=1.1`
- `force_visual_search=true`

Quick Answer 条件是 `root_confidence > lower + fast_threshold`。置信度范围为 `[-1,1]`，所以 `-1、0、1` 都不可能进入旧 branch 1（轨迹中的 `search_mode=0`）。每条生成轨迹会对此做强制断言。

修正版 schema 为 `trace_schema_version=4`：

- `search_mode=1`：所有视觉目标第一次 SAM 定位成功，即原 branch 2。
- `search_mode=2`：至少一个目标第一次 SAM 失败，随后进入树/局部搜索并获得最终 node，即原 branch 3。
- `search_mode=3`：搜索失败；只记录，不进入两条训练数据。
- `target_traces`：逐目标保存 `target_text`、首次 SAM 成败及预测、最终 node 列表、node 来源和置信度。
- `final_mllm_crop`：保存实际 Qwen 视觉输入的原图坐标范围、node 扩 patch 范围、方图 padding 和最终 crop 范围。
- `final_mllm_pixel_sha256`：对实际传入 Qwen processor 的 PIL 像素计算哈希。

baseline、SAM-only、MLLM-only、both 的 teacher artifact 使用各自独立的 tag 目录，禁止同一 sample ID 的不同组合互相覆盖。schema v4 专门加入了这项路径隔离；v3 轨迹不得恢复复用。

上游 Qwen 接口只新增可选 `return_artifacts`；默认 CVSearch 调用行为不变。teacher 图使用无损 PNG 保存，保存后重新按 mode、尺寸和像素字节计算 SHA，不一致立即失败。旧的“根据搜索框事后重建 montage”不再用于训练。

### 两条训练数据流

```text
无 bbox MCQ
  └─ CVSearch（强制跳过 Quick Answer）
      ├─ mode 1 + 回答正确 ──────────────┐
      ├─ mode 2 + 回答正确 ────────┐     │
      │                            │     ├─ Vision-OPD：每个原问题恰好 1 条
      │                            │     │  student=完整无标注原图
      │                            │     │  teacher=当次 Qwen 实际视觉输入
      │                            │
      │                            └─ SAM3：只展开首次 SAM 失败的目标
      │                               prompt=该 target_text
      │                               pseudo bbox=该目标唯一最终 node
      └─ mode 3 / 回答错误 ─────────── 仅审计，不训练
```

Vision-OPD 接受条件只包括：`search_mode in {1,2}`、最终答案正确、最终视觉输入有效。旧的 GT bbox、teacher 重答、GT token log-prob 增益和 jitter 稳定性不再是硬过滤。多目标问题也只生成一条原问题记录，不产生没有标准答案的子问题。

SAM COCO 只读取“mode 2、最终回答正确”的轨迹，并逐目标处理：

- 首次 SAM 已成功的目标不训练。
- 首次失败且恰好有一个最终 node，使用完整原图、`target_text` 和该 node 的 `xywh` 伪框。
- 无 node、多个 node 无法唯一归属、退化/越界，或裁剪后面积比例不在 `[0.001,0.85]`，都会写入 `*_pseudo_bbox_rejections.jsonl`。
- annotation 固定标记 `pseudo_bbox_source=tree_final_node`，并保留实际 `pseudo_bbox_node_source`；COCO info 明确 `uses_dataset_gt_bbox=false`。
- 不再构造“全部 target + correction 复制”；SAM 样本数等于有效的首次失败 target-node 对数。

因此 SAM 继续复用已跑通的新 LoRA 方法：`loss_ce + presence + L1 bbox + GIoU + ranking`，rank 16/alpha 32，仅训练 DETR q/k/v/out；正负比例 1:1、presence weight 5、ranking weight 2/margin 0.2。这里的框是 CVSearch 弱伪标注，所有报告统一使用 `pseudo_*`，不得称为人工 GT 指标。

### 自动负例

每轮先从新的 SAM train/valid COCO 自动调用 `/home/cpy/sam3_lora_test/scripts/build_negative_manifests.py`：

- train 生成并均衡选择 verified generic/random，一条正样本对应一条负 prompt。
- valid 生成并尽量均衡选择 generic/random/semantic/counterfactual/crop。
- verifier 固定 `/data1/shared/Qwen3-VL-8B-Instruct`，seed 42，`absent_probability >= 0.9`。
- 未验证行、coverage 非 100%、image ID 不一致或 train/valid source ID 泄漏都会直接失败。
- 最终 manifest、COCO、verifier `config.json` 的 SHA 和各类型数量写入同目录 metadata；已冻结且覆盖完整的 manifest 可恢复复用。

### 无 GT bbox 晋升规则

同一冻结 valid 仍比较 baseline、SAM-only、MLLM-only、both，并单独运行 Qwen direct regression。正式硬门为：

- candidate correct 不得少于 baseline。
- correct 至少增加 1 条即可构成收益。
- 若 correct 相同，则 mean pop 必须至少下降 5%，且 Tree Search 样本数不得增加。
- 使用新 SAM 时，负 prompt `FPR@0.5` 不得高于 baseline。
- 使用新 Qwen 时，direct accuracy 下降不得超过 1 个百分点。
- 多组合通过时依次按 accuracy 高、mean pop 低、FPR 低选择。

held-out branch-3 伪框的 `pseudo AP50/Recall/top1 IoU` 仍报告，但明确是诊断项，不参与晋升。test 在最终模型选择完成前继续封存。

### 修正版命令与当前验收

```bash
cd /home/cpy/self_evolve_cvsearch

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b_bbox_free.yaml doctor

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b_bbox_free.yaml prepare

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b_bbox_free.yaml smoke --round 0
```

已完成的数据层验收：新 workspace 已生成冻结 4993/624/624 与 64/16 smoke split；当前特殊 VOPD 源数据的 6241 个 bbox 全部只保存为 `optional_gt_bbox_*`。自动单测覆盖无 bbox 标准 MCQ、root `-1/0/1` 边界、mode 1/2 多目标展开、歧义 node 拒绝、teacher 像素 SHA、无 GT SAM COCO、`pseudo_*` 指标及五种晋升拒绝/接受情形。GPU 64/16 完整 smoke 的实际数量与结果记录在下一节。

### 2026-08-18 修正版 64/16 smoke 实际结果

完整 smoke 已于 01:34–01:51 完成，完成标志为：

`/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_bbox_free/runs/smoke/round_00/.done`

全程没有 CUDA OOM、内核 OOM 或未验证负例混入，workspace 完成后约 21GB。80 条 baseline 轨迹的 `search_mode=0` 为零，保存的 teacher PNG 全部通过逐像素 SHA 复核。

首次审计发现 v3 的四组合共用同一 teacher 文件名，后运行组合会覆盖先运行组合；答案与训练结果未受影响，但文件级审计不成立。现已升级为 schema v4 并重新生成全部轨迹：128 条 train/valid/四组合轨迹对应 128 个唯一 teacher 路径，逐像素 SHA 全部通过，且重生成前后的 correct、accuracy、mean pop、Tree Search 数和 routes 完全一致。

数据构建与训练实际数量：

- train 64 条中，34 条正确 mode 1/2 原问题进入 Vision-OPD；仍是每个原问题一条。
- branch-3 逐目标展开后，SAM 使用 12 个 train 伪框和 2 个 valid 伪框；没有使用 `optional_gt_bbox_*`，没有 correction duplication。
- train 负例为 generic 6 + random 6，valid 为 generic 1 + crop 1；均由 Qwen3-VL-8B 验证 `absent_probability >= 0.9`，coverage 100%。valid 只有 2 个 SAM 伪框，semantic/counterfactual 没有可生成候选，因此本次小样本不可能覆盖五类，完整 valid 才用于五类均衡目标。
- SAM LoRA 训练 1 epoch，983,040 个可训练参数，约 18.5 秒；adapter SHA-256 为 `dcdf8916c2af9a4f922715bd3f80c388708401735dd90cfc4f0107d6e3b48088`，合并 checkpoint SHA-256 为 `ddba71b91401aaf01c6da3620ce73530a3e8df18439b7c7f79eaa11bcf1feda5`。
- Vision-OPD 使用 34 条数据完成 8 个 optimizer steps；teacher image swap fraction=1.0、policy fallback=0、empty target batch=0，`actor/vopd_loss` 从 step 1 的约 0.0775 到 step 8 的约 0.0117。GPU 0–3 实际峰值监控约 22.6GB/卡。

冻结 smoke valid 上的结果：

| 组合 | correct / 16 | accuracy | mean pop | Tree Search 数 |
|---|---:|---:|---:|---:|
| baseline | 9 | 56.25% | 3.875 | 7 |
| SAM-only | 8 | 50.00% | 1.500 | 2 |
| MLLM-only | 10 | 62.50% | 3.625 | 7 |
| both | 8 | 50.00% | 1.625 | 2 |

SAM candidate 与 baseline 在这 2 条 verified valid 负例上的 FPR@0.5 都为 0；样本过少，只能证明评测管线接通，不能证明选择性不退化。branch-3 伪框上的 pseudo AP/Recall 也仅作诊断。

Qwen direct 从 baseline 的 9/16 降到 candidate 的 8/16，下降 6.25 个百分点；因此按正式 1 个百分点门限，MLLM-only 和 both 都会被拒绝。SAM-only/both 又各少答对 1 条，也会被正式门控拒绝。`promotion.json` 中的 `selected=mllm_only` 只来自 smoke plumbing override，不能解释为正式模型晋升。test 仍未读取。

### 2026-08-18 修正版 999/624 正式单轮结果

正式 round 0 已完成，完成标志、原始指标和门控决定为：

- `/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_bbox_free_one_fifth/runs/full/round_00/.done`
- `/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_bbox_free_one_fifth/runs/full/round_00/metrics.json`
- `/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_bbox_free_one_fifth/runs/full/round_00/promotion.json`

数据和训练实际数量：

- train 999 条路由为 mode 1=`633`、mode 2=`366`、mode 0/3=`0`；其中回答正确且视觉输入有效的 583 条原问题进入 Vision-OPD。
- valid baseline 624 条路由为 mode 1=`416`、mode 2=`208`、mode 0/3=`0`。
- SAM 按首次失败目标展开为 train 205、valid 122 个 `tree_final_node` 伪框，没有使用数据集 GT bbox，也没有 correction duplication。
- 训练负例为 verified generic 103 + random 102；valid 负例为 generic 30 + random 30 + semantic 29 + counterfactual 4 + crop 29，coverage 均为 100%。
- SAM LoRA 使用单卡、`num_workers=0` 完成 1 epoch，训练约 264.5 秒；adapter SHA-256 为 `adbce116472eccdbcb8d6f69131093987932e6d325ded813fe11ce7b66427693`，合并 checkpoint SHA-256 为 `bd41f20b93a2221eef5821a2146f1c6d98b9491634353b06e3f4c8188d1d7fb4`。
- Qwen3-VL-2B Vision-OPD 使用 583 条记录、batch 96 完成 6 个 optimizer steps，纯训练约 4 小时 3 分；各步 `actor/vopd_loss` 为 `0.0664、0.0559、0.0440、0.0375、0.0401、0.0510`。teacher swap fraction 始终为 1、policy fallback 和 empty target batch 始终为 0。
- 最终 workspace 约 24GB；test 624 条仍未读取。

冻结 valid 的正式结果：

| 组合 | correct / 624 | accuracy | 相对 baseline | mean pop | mode 1 / mode 2 |
|---|---:|---:|---:|---:|---:|
| baseline | 386 | 61.86% | - | 2.8269 | 416 / 208 |
| SAM-only | 332 | 53.21% | -8.65 pp / -54 | 7.6811 | 0 / 624 |
| MLLM-only | 385 | 61.70% | -0.16 pp / -1 | 2.7580 | 418 / 206 |
| both | 327 | 52.40% | -9.46 pp / -59 | 7.6346 | 0 / 624 |

Qwen direct accuracy 从 288/624（46.15%）变为 287/624（45.99%），仅下降 0.16 个百分点，因此 direct regression 门本身通过；但 MLLM-only 在完整 CVSearch 中少答对 1 题，违反 `candidate correct 不得低于 baseline`，仍不晋升。

SAM 的 verified-negative FPR@0.5 从 stock 的 3/122（2.46%）降到 candidate 的 0/122（0%），所以 FPR 门通过。但新 SAM 让全部 624 条 valid 都从初始定位转入 Tree Search，correct 少 54 条且 mean pop 增加 4.8542，因而明确拒绝。`both` 同样拒绝。最终 `accepted=false`、`selected=null`，SAM 与 Qwen 均保持 round 0 之前的模型。

branch-3 伪框诊断中，stock/candidate 的 pseudo AP50 分别为 0.634%/0.378%，Recall@1 IoU 0.5 都为 4.10%，top-1 mean IoU 为 6.95%/7.99%。这些框来自弱伪标注，结果只说明与 held-out tree node 的一致性很低，不作为正式晋升硬门。candidate 的负 prompt presence mean 从 0.053 升到 0.268，但最终框分数均未超过 0.5；报告 FPR 时必须同时保留这项诊断，不能只写“FPR 为零”。

本轮恢复过程暴露并修复了两个资源编排问题：

1. 轨迹评测原先固定启动 GPU 0–7。外部任务占用 GPU 4–7 每卡约 22GB 时，SAM-only shard 6 在加载 Qwen/SAM 后 OOM。现改为 `trajectory_workers` 配置，本轮固定 4，只使用 GPU 0–3；完整 schema-v4 主轨迹存在时会直接复用，恢复不会重跑 baseline 或训练。
2. SAM 选择性评测默认两个 DataLoader worker，在本机出现 worker poll、主线程 futex 等待，并伴随长路径 `AF_UNIX path too long`。既有脚本新增兼容参数 `--num-workers`（默认仍为 2），本框架传 0 后 baseline/candidate 各约 46 秒完成。该修改只改变数据加载进程数，不改变模型、样本或指标计算。

最终 12 项核心单测全部通过，正式 `promotion.json` 三个候选均为 rejected；旧 GT bbox 实验和本次无 bbox 实验必须继续分开解释。

## 以下为 2026-08-17 旧 GT bbox 对照记录

本目录是独立编排层。模型训练数据、轨迹、权重、日志和报告只写入本目录配置的 `workspace`，不会写入或清理三个上游 Git 仓库。

## 当前固定实现

### 三个上游仓库

| 模块 | 本地路径 | 用途 |
|---|---|---|
| CVSearch | `/home/cpy/ICML26-CVSearch` | 目标提取、SAM 定位、Fast/Tree Search 和最终回答 |
| Vision-OPD | `/home/cpy/Vision-OPD-training-clean` | 固定教师的区域–全局 OPSD 训练 |
| SAM3-LoRA | `/home/cpy/sam3_lora` | SAM3 模型、LoRA 注入和底层训练依赖 |

SAM 的任务化训练入口不重新编写，直接复用已经跑通并完成 VSTAR/VOPD/HR-Bench 验证的 `/home/cpy/sam3_lora_test` 方法。该目录不是第四个上游仓库，而是既有方法、冻结 manifest、checkpoint 和机器报告的实验工作区。

关键复用文件：

- `scripts/train_vopd_bbox_lora.py`：annotation-level bbox LoRA 训练。
- `scripts/vopd_negative_common.py`：manifest 负查询、真/假框 ranking loss。
- `scripts/evaluate_vopd_selectivity.py`：正框定位与负 prompt FPR 联合评测。
- `scripts/merge_vopd_lora_checkpoint.py`：将 adapter 合并回 stock SAM3 checkpoint。
- `manifests/splits.json`：冻结的 4993/624/624 数据划分。
- `manifests/negative_repair/full/train_mix_generic_random_1to1.jsonl`：已由视觉模型确认的 generic/random 1:1 训练负例。
- `manifests/negative_repair/full/valid_benchmark_balanced_624.jsonl`：冻结 valid 选择性评测集。

已验证的参考模型是 `generic_random_e3`：

- adapter：`/home/cpy/sam3_lora_test/outputs/negative_repair/full_candidates/generic_random_p5_r1_rank2/epoch_03_lora_weights.pt`
- 合并权重：`/home/cpy/sam3_lora_test/outputs/negative_repair/merged/generic_random_e3/sam3.pt`
- 合并权重 SHA-256：`b2604d100334b7a448105b62d5d502a4ee1cfc8433510cfc9b6afedb7e96c597`

### 初始模型与数据

- MLLM：`/home/cpy/workspace/model/student/Qwen3-VL-2B-Instruct`
- SAM3：`/home/cpy/workspace/model/sam3/sam3.pt`
- 数据：`/home/cpy/workspace/dataset/vopd_dataset`
- 配置：`configs/vopd_qwen3vl_2b.yaml`
- 隔离工作目录：`/home/cpy/workspace/self_evolve_cvsearch/qwen3vl_2b_evolution`
- seed：42

`doctor` 会检查 MLLM 的 architecture 必须为 `Qwen3VLForConditionalGeneration`，同时检查三个 Git 仓库、既有 SAM 方法脚本、冻结 split 和负例 manifest 均存在。SAM 参数若偏离已经验证的方法，配置加载会直接失败。

## 一轮自进化

每轮从冻结的 `SAM_t + MLLM_t` 开始：

1. 在 train 上运行 CVSearch，保存完整搜索轨迹。
2. 从轨迹构造 SAM 数据：GT bbox 正例、CVSearch 纠错样本 2 倍采样、已验证的 generic/random 负查询。
3. 从通过过滤的轨迹构造 Vision-OPD parquet：完整原图给 student，单 crop 或确定性拼图给 fixed teacher。
4. 串行训练 SAM candidate 和 MLLM candidate，各完整遍历 1 epoch。
5. 在同一个 valid 上评测 baseline、SAM-only、MLLM-only、both 四种组合。
6. 应用定位、选择性、直接回答、搜索准确率和搜索成本门控，允许只晋升一个模块。
7. 无组合通过时停止；通过时保存新的模型谱系并进入下一轮。

`test` 不用于训练、筛选或门控。只有最终模型确定后才允许读取。

## 数据与防泄漏

- 直接复用 `sam3_lora_test/manifests/splits.json`，不重新随机划分。
- 固定规模为 train 4993、valid 624、test 624。
- smoke 固定读取 train 前 64 条和 valid 前 16 条，不读取 test。
- MLLM 学生只读取 `original_images` 无红框原图。
- 问题会删除红框提示，统一为一个 `<image>` 占位符和仅输出 A/B/C/D 的格式。
- bbox 内部统一为 `xyxy`；CVSearch 接口处转换为 `xywh`。
- 每条轨迹记录图像、模型和配置哈希，以及所有拒绝原因。

OPSD 轨迹必须通过：CVSearch 回答正确、zoom-in teacher 回答正确、GT token log-prob 至少提升 0.05、面积比例处于 `[0.001, 0.85]`、bbox 扰动 ±5% 后至少 2/3 正确、无缺图/空 target/非法解析。根图直接回答不进入 OPSD。

## 已验证 SAM 训练方式

自进化中的 SAM candidate 严格沿用 `generic_random_e3`：

- annotation-level referring prompt，不使用 category loader。
- GT bbox 始终是真框监督；搜索框不冒充 GT。
- train 负例为视觉确认 absent 的 generic/random prompt，约 1:1；正负查询比例 1:1。
- presence weight 5。
- ranking margin 0.2、weight 2。
- rank 16、alpha 32、dropout 0.1。
- 只训练 DETR encoder/decoder 的 `q_proj/k_proj/v_proj/out_proj`。
- vision、text、geometry、mask decoder 冻结。
- VOPD 没有 mask，mask/Dice loss 为 0；保留 one-to-many 分支。
- 每轮 1 epoch。
- 参考方法的 `distill_weight=0`，因此当前配置不会构造 Base-SAM replay。若未来做蒸馏消融，必须使用新配置名和新 workspace，不能覆盖本基线。

round 1 以后从上一轮合并 checkpoint 继续，因此累计保留此前 LoRA 更新；每轮仍同时保存本轮 adapter 和新的 stock-layout 合并 checkpoint。

## Vision-OPD / Qwen3-VL-2B

- student：当前 `MLLM_t` + 完整原图。
- teacher：同一个冻结 `MLLM_t` + zoom-in 图。
- `teacher_model_source=fixed`，一轮内不使用 EMA 漂移。
- rollout n=8；smoke n=2。
- top-k distillation=100，generalized JSD alpha=0.5。
- importance ratio clipping=2.0。
- learning rate=2e-6。
- 每轮 1 epoch。
- 当前使用 4 张 RTX 5090，sequence parallel=2，rollout TP=1。

## 命令

所有命令从本目录执行，`--config` 位于子命令之前：

```bash
cd /home/cpy/self_evolve_cvsearch

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b.yaml doctor

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b.yaml prepare

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b.yaml smoke --round 0

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b.yaml run-loop --stage full --max-rounds 3

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b.yaml evaluate --split test --checkpoint best

conda run -n cvsearch env PYTHONPATH=src \
  python -m evolve --config configs/vopd_qwen3vl_2b.yaml report
```

阶段是可恢复的：轨迹按样本和 shard 增量保存；已经完整合并的模型不会重复训练；只有整轮指标、晋升决定和 state 都成功写入后才生成 `.done`。

## 输出结构

```text
qwen3vl_2b_evolution/
├── manifests/                 # 数据统计、三个仓库运行前后快照
├── data/
│   ├── splits/                # 冻结 train/valid/test 与 smoke manifest
│   ├── trajectories/          # baseline 与三种 candidate 组合轨迹
│   ├── teacher_images/        # crop/montage 与逐样本搜索日志
│   ├── sam_coco/              # annotation-level COCO 和负例 manifest
│   └── opd_parquet/           # Vision-OPD 区域–全局配对
├── models/                    # 每轮 SAM adapter/merged 和 MLLM checkpoint
├── runs/                      # 命令、训练日志、指标、晋升与 state
├── cache/                     # HF、Torch、vLLM/Ray 等隔离缓存
└── reports/README.md          # 自动汇总报告
```

## 门控规则

正式轮次要求：

- CVSearch valid 至少多答对 2 条。
- SAM AP50 和 Recall@0.5 相对基线下降均不超过 0.01。
- 负 prompt FPR@0.5 不高于基线。
- correction/真实定位指标至少一项提升。
- MLLM Direct accuracy 下降不超过 1 个百分点。
- mean pop 不超过基线 1.25 倍；accuracy 提升超过 1 个百分点时放宽。
- 固定回归集下降不超过 1.5 个百分点。

smoke 的目标是验证“轨迹→过滤→双训练→四组合评测→晋升”的工程链路，会记录所有质量门控，但不以 16 条 valid 的随机波动阻断管线。

## 验收与已知约束

当前自动验收包括：6241 条数据与冻结 split、bbox 往返、问题/答案解析、单/多 crop 拼图、完美框/空框指标、晋升规则、Qwen3-VL architecture 和已验证 SAM 配置检查。运行前后会记录三个上游仓库的 commit 与 dirty 状态；编排不会恢复、清理或覆盖这些既有改动。

需要特别注意：

- 四组合轨迹使用独立 tag 和 shard 文件，禁止把 baseline shard 混入 candidate 指标。
- 非法或缺失答案统一计错。
- `sam3_lora_test` 的既有 checkpoint 仅作方法参考和回归基准，不直接冒充自进化 round candidate。
- test 在最终选模前保持封存。

## 2026-08-17 Qwen3-VL-2B 冒烟验收

新的隔离 workspace 已完成一次 64 train + 16 valid 的完整闭环，完成标志为：

`qwen3vl_2b_evolution/runs/smoke/round_00/.done`

实际结果：

- 64 条 train 轨迹全部生成；48 条含可训练 target。
- SAM 训练负例全部来自既有验证 manifest：generic 26、random 22。
- 8/64 条轨迹通过严格 OPSD 过滤并用于训练，没有为了凑 batch 重复样本。
- SAM：1 epoch、48 个样本、983,040 个可训练参数（总参数的 0.1168%）；ranking loss 非零，distillation loss 为 0。
- SAM candidate 已保存 adapter，并合并为约 3.45GB 的 stock-layout checkpoint。
- OPSD：8 条记录、batch 4、共 2 个 optimizer step；`actor/vopd_loss` 约从 0.0972 降至 0.0683，`grad_norm` 约为 24.38/23.38。
- OPSD teacher image swap fraction=1.0、policy fallback=0、empty target batch=0，证明 fixed teacher 确实读取了 bbox image。
- baseline / SAM-only / MLLM-only / both 四种组合各自完成 16 条独立 valid 轨迹。

CVSearch smoke accuracy：

| 组合 | correct / 16 | accuracy | mean pop |
|---|---:|---:|---:|
| baseline | 10 | 62.50% | 3.8125 |
| SAM-only | 9 | 56.25% | 3.3750 |
| MLLM-only | 11 | 68.75% | 3.8125 |
| both | 10 | 62.50% | 3.3750 |

Direct accuracy 在 baseline/candidate 上均为 9/16。SAM candidate 的 AP50 从 0.2280 升到 0.3785、Recall@0.5 持平为 0.5714，但负例 FPR 从 0.0714 升到 0.2857，因而正式门控不会晋升该 SAM。smoke 为验证工程链路，最终按最高 CVSearch accuracy 记录 MLLM-only；不能把 16 条 valid 的变化当作正式收益结论。

本次验收中修复了两个会污染结论的问题：

1. tagged 四组合 shard 合并原先会误读未带 tag 的 baseline shard，现已使用 `valid_sam_only.*`、`valid_mllm_only.*`、`valid_both.*` 独立合并。
2. 部分轨迹有 teacher prediction 但因未通过后续过滤而没有 log-prob；指标现在只对完整 log-prob pair 计算增益，不再因 `None` 中断整轮。

## 2026-08-17 单轮 1/5 数据实验

本实验使用独立配置 `configs/vopd_qwen3vl_2b_one_fifth.yaml`，不修改正式全量和 smoke 状态：

- workspace：`/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_one_fifth`（`/home/cpy/workspace` 实际链接到 `/data2/cpy`）。
- train：冻结 seed 42 train 顺序的前 999/4993 条，实际比例约 20.008%；子集 SHA-256 为 `64a961d3813b07e72cfd2296f681728a01f292b7cae6d42401b98874d2c94c8a`。
- valid：仍使用完整冻结 624 条；test 624 条保持封存。
- 只运行 round 0；SAM 方法、1:1 负例、OPSD 和晋升门控均保持不变。
- SAM adapter、SAM 合并权重、Qwen FSDP 最终状态和合并模型均写入该 workspace 的 `models/full/round_00/`。
- Qwen OPSD 当前是全参数 FSDP（`lora_rank=0`），磁盘大头不是 SAM LoRA。该实验设置 `checkpoint_save_freq=-1` 且 `max_actor_ckpt_to_keep=1`，只保留强制写出的最终训练 checkpoint，避免每个 global step 重复保存约 13GB。
- 机器可审计定义：`manifests/one_fifth_experiment.json`。

### Round 0 实际运行结果

本轮从 2026-08-17 19:59:53 运行至 22:37:22，墙钟时间约 2 小时 37 分。完成标志、指标和门控决定分别位于：

- `runs/full/round_00/.done`
- `runs/full/round_00/metrics.json`
- `runs/full/round_00/promotion.json`
- 自动报告：`reports/README.md`

数据与训练过程：

- train CVSearch 轨迹完整生成 999/999；valid baseline 及 SAM-only、MLLM-only、both 三个候选组合均完整生成 624/624。
- 严格 OPSD 过滤接受 139/999 条轨迹。Qwen3-VL-2B 使用全参数 FSDP 完成 1 个 optimizer step，耗时约 58 分 26 秒；最终合并模型约 4.6GB，保留的原始 checkpoint 约 13GB。
- SAM annotation-level bbox LoRA 实际使用 839 个 train 样本和 518 个 valid 样本，在 4 张 GPU 上完成 1 epoch，耗时约 410 秒；可训练参数 983,040，占总参数 0.1168%。
- 整个实验 workspace 最终约 21GB，没有发生 CUDA OOM 或内核 OOM。OPSD checkpoint 保存后的 Ray 清理阶段曾有一个 DataLoader worker 收到 `SIGKILL`；checkpoint 合并成功且主流程继续完成，因此它不是已确认的训练 OOM，但仍应作为资源清理异常保留在审计记录中。

冻结 valid 624 条上的 CVSearch 结果：

| 组合 | correct / 624 | accuracy | 相对 baseline | mean pop | 门控 |
|---|---:|---:|---:|---:|---|
| baseline | 372 | 59.62% | - | 2.5433 | - |
| SAM-only | 389 | 62.34% | +2.72 pp / +17 | 1.2933 | 拒绝：负 prompt FPR 上升 |
| MLLM-only | 378 | 60.58% | +0.96 pp / +6 | 2.4696 | **接受并晋升** |
| both | 387 | 62.02% | +2.40 pp / +15 | 1.2420 | 拒绝：负 prompt FPR 上升 |

配对统计中，SAM-only 相对 baseline 的 95% CI 为 `[+0.32, +5.13]` pp，McNemar `p=0.0372`；both 的 95% CI 为 `[-0.16, +5.13]` pp，`p=0.0966`；MLLM-only 的 95% CI 为 `[-0.32, +2.24]` pp，`p=0.2113`。这些统计用于描述波动范围，当前门控本身并不要求提升达到统计显著。

SAM 选择性评测解释了为什么最高 accuracy 的 SAM-only 没被晋升：

| SAM | AP50 | Recall@0.5 | 负 prompt FPR@0.5 |
|---|---:|---:|---:|
| stock baseline | 29.85% | 55.79% | 14.67% |
| round-0 candidate | 60.52% | 61.58% | 25.87% |

candidate 的正框 AP50 和 recall 明显提高，但负 prompt FPR 增加 11.20 pp，违反“不得高于 baseline”的选择性门控。因此 `sam_only` 和 `both` 的其他检查虽然全部通过，仍被拒绝；这也避免了仅凭 CVSearch accuracy/搜索次数改善而接受更容易对不存在目标产生框的 SAM。

Qwen direct accuracy 从 288/624（46.15%）变为 287/624（45.99%），下降 0.16 pp，在允许下降 1 pp 的范围内。最终 `promotion.json` 选择 `mllm_only`：当前 MLLM 更新为 round-0 Qwen candidate，当前 SAM 保持 stock `/home/cpy/workspace/model/sam3/sam3.pt`；test 624 条仍未读取。
## 2026-08-18：局部 SAM 恢复 mask 多实例方案（当前实验）

本节是在既有记录之后新增的实验方案；上文的 GT-bbox 对照、`tree_final_node` 弱伪框实验及其结果均保留，不应与本节混写。修改前全文已再次归档到 `docs/archive/README_before_sam_mask_multi_2026-08-18.md`。

### 为什么再次修改 SAM 数据

上一版只选择初始 SAM 失败的目标，却把 CVSearch 最终树节点矩形当成伪框。树节点只是搜索窗口，可能包含大量背景，也不代表 SAM 真正学会了定位。本轮只学习以下增量：

```text
原图 SAM 失败
→ CVSearch 用树搜索得到更小的局部 crop
→ 在该 crop 上重新运行 SAM
→ local_retry_1 或 local_retry_2 首次成功输出 mask
→ mask 映射回原图并生成紧致 pseudo bbox
```

mode 1、mode 2 中初始已成功的其他目标、仅由 Qwen 判定成功的 `fine` 节点、`fine_fallback`、mode 3及最终答案错误的轨迹都不进入 SAM 正样本。这里的 mode 是整题路由；真正的 SAM 入集条件仍在目标级检查。

### 多实例与损失

同一 `target_text` 在一次局部 SAM 成功中可能返回多个 mask。本轮引入原 `sam3_lora` 的 exhaustive 多目标表示：一张完整原图只建立一个文本查询，各有效 mask 分别生成紧致 bbox，并共同成为 `object_ids_output=[0..N-1]`。不再做跨实例 union 框，也不把同图复制成互相矛盾的单实例样本。

mask 以无损二值 PNG 保存并记录像素 SHA、局部 crop、SAM score、retry 阶段和原图坐标，但第一轮仍为 bbox-only LoRA：`segment=None`、`with_seg_masks=false`、`apply_to_mask_decoder=false`，不计算 mask/Dice loss。损失保持 `loss_ce×20 + presence×5 + L1 bbox×5 + GIoU×2 + ranking×2`，ranking margin 为 `0.2`，rank/alpha 为 `16/32`。训练继续使用 Qwen3-VL-8B 确认不存在的负 prompt，总正负查询比例为 `1:1`。

首轮smoke还验证出：旧tree窗口使用的最小面积比例`0.001`会错误排除有效的小目标紧致mask（两个valid恢复框分别约为`0.000504`和`0.000736`），导致valid正样本为零。新方案因此仅对`sam_success_mask`采用`0.0001`，旧tree-node配置仍保留`0.001`；这不是放宽SAM成功条件，空mask、退化框、答案错误和非局部恢复目标仍全部拒绝。首次失败运行在训练前安全停止，没有启动任何模型训练，也没有OOM。

### VOPD 数据不变

Vision-OPD仍接受最终答案正确且视觉输入有效的mode 1/2原问题：student为完整原图，teacher为CVSearch当次实际交给Qwen的最终图像，每道原题只生成一条选择题记录。某条轨迹可以进入VOPD，但如果没有目标满足“初始SAM失败、局部SAM恢复成功”，就不会生成SAM记录。

### 隔离实验与当前验证

- 配置：`configs/vopd_qwen3vl_2b_sam_recovered_multi_one_fifth.yaml`
- workspace：`/home/cpy/workspace/self_evolve_cvsearch_weights/qwen3vl_2b_sam_recovered_multi_one_fifth`
- smoke：64 train / 16 valid；为保证能测试局部恢复管线，在冻结split内固定加入此前观测到的恢复锚点，再用原顺序补足样本。
- 正式实验：999 train / 624 valid、round 0、1 epoch；test继续封存。
- 评测：baseline、SAM-only、MLLM-only、both和Qwen direct regression，沿用correct、mean pop、Tree数量、FPR和direct回归门控。

当前15项单元测试已通过，包括mode 1零SAM样本、初始成功目标排除、mask原图坐标映射和一prompt多Object loader。真实锚点 `0168b1fd1282914bc3de` 的单轨迹验证也已通过：initial失败，`local_retry_1`成功，mask紧致框映射为原图 `[70,1721,131,1799]`，COCO明确标记 `pseudo_bbox_source=sam_success_mask`、`uses_dataset_gt_bbox=false`、`uses_tree_node_bbox=false`、`uses_mask_loss=false`。后续smoke及正式结果继续追加在本节，不覆盖历史内容。

### 64/16 smoke 实际结果

smoke 已于 2026-08-18 完整结束，完成标志为 `runs/smoke/round_00/.done`，全流程没有 CUDA OOM 或内核 OOM。64 条 train 和 16 条 valid 轨迹均完整生成；严格恢复条件最终得到 train 3 个目标/3 个 mask 实例、valid 2 个目标/2 个 mask 实例。train 三个恢复事件均来自 `local_retry_1`，valid 分别有一个 `local_retry_1` 和一个 `local_retry_2`。这说明工程链路有效，但困难恢复样本非常稀疏。

SAM LoRA 使用 983,040 个可训练参数完成 1 epoch，耗时约 7.88 秒；mask decoder 不训练且没有 mask/Dice loss。训练 manifest 有 3 条已验证负查询，与 3 条正查询保持 `1:1`。候选 adapter 已合并为 stock-layout checkpoint，SHA-256 为 `09e92178781a43697ea916b39e8039cbcab160de2e478ac8ad98e228515fffac`。在两个 mask 伪框 valid 样本上的诊断 AP50 为 53.80%，该数字仅作小样本管线检查，不是正式定位结论。

35/64 条回答正确且视觉输入有效的轨迹进入 VOPD，Qwen3-VL-2B 完成 8 个 optimizer step。冻结 16 条 smoke valid 的结果如下：

| 组合 | correct / 16 | accuracy | Tree 数 | mean pop |
|---|---:|---:|---:|---:|
| baseline | 11 | 68.75% | 8 | 5.2500 |
| SAM-only | 11 | 68.75% | 7 | 3.9375 |
| MLLM-only | 12 | 75.00% | 8 | 5.1250 |
| both | 12 | 75.00% | 7 | 3.9375 |

Qwen direct regression 为 9/16 → 9/16，没有下降；负 prompt FPR@0.5 为 baseline 0、candidate 0。smoke 记录选择 `both`：相对 baseline 多答对 1 条、Tree 少 1 条、mean pop 降低 1.3125。由于 valid 只有 16 条且 SAM 诊断集只有两个目标，这些变化只用于确认完整闭环，不宣称统计显著收益；正式结论以 999/624 单轮实验为准。

### 正式轮次中断与可恢复重启

首次正式 Qwen 训练以 Codex PTY 子进程运行。它完成第 1/6 个 optimizer step 后，在第 2 步过程中于 2026-08-19 01:29 左右与父会话、全部 Ray worker 一同被外部终止。内核、Ray、CUDA 和训练日志均没有 OOM、Traceback 或模型错误；停止前第 1 步 `vopd_loss=0.067856`、`grad_norm=19.125`，数值正常。该步处理约 257 万 token，耗时约 4823 秒。由于当时 `checkpoint_save_freq=-1`，只留下 rollout 审计文件，没有可恢复模型 checkpoint。

重启不修改算法和全局训练量：仍使用 583 条 VOPD 记录、全局 batch 96、rollout `n=8`、学习率 `2e-6`。工程设置改为 GPU 0–3 共四张卡，并将 `checkpoint_save_freq` 改为 1、最多保留一个 actor checkpoint；每个 optimizer step 后都可恢复。正式编排改由独立 user systemd unit `cvsearch-recovered-full-r0.service` 承载，不再依赖对话/PTTY生命周期。已有 999/999 train、624/624 valid baseline 轨迹、负例、SAM adapter 和合并 checkpoint 全部复用；Qwen 从 step 0 重训，完成后自动继续 SAM-only、MLLM-only、both、direct regression、SAM selectivity 和晋升汇总。

## 2026-08-19：Fine Search 成功后强制 SAM 的两组对照

本节只追加新实验，不覆盖前文。修改前 README 全文已归档为 `docs/archive/README_before_fine_sam_variants_2026-08-19.md`。

严格恢复方案在 999 条 train 轨迹中只有 4 个目标满足“初始 SAM 失败、原流程局部 SAM 重试成功”。主要原因不是 206 个 Fine 成功目标上的 SAM 失败，而是原 CVSearch 在 Fine Search 成功后直接把树节点交给 Qwen，根本不会再运行 SAM。为区分“增加 SAM 伪标注”与“同时改变 CVSearch 最终视觉区域”的效果，新增两个可配置策略：

- `fine_sam_policy=trace_only`：First/Second/Third Fine 成功后，在该 Fine 节点 crop 上额外运行 SAM并保存 mask；最终搜索节点、Qwen 输入、路由和 `num_pop` 仍保持原值。
- `fine_sam_policy=replace`：执行同一 SAM 探测；存在 SAM 输出时，将每个局部框映射回原图并作为 `fine_sam` 最终节点，SAM 无输出时回退原 Fine 节点。该组会真实改变 Qwen 的最终视觉输入。

两个策略均不改变树的构建、节点排序、Fine 阈值、原有 `local_retry_1/2` 或 mode 定义。SAM 训练仍要求整题 mode 2、最终答案正确且该目标 initial SAM 失败。每个目标只选择执行顺序中第一次成功的后续 SAM 事件：

```text
fine_probe_1 → local_retry_1 → fine_probe_2 → local_retry_2 → fine_probe_3
```

mask 继续只生成多实例紧致 bbox；损失、LoRA rank/alpha、正负 `1:1` 和不训练 mask decoder 的设置全部不变。默认 `fine_sam_policy=off`，因此现有严格实验与原 CVSearch 行为不受影响。

隔离配置及 workspace：

- trace-only：`configs/vopd_qwen3vl_2b_fine_sam_trace_one_fifth.yaml`，workspace `qwen3vl_2b_fine_sam_trace_one_fifth`；
- replace：`configs/vopd_qwen3vl_2b_fine_sam_replace_one_fifth.yaml`，workspace `qwen3vl_2b_fine_sam_replace_one_fifth`。

新增策略、坐标回映、off 零副作用、可配置 recovery stages 和受控模型复用后共 17 项单元测试通过。为避免与严格正式轮次争抢 GPU 0–3，两组实验由独立服务 `cvsearch-fine-sam-variants.service` 排队：严格轮次退出后依次执行各自的 `prepare → 64/16 smoke → 999/624 full round 0`。其中 trace-only 不改变 Fine 节点和 teacher 像素，故复用严格实验对应 smoke/full Qwen candidate，以隔离 SAM 数据策略并避免重复约 4 小时的相同 VOPD 训练；复用路径和来源会写入 `opd/reuse_model.json`。replace 会改变最终 Qwen 视觉输入，仍完整独立训练 Qwen。最终将在本节继续追加恢复目标/实例数、梯度与 OOM、四组合 accuracy、Tree、mean pop、FPR、direct regression 和晋升结论。

### 严格局部恢复正式轮次最终结果

严格方案于 2026-08-19 08:02 完成，`runs/full/round_00/.done`、`metrics.json` 和 `promotion.json` 均已落盘。训练数据仅有 train 3 个初始失败后恢复的目标/3 个 mask 实例、valid 2/2；所有框均为 `sam_success_mask`，没有 GT bbox、tree node bbox 或 mask loss。SAM LoRA 1 epoch 约 6.82 秒，valid 负 prompt FPR@0.5 为 0。

Qwen 使用 583 条 VOPD 记录完成 6 个 optimizer step，总训练时间约 4 小时 4 分。每步 `vopd_loss` 为 `0.06638, 0.05440, 0.04481, 0.03756, 0.04027, 0.05023`，对应 `grad_norm` 为 `19.0, 12.25, 5.09, 7.0, 3.98, 4.5`；整体下降且无 CUDA OOM。训练 100% 完成并保存 step 6 后，Ray 清理阶段有一个 DataLoader worker 收到 `SIGKILL`，不影响 checkpoint 或合并模型。评测启动时还修复了未使用的 LLaMA-3 conversation tokenizer 在线探测：改为 `local_files_only=True`，避免离线环境每个进程等待 Hugging Face 超时；候选轨迹在零已完成样本时安全重启并全部补齐。

冻结 valid 624 条的最终结果：

| 组合 | correct / 624 | accuracy | Tree 数 | mean pop | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 386 | 61.86% | 208 | 2.8269 | - |
| SAM-only | 387 | 62.02% | 157 | 2.2404 | +1 correct，Tree -51 |
| MLLM-only | 389 | 62.34% | 210 | 2.7853 | +3 correct，Tree +2 |
| both | 389 | 62.34% | 160 | 2.2147 | +3 correct，Tree -48 |

SAM-only 的配对变化为 22 条由对变错、23 条由错变对，净增 1；both 为 24 条由对变错、27 条由错变对，净增 3。提升均未达到统计显著，但 SAM candidate 明显减少进入树搜索的样本：SAM-only 的 Tree 数下降 24.5%，both 下降 23.1%，mean pop 分别下降约 20.8% 和 21.7%。这说明少量困难恢复样本主要改善了搜索路由效率，而不是大幅提高最终答题准确率。

Qwen direct accuracy 为 288/624（46.15%）→285/624（45.67%），下降 0.48 pp，小于允许的 1 pp。SAM 负 prompt FPR@0.5 保持 baseline 0、candidate 0；伪框诊断 AP50 为 51.91%→53.03%，Recall@0.5 均为 50%，这些伪框指标不作正式晋升硬门。三个候选均通过门控；按 accuracy、mean pop、FPR 排序后最终选择 `both`。

### Fine 后 SAM trace-only 正式结果

trace-only 保持原 Fine 节点和 Qwen teacher 图不变，只在 Fine 成功 crop 上额外运行 SAM。999 条 train 轨迹中找到 38 个初始失败后恢复目标、44 个 mask 实例，来源为 `fine_probe_1=33`、`fine_probe_2=1`、`local_retry_1=4`。应用 mask 与面积过滤后形成 train 33 个 datapoint/37 个 bbox 实例、valid 22/27，分别是严格方案 datapoint 数的 11 倍。Qwen candidate 和 direct 结果复用严格实验；复用来源记录于 `opd/reuse_model.json`。

SAM LoRA 1 epoch 使用正负 `1:1`，约 47.13 秒。训练诊断已显示 valid FPR@0.5 为 22.73%，正式冻结 selectivity 评测确认 baseline FPR 0、candidate FPR 22.73%。冻结 valid 624 条结果：

| 组合 | correct / 624 | accuracy | Tree 数 | mean pop | 相对 baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 386 | 61.86% | 208 | 2.8269 | - |
| SAM-only | 382 | 61.22% | 71 | 1.1154 | -4 correct，Tree -137 |
| MLLM-only | 389 | 62.34% | 210 | 2.7853 | +3 correct，Tree +2 |
| both | 381 | 61.06% | 72 | 1.0753 | -5 correct，Tree -136 |

强制探测使 candidate 把 mode 1 数从 416 提高到 553，Tree 数下降 65.9%，但 SAM-only 有 35 条由对变错、31 条由错变对，净退化 4；both 净退化 5。它同时违反 `correct_not_lower` 和 `negative_fpr_not_higher` 两个硬门，SAM-only 与 both 均拒绝，最终只能选择复用的 MLLM-only。结论是：Fine 节点上 SAM 能输出 mask 并不等于该 mask 足以替代树搜索证据；不加质量筛选地全部当正监督会把“更容易出框”误当成“更会定位”，显著压低搜索成本但损害选择性与最终正确率。

### Fine 后 SAM replace 正式结果

replace 组于 2026-08-19 17:59 完成，`runs/full/round_00/.done`、`metrics.json` 和 `promotion.json` 均已落盘。该组不仅收集 Fine crop 上的 SAM mask，还在 SAM 有输出时用映射回原图的紧致框替代原 Fine tree node，因而 stock baseline 本身也使用 replace 推理策略；它的 baseline 383/624 不能与 off/trace-only 的 386/624 当作逐样本完全相同的控制组。

999 条 train 轨迹得到 41 个初始失败后恢复目标、47 个原始 mask 实例，来源为 `fine_probe_1=35`、`fine_probe_2=2`、`local_retry_1=4`。质量过滤后形成 train 36 个 datapoint/40 个 bbox 实例；valid 原始 20 个目标/21 个实例，过滤后为 19/20。所有 60 个训练/验证框均为 `sam_success_mask`，COCO 明确 `uses_dataset_gt_bbox=false`、`uses_tree_node_bbox=false`、`uses_mask_loss=false`。拒绝统计为：

- train：`missing_local_sam_recovery=170`、`initial_sam_success=79`、`pseudo_bbox_area_out_of_range=7`、`no_valid_recovered_mask=5`；
- valid：`missing_local_sam_recovery=104`、`initial_sam_success=27`、`pseudo_bbox_area_out_of_range=1`、`no_valid_recovered_mask=1`。

SAM LoRA 仍为 bbox-only、多实例 exhaustive 表示，train 负例为 verified generic 18 + random 18，与 36 个正查询保持 `1:1`；valid 19 条负例 coverage 为 100%。983,040 个参数训练 1 epoch 约 49.69 秒，train loss 为 33.7215。候选 SAM 在训练期 valid 上的负 prompt FPR 已达到 15.79%，正式 selectivity 复核结果相同。

replace 改变了 Qwen 实际视觉输入，因此没有复用严格组模型。586 条 VOPD 记录完成 6 个 optimizer step，累计 step 时间约 3 小时 25 分；`vopd_loss` 为 `0.05456, 0.05928, 0.04666, 0.04543, 0.04717, 0.04274`，对应 `grad_norm` 为 `23.25, 7.97, 5.31, 4.28, 6.13, 4.72`。loss 总体下降、梯度有限且没有 CUDA OOM。step 6 已保存并成功合并；其后的 Ray DataLoader worker `SIGKILL` 仍发生在清理阶段，不影响 checkpoint、评测或完成标志。

冻结 valid 624 条结果：

| 组合 | correct / 624 | accuracy | Tree 数 | mean pop | 相对本组 baseline |
|---|---:|---:|---:|---:|---:|
| baseline（stock SAM + replace policy） | 383 | 61.38% | 208 | 2.8269 | - |
| SAM-only | 383 | 61.38% | 68 | 1.1074 | correct 持平，Tree -140 |
| MLLM-only | 384 | 61.54% | 208 | 2.7147 | +1 correct，Tree 持平 |
| both | 387 | 62.02% | 68 | 1.0465 | +4 correct，Tree -140 |

`both` 的绝对 accuracy 最高，并将 Tree 数降低 67.3%、mean pop 降低 63.0%；但配对变化为 baseline-only correct 30 条、both-only correct 34 条，净增 4 的 95% CI 为 `[-1.92,+3.21]` pp，McNemar `p=0.7077`，没有统计显著。Qwen direct 为 288/624（46.15%）→287/624（45.99%），只下降 0.16 pp，direct 门通过。

正式 SAM selectivity 为：

| SAM | pseudo AP50（诊断） | pseudo Recall@0.5（诊断） | 负 prompt FPR@0.5 |
|---|---:|---:|---:|
| stock baseline | 32.43% | 52.63% | 0% |
| replace candidate | 30.00% | 52.63% | 15.79% |

因此 `sam_only` 和 `both` 尽管显著减少搜索成本，均因 FPR 上升而拒绝。`mllm_only` 增加 1 个 correct、direct 回归在限值内且不使用候选 SAM，成为唯一通过者；`promotion.json` 的最终选择为 `mllm_only`。这表明“Fine 成功后强制用 SAM 框作为最终视觉输入”在本轮可恢复一部分答案并大幅减少树搜索，但训练出的 SAM 仍过度出框，不能安全晋升。

### 当前所有 999/624 正式实验对比

下表汇总目前已完成且有 `.done` 的五次 round-0 正式实验。前两次属于历史数据构造对照，后三次共享“初始 SAM 失败后、局部 SAM 成功 mask 生成紧致 bbox”的主方案。`best raw` 是该次实验四组合中的最高原始 accuracy，不等于门控允许晋升。

| 实验 | SAM 正监督 | 推理策略 | baseline | best raw | SAM FPR（base→cand） | 门控选择 | 主要结论 |
|---|---|---|---:|---:|---:|---|---|
| 旧 GT-bbox/纠错过采样 | 数据集 GT bbox，839 train | 原流程 | 372/624，59.62% | SAM-only 389/624，62.34% | 14.67%→25.87% | MLLM-only | accuracy 提高，但 SAM FPR 恶化；仅作特殊 VOPD 对照 |
| bbox-free tree-node | 205 个 `tree_final_node` 弱框 | 原流程 | 386/624，61.86% | baseline 386/624，61.86% | 2.46%→0% | 不晋升 | candidate 令全部 624 条进入 Tree，SAM-only 退化至 332/624 |
| 严格局部恢复（off） | 3 个 mask-derived train datapoint | 原流程 | 386/624，61.86% | MLLM-only/both 389/624，62.34% | 0%→0% | **both** | 唯一同时保持 FPR、提高 correct 并减少 Tree 的 SAM 方案 |
| Fine probe trace-only | 33 个 mask-derived train datapoint | 只记录 probe，不替换 Fine 节点 | 386/624，61.86% | MLLM-only 389/624，62.34% | 0%→22.73% | MLLM-only | Tree 明显减少，但 SAM-only/both 分别净退化 4/5 |
| Fine SAM replace | 36 个 mask-derived train datapoint | Fine 有 SAM 输出即替换节点 | 383/624，61.38% | both 387/624，62.02% | 0%→15.79% | MLLM-only | both 净增 4、Tree -140，但 FPR 硬门失败 |

在后三个 mask-derived 实验中，严格局部恢复的 `both=389/624`、FPR 0%、Tree 160、mean pop 2.2147，是当前唯一可晋升的 SAM 更新。replace 改变了 baseline 推理策略，只能作绝对结果参考；trace-only 与 replace 共同证明：增加 Fine crop 上的 SAM 正样本会让模型更容易直接出框，从而把 Tree 数压到约 68～72，但未经质量筛选时会表现为 FPR 15.79%～22.73%，不能解释为可靠的定位能力提升。

本轮按用户要求在 replace 完成后停止，不再启动新变体。五个正式实验的 test 624 条均继续封存；当前结论只来自冻结 valid。

## Self-Evolve CVSearch 实验总览（精炼时间线）

以下为当前结论入口；上文保留实现细节和完整审计记录。

### 2026-08-17：旧 GT-bbox 对照

- 做法：999 train / 624 valid；SAM 使用 839 条 GT-bbox/纠错过采样数据，Qwen 使用旧 OPSD 过滤。
- 结果：baseline 372；SAM-only 389；MLLM-only 378；both 387。FPR 14.67%→25.87%。
- 结论：SAM accuracy 提高但错误出框增加，只晋升 MLLM；该实验依赖答案 bbox，不符合最终任务。
- 工程问题：Ray 结束时 DataLoader `SIGKILL`；checkpoint 已合并，判定为清理异常而非 OOM。

### 2026-08-18：无 GT bbox，tree node 弱伪框

- 修改：禁用 Quick Answer；VOPD 使用 mode 1/2 正确轨迹；SAM 使用 mode 2 的 `tree_final_node`，不读取 GT bbox。
- 结果：baseline 386；SAM-only 332；MLLM-only 385；both 327。candidate FPR 2.46%→0%，但 624/624 全部进入 Tree。
- 结论：tree node 是搜索窗口，不是目标紧致框；SAM 严重退化，全部拒绝。
- 修复：teacher 文件按组合隔离并升级 schema v4；评测限制 GPU 0–3，解决 GPU 4–7 被占用导致的 OOM；SAM evaluator 改 `num_workers=0`，解决 worker 卡死和 Unix socket 路径过长。

### 2026-08-18～19：严格局部 SAM 恢复

- 修改：仅采集 `initial SAM失败 → tree crop局部SAM首次成功`；用 mask 生成多实例紧致 bbox；不使用 GT/tree 框，不训练 mask decoder。
- 数据：train 3 datapoint/3 bbox，valid 2/2；正负 `1:1`。
- 结果：baseline 386；SAM-only 387；MLLM-only 389；both 389。both Tree 208→160，mean pop 2.8269→2.2147，FPR 0%→0%。
- 结论：`both` 通过全部门控，是当前唯一可晋升的 SAM 更新。
- 问题与修复：mask 最小面积 `0.001` 误删小目标，mask-derived bbox 单独改为 `0.0001`；Qwen 前台进程被外部终止，改为 systemd 托管、每步 checkpoint；无用 tokenizer 联网超时，改 `local_files_only=True`。

### 2026-08-19：Fine 后 SAM trace-only

- 修改：Fine 成功后额外运行 SAM并采集 mask，但不改变最终 Fine 节点或 Qwen 输入。
- 数据：train 33 datapoint/37 bbox，valid 22/27；Qwen 与严格组完全相同并复用。
- 结果：baseline 386；SAM-only 382；MLLM-only 389；both 381。SAM-only Tree 208→71，但 FPR 0%→22.73%。
- 结论：更多 Fine mask 使 SAM 更容易出框，却降低正确率和选择性；拒绝 SAM，只选 MLLM-only。

### 2026-08-19：Fine 后 SAM replace（最终变体）

- 修改：Fine crop 上 SAM 成功时，用紧致 SAM 框替换 tree node，并独立重训 SAM/Qwen。
- 数据：train 36 datapoint/40 bbox，valid 19/20；VOPD 586 条；正负 `1:1`。
- 训练：SAM 49.69 秒；Qwen 6 step，loss 0.05456→0.04274；无 CUDA OOM。
- 结果：本组 baseline 383；SAM-only 383；MLLM-only 384；both 387。both Tree 208→68，mean pop 2.8269→1.0465，但 FPR 0%→15.79%。direct 288→287。
- 结论：both 原始效果最好但 FPR 硬门失败；最终只晋升 MLLM-only。按要求在此停止，不再启动新变体。

### 总结

64/16 smoke 只用于验收链路，不用于模型结论：bbox-free tree-node 为 `baseline/SAM/MLLM/both=9/8/10/8`，direct `9→8`；严格恢复为 `11/11/12/12`，Tree `8/7/8/7`；trace-only 为 `11/11/12/12`，Tree `8/4/8/4`；replace 为 `11/11/12/12`，Tree `8/4/8/4`。后三组 direct 均为 `9→9`、FPR 均为 0；正式 624 条 valid 才暴露 trace/replace 的 FPR 退化。

| 方案 | best raw correct | candidate FPR | 最终选择 |
|---|---:|---:|---|
| GT-bbox 对照 | 389 | 25.87% | MLLM-only |
| tree-node 弱框 | 386 | 0% | 不晋升 |
| 严格局部恢复 | 389 | 0% | **both** |
| Fine trace-only | 389 | 22.73% | MLLM-only |
| Fine replace | 387 | 15.79% | MLLM-only |

共性工程修复：四组合 shard 改为按 tag 独立合并；缺失 log-prob 的诊断记录不再中断指标；teacher 文件按组合隔离；轨迹评测固定空闲 GPU 并并行分 shard；SAM evaluator 使用 `num_workers=0`；Qwen 训练改为 systemd 托管和逐 step checkpoint；离线 tokenizer 强制本地加载。已知 Ray DataLoader `SIGKILL` 均发生在最终 checkpoint 保存后的清理阶段，未出现训练期 CUDA OOM。

最终判断：训练 SAM 的有效增量应严格来自“原图失败、局部 crop 真正恢复成功”的 mask；直接扩充所有 Fine crop mask 会降低树搜索成本，但会把 FPR 推高。下一步若继续，应先做伪 mask 质量/语义一致性筛选或更强 hard negatives，而不是继续增加未经筛选的 Fine 正样本。
