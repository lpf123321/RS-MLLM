# ModelScope 复现实验资产清单

本文档冻结报告第 5 章实验所需的外部资产边界。Git 仓库只提交代码、配置、
小型清单和校验器；训练数据、生成轨迹及模型权重通过 ModelScope 下载。

> 状态：命名空间与登录身份已确认是 `Uchitachi`。五个资产仓库均已上传并回读
> manifest 校验。Expert Checkpoints 的远端/本地 `ASSET_MANIFEST.json` SHA-256 为
> `514343aa3a2496446be665aeeea01939e9cd20cf23066caf312e8b68e33c3d72`，
> `SHA256SUMS` 为
> `1421145aac4ead322bb550d95ccea5fd47638499216f52808155db499d5e1eab`。
> Distillation Data 是不含上游原图的薄包，并已包含结构化生成审计/展示包。

## 建议的 ModelScope 仓库

| 类型 | 建议名称 | 内容 |
|---|---|---|
| Dataset | `Uchitachi/RS-MLLM-Distillation-Data` | General/Grounding Expert 冻结标注、图片索引与数据生成展示（不重复上传官方原图） |
| Dataset | `Uchitachi/RS-MLLM-Self-Evolution-Data` | Vision-OPD 固定切分、CVSearch 轨迹、OPSD 与 SAM3 训练样本 |
| Model | `Uchitachi/RS-MLLM-Expert-Checkpoints` | General/Grounding Expert delta 与最终 LoRA |
| Model | `Uchitachi/RS-MLLM-Distillation-Checkpoints` | SFT、OPD 与冻结轨迹对照 LoRA |
| Model | `Uchitachi/RS-MLLM-Self-Evolution-Checkpoints` | 三组自进化实验的 MLLM/SAM3 候选权重 |

基础模型不重复上传。Qwen3-VL-2B-Instruct 和 stock SAM3 直接使用官方
ModelScope 仓库；Qwen3.5-4B 基座同样优先引用官方仓库。

已核实的官方 ID：

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3-VL-2B-Instruct`
- `facebook/sam3`

当前五个仓库必须先保持私有；下载与上传均使用官方 `modelscope-hub` 客户端的交互式
`ms-hub login`，凭据只保存在用户机器；仓库和命令中都不写 token。统一创建/上传
入口为 `training/scripts/upload_modelscope_assets.sh`。

## `RS-MLLM-Distillation-Data`

### General Expert（有效配置 Exp3）

薄包必须上传：

| 资产 | 样本数 | 标注文件体积 | 图像依赖 |
|---|---:|---:|---|
| `general/exp3_mme3736_xlrs3080.json` | 6,816 | 2.7 MiB | `IMAGE_INDEX.jsonl` 引用官方 MME/XLRS 图像 |
| `ASSET_MANIFEST.json` | - | 很小 | 数据来源、样本数、冻结 JSON SHA-256 |
| `IMAGE_INDEX.jsonl` | 23,086 个唯一 blob | 数 MiB | 上游仓库、上游路径、字节数和 SHA-256 |

`exp3_mme3736_xlrs3080.json` 是当前 General Expert 效果提升实验实际使用的
冻结训练集。图片由官方仓库拉取后按哈希物化，不在本仓库重复分发。

### Grounding Expert（三组配置全部保留）

共同分量：

| 资产 | 样本数 | 标注文件体积 | 唯一图像数 |
|---|---:|---:|---:|
| `grounding/bootstrap_vrsnew942.json` | 942 | 360 KiB | 700 |
| `grounding/exp1_washed_vrs36285.json` | 36,285 | 15 MiB | 20,211 |
| `grounding/exp4_xlrs6611.json` | 6,611 | 6.3 MiB | 1,023 |
| `grounding/exp5_all43838.json` | 43,838 | 21 MiB | 21,234 |

三组实验数据：

| 配置 | 初始化 | 本轮训练数据 | 冻结数据文件 |
|---|---|---|---|
| Exp1 | `vrs_new_942` LoRA | washed VRSBench 36,285 | `grounding/exp1_washed_vrs36285.json` |
| Exp4 | `vrs_new_942` LoRA | XLRS official 6,308 + XLRS-new 303 | `grounding/exp4_xlrs6611.json` |
| Exp5 | `vrs_new_942` LoRA | washed VRS 36,285 + VRS-new 942 + XLRS 6,611 | `grounding/exp5_all43838.json` |

Exp4 的组合文件约 6.3 MiB；Exp5 的组合文件约 21 MiB。Exp1/Exp4/Exp5 的
拼接 manifest、样本来源字段和路径改写表必须一起上传。

### 数据生成展示

以下是生成流程的展示/审计数据，不与训练分量重复存储图像：

- VRSBench VQA/Referring 展示：2,063 条，其中 VQA 1,241、Referring/
  spatial reasoning 526、caption 296；引用 296 张图像，约 0.121 GiB。
- MME-new 展示：只保留冻结的 3,736 条最终有效训练输出；历史输入/audit manifest
  已缺失，不伪造补齐。
- XLRS-new 展示：冻结的 3,080 条有效输出、style profile、accepted/rejected
  sidecar 与 smoke-review 报告。
- VRSBench-new Grounding 展示：942 条、SAM3 检测审计与 review gallery。
- XLRS Grounding-new 展示：303 条、SAM3 检测审计与 review gallery。

实际发布的 `generation/` 子目录只包含结构化审计，不重复上传官方原图或历史 review
图片：VRSBench VQA/Referring/Caption 分别为 1,241/526/296 条；XLRS-new 保留 800 图、
3,080 条的 source/question/style/accepted/audit；VRSBench-new 保留 7,552→942 的候选与
过滤链；XLRS Grounding-new 保留 1,975→303 的链路。复现者下载官方图片后，可用仓库内
`sam3_pipeline` finalize 命令重建 gallery。OPSD 展示放在 Self-Evolution Data，避免重复。

MME 有效的 3,736 条版本只剩最终冻结训练 JSON，原 source/question/style/audit 中间件已
不在历史路径，因此不伪造其生成展示；项目中的 `MME-new=3736` 是对保留有效实验的命名
约定。

生成脚本采用 `codex login` 后的 `codex exec`，不读取
`OPENAI_API_KEY`。冻结输出用于复现实验；重新生成属于流程复现，远端模型版本变化
时不承诺逐字节一致，因此必须记录 Codex 版本、模型、prompt、JSON schema 和
生成时间。

## `RS-MLLM-Self-Evolution-Data`

### 固定输入与切分

| 资产 | 数量 | 约占空间 | 说明 |
|---|---:|---:|---|
| `splits/train.jsonl` + 原图 | 999 | 0.937 GiB | 三种方法共同的数据构造输入 |
| `splits/valid.jsonl` + 原图 | 624 | 0.574 GiB | 初始能力展示与候选复测 |
| `splits/test_ids.sealed.json` | 624 个 ID | 很小 | answer-independent sealed ID；不发布或使用 test 答案 |

不上传完整 71 GiB 的本地 Vision-OPD 工作目录。发布包只保留本实验用到的
train/valid 原图和 portable split；源数据许可证为 Apache-2.0。

注意：历史 schema 的 `sample_id` 将答案放入哈希 preimage，不能直接作为 sealed
test ID 发布，否则可对 A/B/C/D 穷举反推。staging 脚本会改用不包含答案的
`sealed_id`，且不上传 test 问题、选项、原图或标签。

### 三种数据构造方法

三种方法都产生 583 条语义相同的 OPSD 记录；当前三个 Parquet 的差异只是本机
绝对路径和 trace provenance。发布时存一份规范化的公共 OPSD 数据，并为每种方法
保留各自 manifest。

| 方法 | OPSD | SAM3 正样本 | 必须上传的生成资产 |
|---|---:|---:|---|
| tree-node weak box | 583 | 205 图 / 205 框 | 轨迹、规范化 OPSD、COCO、负例/拒绝清单 |
| strict local recovery | 583 | 3 图 / 3 框 | 轨迹、规范化 OPSD、COCO、恢复审计 |
| fine trace-only | 583 | 33 图 / 37 框 | 轨迹、规范化 OPSD、COCO、trace-only 审计 |

OPSD 训练所引用的 583 张原图约 544.1 MiB（已包含在 999 张 train 原图中），
583 张 teacher crop 约 331.2 MiB。三个实验的 teacher crop 内容完全一致，只上传
一份。三份 SAM COCO 正样本连同图像合计不到 5 MiB。

还应上传：

- 三种方法的 canonical `train.jsonl`、`valid*.jsonl` 轨迹（不上传重复 shard）。
- 初始 Qwen3-VL-2B + SAM3 在 624 条 valid 上的预测、搜索统计和汇总指标。
- baseline、SAM-only、MLLM-only、both 的复测指标与 promotion decision。
- 构造配置、模型/配置 hash、拒绝原因统计和数据完整性报告。

SAM mask 中间结果可作为可选 audit bundle；训练所需的最终 COCO 已经包含图像和框，
因此 mask 缓存不是必需下载项。

## 图像许可证与可见性策略

- **XLRS：** 图像来源许可混合，Toronto 图像禁止向第三方再分发，所以只发布索引，
  复现者从官方仓库下载。
- **MME：** 官方条款限定学术用途，且未经许可不得分发、发布或复制全部/部分数据；
  脚本要求复现者明确确认条款后才从官方仓库下载。
- **VRSBench：** 文本为 CC BY 4.0；部分 DOTA 图像仅限学术用途。原图仍由官方仓库提供。
- **Vision-OPD 实验子集可上传。** 其本地数据卡为 Apache-2.0，只打包本实验使用的
  999 train + 624 valid 原图。

## 权重不是 Dataset 资产

权重放入 Model 类型仓库，不混进训练数据：

- General/Grounding full-rank delta：各约 8.5 GiB。
- General Exp3、Grounding bootstrap-942、Exp1、Exp4、Exp5 LoRA：各约 267 MiB。
- SAM3 自进化只上传每组约 4 MiB 的 LoRA，不上传 3.3 GiB merged checkpoint。
- FSDP raw checkpoint、optimizer state、缓存和临时 smoke 模型不上传。
- MLLM 自进化训练并非 LoRA；上传两个约 4.9 GB 的合并候选，排除 raw FSDP 与
  optimizer shards。Fine trace-only 按历史控制实验复用 strict 的 MLLM 候选。

## 永不上传的本地内容

- `cache/`、Hugging Face/vLLM cache、`__pycache__`、Ray 临时目录。
- optimizer state、重复 rank shard、重复 smoke/full 模型、merged 基座模型。
- 重复的 trajectory shard；只保留 canonical 合并文件。
- 本机绝对路径、账户信息、Codex 登录凭据、ModelScope token。
- test 数据生成结果或任何把 test 答案用于训练/筛选的产物。
