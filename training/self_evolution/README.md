# CVSearch 驱动的 OPSD 自进化（报告 5.3）

该模块实现以问题为单位的可恢复自进化闭环：CVSearch 生成结构化搜索轨迹，筛出
回答正确且视觉证据有效的记录，分别构造 SAM3 纠错数据和 Region–Global OPSD
数据，训练候选模型，并在冻结验证集上决定是否晋级。

## 目录

| 目录/文件 | 内容 |
|---|---|
| `self_evolution.py` | 顶层命令入口 |
| `orchestrator/src/evolve/` | 数据冻结、轨迹生成、问题级筛选、训练、评测、晋级和报告 |
| `orchestrator/configs/` | 可移植正式配置与已运行的 smoke/full/消融配置 |
| `orchestrator/tests/` | 核心数据、路由、轨迹和晋级规则单测 |
| `cvsearch/` | CVSearch 搜索框架及其搜索时 SAM3 fork；该 fork 会额外暴露视觉 backbone features |
| `sam3_lora/` | 自进化实际使用的独立 SAM3 LoRA 训练源码、LoRA 注入、tokenizer 与权重合并器 |
| `sam_bridge/scripts/` | SAM3 LoRA 训练、负例构建、选择性评测和权重合并桥接脚本 |
| `scripts/download_assets.sh` | 从 ModelScope 下载 Qwen3-VL-2B、SAM3 和冻结数据 |
| `scripts/run_experiment.sh` | 统一运行初始展示、三种构造/训练及 valid 再测试 |

## 三组正式可复现配置

| 方法 | 配置 | SAM3 监督来源 |
|---|---|---|
| tree-node weak box | `repro_tree_node_weak_box.yaml` | 初始 SAM 失败后的唯一 `tree_final_node` |
| strict local recovery | `repro_strict_local_recovery.yaml` | `local_retry_1/2` 首次成功 mask 的紧致框 |
| fine trace-only | `repro_fine_trace_only.yaml` | Fine 探测 + local retry 的首次成功 mask；Qwen 仍使用原 Fine node |

三组均以 Qwen3-VL-2B-Instruct 和 stock SAM3 为 T0。正式发布包同时冻结各组已经
审过的负例，因此复现时不再运行历史使用的 8B 判别器；配置中的 2B 只作为负例缺失
时的回退，完整发布包不会走这条分支。fine trace-only 与历史受控实验一致，复用
strict 组训练出的 MLLM candidate，只隔离比较 SAM 数据与 SAM candidate。

## 问题级有效轨迹

正式无 GT bbox 版本只接收 `search_mode in {1, 2}`、最终回答正确且最终 MLLM
视觉输入有效的原问题；每个原问题最多生成一条 OPSD 记录。SAM 数据只展开首次
定位失败、最终可唯一归属到有效 node/mask 的目标。所有拒绝原因、图像哈希、模型
哈希和配置哈希都会写入轨迹。

## 使用

`cvsearch/` 包含搜索时 SAM3 fork，`sam3_lora/` 单独保留训练时 SAM3 runtime 和
LoRA 层，两者不共享已修改的 forward 接口；`sam_bridge/` 包含任务训练与合并入口；
OPSD 所需的 Vision-OPD fork 已提交在相邻的
`training/distillation/opsd/vision_opd/`。权重和冻结数据发布后可直接：

```bash
bash training/self_evolution/scripts/download_assets.sh

# 624 条 valid 上的 Qwen3-VL-2B + stock SAM3 初始能力展示
bash training/self_evolution/scripts/run_experiment.sh tree initial

# 单组：doctor / prepare / smoke / full / retest
bash training/self_evolution/scripts/run_experiment.sh tree full
bash training/self_evolution/scripts/run_experiment.sh strict full
bash training/self_evolution/scripts/run_experiment.sh fine full

# 依次运行三组；strict 会先于 fine 产生其受控复用的 MLLM
bash training/self_evolution/scripts/run_experiment.sh all full
bash training/self_evolution/scripts/run_experiment.sh all retest
```

默认 Conda 环境名为 `cvsearch`、`Sam3_lora`、`vision_opd`，可分别通过
`ORCHESTRATOR_CONDA_ENV`、`SAM_CONDA_ENV`、`OPD_CONDA_ENV` 覆盖。公开数据包只含
999 train 与 624 valid；test 仅发布不可反推答案的 sealed ID/hash，代码会拒绝在
`test_size=0` 的公开配置上运行 test evaluation。

如需手工运行，正式配置所需环境变量由 `run_experiment.sh` 自动设置；等价命令为：

```bash
CONFIG=training/self_evolution/orchestrator/configs/repro_tree_node_weak_box.yaml
python training/self_evolution/self_evolution.py --config "$CONFIG" doctor
python training/self_evolution/self_evolution.py --config "$CONFIG" prepare
python training/self_evolution/self_evolution.py --config "$CONFIG" smoke --round 0
python training/self_evolution/self_evolution.py --config "$CONFIG" \
  run-loop --stage full --max-rounds 1
```

旧 `vopd_qwen3vl_2b_*one_fifth.yaml` 保留已运行实验的原始机器路径，仅用于历史审计；
新机器必须使用三个 `repro_*.yaml`。

测试：

```bash
PYTHONPATH=training/self_evolution/orchestrator/src \
python -m pytest -q training/self_evolution/orchestrator/tests/test_core.py
```

当前 20 项测试覆盖 tree-node 与 mask-recovery 的独立分支、portable split、test
封存、已发布相对路径重定位、Fine trace policy、OPSD 构建、轨迹路由和晋级规则。补回的 tree-node 构建器
还已对历史正式轨迹逐条回放：train 为 205 图/205 框/83 拒绝，valid 为
122 图/122 框/34 拒绝，image 与 annotation JSON 均逐字段一致。

来源：CVSearch 基线 `liliupeng28/ICML26-CVSearch@a7e59b1209e09352c8aa86015a589ccc74646e98`；
SAM3-LoRA 基线 `Sompote/sam3_lora@e8e6724f71e7530278a83d3d286a7313086542f2`。当前工作区对
`CVSearch.py` 与 Qwen-VL adapter 的结构化轨迹改动已包含在此快照。
