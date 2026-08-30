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
| `cvsearch/` | CVSearch 搜索框架及自进化所需的结构化 artifact 接口 |
| `sam_bridge/scripts/` | SAM3 LoRA 训练、负例构建、选择性评测和权重合并桥接脚本 |

## 问题级有效轨迹

正式无 GT bbox 版本只接收 `search_mode in {1, 2}`、最终回答正确且最终 MLLM
视觉输入有效的原问题；每个原问题最多生成一条 OPSD 记录。SAM 数据只展开首次
定位失败、最终可唯一归属到有效 node/mask 的目标。所有拒绝原因、图像哈希、模型
哈希和配置哈希都会写入轨迹。

## 使用

`cvsearch/` 已包含 CVSearch、SAM3 runtime 和本实验所需的 LoRA 层；
`sam_bridge/` 已包含训练与合并入口。完整运行还需要一个完整 Vision-OPD checkout，
以及 MLLM/SAM3 权重、数据和冻结 split manifest。正式配置通过环境变量接收这些
机器相关路径：

```bash
export SELF_EVOLUTION_ROOT="$PWD/training/self_evolution"
export SELF_EVOLUTION_WORKSPACE=/path/to/run-workspace
export VISION_OPD_ROOT=/path/to/full/Vision-OPD
export VOPD_DATA_ROOT=/path/to/vopd_dataset
export VOPD_TRAIN_JSONL=/path/to/vopd_dataset/train.jsonl
export SAM_SPLIT_MANIFEST=/path/to/splits.json
export SAM_REFERENCE_ADAPTER=/path/to/reference_lora_weights.pt
export SAM_REFERENCE_CHECKPOINT=/path/to/reference_sam3.pt
export SAM_T0=/path/to/stock_sam3.pt
export MLLM_T0=/path/to/Qwen3-VL-2B-Instruct
export SPACY_MODEL=/path/to/spacy_model
export NEGATIVE_VERIFIER_MODEL=/path/to/Qwen3-VL-2B-Instruct
```

然后运行：

```bash
CONFIG=training/self_evolution/orchestrator/configs/vopd_qwen3vl_2b_bbox_free_one_fifth.yaml
python training/self_evolution/self_evolution.py --config "$CONFIG" doctor
python training/self_evolution/self_evolution.py --config "$CONFIG" prepare
python training/self_evolution/self_evolution.py --config "$CONFIG" smoke --round 0
python training/self_evolution/self_evolution.py --config "$CONFIG" \
  run-loop --stage full --max-rounds 3
```

其余配置保留已运行实验的原始机器路径，用于审计；新机器运行应以
`vopd_qwen3vl_2b_bbox_free_one_fifth.yaml` 为模板。

测试：

```bash
PYTHONPATH=training/self_evolution/orchestrator/src \
pytest -q training/self_evolution/orchestrator/tests/test_core.py
```

来源：CVSearch 基线 `liliupeng28/ICML26-CVSearch@a7e59b1209e09352c8aa86015a589ccc74646e98`；
SAM3-LoRA 基线 `e8e6724f71e7530278a83d3d286a7313086542f2`。当前工作区对
`CVSearch.py` 与 Qwen-VL adapter 的结构化轨迹改动已包含在此快照。
