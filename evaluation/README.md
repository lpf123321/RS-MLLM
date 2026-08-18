# RS-MLLM 评估框架

支持在 **VRSBench**、**MME-RealWorld-RS**、**XLRS-Bench-lite**、**LEVIR-CC** 四个数据集上对多模态遥感大模型进行标准化评估。

## 目录结构

```
evaluation/
├── base/                          # 策略接口
│   ├── metric.py                  #   BaseMetric — 评价指标接口
│   └── adapter.py                 #   BaseModelAdapter — 模型推理接口
├── metrics/                       # 指标实现（策略模式，可按需组合）
│   ├── accuracy.py                #   Accuracy — 精确匹配准确率
│   ├── bleu.py                    #   BLEU-1/2/3/4 — n-gram 精度
│   ├── rouge.py                   #   ROUGE-L — 最长公共子序列 F-measure
│   ├── cider.py                   #   CIDEr — TF-IDF 加权余弦相似度
│   └── referring.py               #   ReferringAcc — mean_iou / Acc@0.25~0.7
├── adapters/                      # 模型适配器（策略模式，支持不同模型）
│   └── qwen3vl.py                 #   Qwen3VLAdapter — Qwen3.5 推理适配器
├── evalsets/                      # 数据集加载
│   ├── vrsbench.py                #   VRSBench (VQA / Caption / Referring)
│   ├── mme.py                     #   MME-RealWorld-RS (VQA)
│   ├── xlrs.py                    #   XLRS-Bench-lite (VQA)
│   └── levircc.py                 #   LEVIR-CC (变化描述, 多参考)
├── main.py                        # 入口
├── run_eval.sh                    # 交互式运行脚本
├── run_eval.slurm                 # SLURM 集群提交脚本
└── README.md                      # 本文件
```

## 数据集与指标对照

| 数据集 | 子任务 | 指标 |
|--------|--------|------|
| **VRSBench** | VQA | Accuracy |
| | Caption | CIDEr, ROUGE-L, BLEU-4 |
| | Referring | Acc@0.5, mean_iou, Acc@0.25, Acc@0.7 |
| **MME-RealWorld-RS** | VQA | Accuracy |
| **XLRS-Bench-lite** | VQA | Accuracy |
| **LEVIR-CC** | 变化描述 (多参考) | CIDEr, ROUGE-L, BLEU-4 |

## 使用方法

### 1. 准备环境

```bash
conda activate rs_mllm
cd RS-MLLM
```

### 2. 运行评估

**交互式运行**（需已在 conda 环境内）:

```bash
# 评估全部数据集（每数据集最多 100 条）
bash evaluation/run_eval.sh

# 评估全部数据集（全部样本）
bash evaluation/run_eval.sh --max_samples 0

# 只评估指定数据集
bash evaluation/run_eval.sh --datasets vrsbench levircc

# 自定义模型和数据量
bash evaluation/run_eval.sh --model_path /path/to/model --max_samples 500
```

**SLURM 集群提交**:

```bash
# 默认参数提交
sbatch evaluation/run_eval.slurm

# 指定数据集和样本数
sbatch evaluation/run_eval.slurm --datasets vrsbench mme --max_samples 500

# 自定义模型路径和输出
sbatch evaluation/run_eval.slurm --model_path /path/to/model --output /path/to/results.json
```

### 固定 XLRS 小样本集

`evaluation/data/xlrs_memory_1000.jsonl` 是已提交的固定 1,000 条 XLRS 小样本集，用于所有显存评测。它按原始 XLRS 文件的既定选择顺序保存，因此不同使用者不需要重新抽样即可使用同一批数据。

```bash
sbatch evaluation/run_xlrs_memory.slurm
```

脚本默认读取该固定文件。其来源文件 SHA-256、选择 seed 和全部原始行索引保存在 `evaluation/data/xlrs_memory_1000.metadata.json`；如需验证或重建，可运行 `evaluation/tools/create_xlrs_memory_subset.py`。

### 3. 查看结果

- 终端输出结果表格
- JSON 结果默认保存到 `evaluation/results.json`
- SLURM 日志输出到 `evaluation/logs/eval_<job_id>.out`

### 4. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `models/Qwen3.5-4B` | 模型路径 |
| `--datasets` | `all` | 评估数据集，可选 `vrsbench` `mme` `xlrs` `levircc` `all` |
| `--max_samples` | `100` (交互) / `0` (SLURM) | 每数据集最大样本数，`0`=全部 |
| `--device` | `cuda` | 推理设备 |
| `--output` | `evaluation/results.json` | 结果 JSON 路径 |

## 扩展指南

### 添加新指标

在 `metrics/` 下新建文件，继承 `BaseMetric` 并实现 `compute` 方法：

```python
from evaluation.base.metric import BaseMetric

class MyMetric(BaseMetric):
    name = "my_metric"

    def compute(self, references, predictions):
        # references: List[List[str]], 每个样本可能有多条参考
        # predictions: List[str]
        return {"MyMetric": 0.0}
```

在 `metrics/__init__.py` 中注册即可。

### 添加新模型

在 `adapters/` 下新建文件，继承 `BaseModelAdapter`：

```python
from evaluation.base.adapter import BaseModelAdapter

class MyModelAdapter(BaseModelAdapter):
    def generate(self, images, prompt):
        # images: List[str] — 图片路径列表
        # prompt: str — 文本指令
        return "model output"
```
