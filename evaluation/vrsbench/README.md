# VRSBench 评估脚本使用说明

## 快速开始

```bash
cd evaluation/vrsbench
bash run_vrsbench.sh
```

所有配置项都在 `run_vrsbench.sh` 顶部的 **可配置项** 区域，修改后直接运行即可，无需命令行参数。

## 三个评估任务

VRSBench 数据集包含三种遥感图像理解任务：

| 任务 | JSON 文件 | 说明 |
|------|----------|------|
| **referring** | `VRSBench_EVAL_referring.json` | 目标定位：根据文字描述，在图像中定位目标，输出边界框坐标 |
| **caption** | `VRSBench_EVAL_Cap.json` | 图像描述：根据图像生成自然语言描述 |
| **vqa** | `VRSBench_EVAL_vqa.json` | 视觉问答：根据图像回答问题 |

### 各任务评测指标

| 任务 | 指标 | 含义 |
|------|------|------|
| **referring** | `mean_iou` | 所有样本预测框与真实框的平均 IoU（交并比），解析失败的样本视为 0 |
| | `Acc@0.25` | IoU ≥ 0.25 的样本占比 |
| | `Acc@0.5` | IoU ≥ 0.5 的样本占比 |
| | `Acc@0.7` | IoU ≥ 0.7 的样本占比 |
| **caption** | `BLEU-1/2/3/4` | 1~4 元语法的命中精度（corpus-level） |
| | `ROUGE-L` | 最长公共子序列的 F-measure |
| | `METEOR` | 基于精确匹配 + 词干匹配 + WordNet 同义词匹配的 F-mean |
| | `CIDEr-D` | 基于 TF-IDF 加权的 n-gram 余弦相似度，带高斯长度惩罚 |
| **vqa** | `Accuracy` | 预测答案与标准答案的精确匹配准确率 |

## 配置项说明

所有可修改的配置集中在 `run_vrsbench.sh` 第 17–49 行。以下逐一说明：

### 1. 模型路径 (`MODEL_PATH`)

```bash
# 修改这一行即可切换模型。支持相对路径（相对于 RS-MLLM 根目录）和绝对路径。
MODEL_PATH="$REPO_ROOT/finetune_framework/VRSbench/output/merged_model"
```

目前已准备好的可选模型：

| 模型 | 路径 | 说明 |
|------|------|------|
| 官方预训练 | `$REPO_ROOT/models/Qwen3-VL-2B-Instruct` | Qwen3-VL 官方权重 |
| 全量微调合并 | `$REPO_ROOT/finetune_framework/VRSbench/output/merged_model` | 全量 SFT 后合并的模型（暂未上传） |


### 2. 测试集路径 (`TEST_DATA_DIR`)

```bash
TEST_DATA_DIR="/users/u2024311136/shared/shared_datasets/VRSBench"
```

该目录需包含以下结构：

```
VRSBench/
├── VRSBench_EVAL_referring.json    # referring 任务测试数据
├── VRSBench_EVAL_Cap.json          # caption 任务测试数据
├── VRSBench_EVAL_vqa.json          # vqa 任务测试数据
└── images/
    └── val/                         # 测试集图像
        ├── xxx.jpg
        └── ...
```

### 3. 评估任务选择 (`TASKS`)

```bash
# 可选值: all | referring | caption | vqa
TASKS="all"            # 三个任务都跑
# TASKS="referring"    # 只跑目标定位
# TASKS="caption"      # 只跑图像描述
# TASKS="vqa"          # 只跑视觉问答
```

### 4. 样本数量 (`MAX_SAMPLES`)

```bash
MAX_SAMPLES=0          # 0 = 使用全部测试样本
# MAX_SAMPLES=100      # 调试时建议设为 50~100，快速验证
```

### 5. Prompt 模板开关 (`USE_PROMPT_TEMPLATE`)

```bash
USE_PROMPT_TEMPLATE="true"    # 使用训练时的 prompt 模板
# USE_PROMPT_TEMPLATE="false" # 使用数据集原始问题
```

设为 `"true"` 时，每个任务的问题会被包装为：

| 任务 | 包装后的 prompt |
|------|----------------|
| caption | `<image>\n[caption] Could you describe the contents of this image for me?` |
| vqa | `<image>\n[vqa] {原始问题}` |
| referring | `<image>\n[refer] could you tell me the location for <p>{原始问题}</p>?` |

这些模板定义在 `test_vrsbench.py` 的 `PROMPT_TEMPLATES` 字典中，可按需修改。

### 6. 输出文件路径 (`OUTPUT`)

```bash
OUTPUT=""              # 留空 → 默认保存到脚本同目录下的 vrsbench_results.json
# OUTPUT="./exp1.json" # 自定义路径
```

输出 JSON 格式示例：

```json
{
  "referring": {
    "mean_iou": 0.5123,
    "Acc@0.25": 0.6500,
    "Acc@0.5": 0.4200,
    "Acc@0.7": 0.2300,
    "samples": 100
  },
  "caption": {
    "BLEU-1": 0.3456,
    "BLEU-2": 0.2345,
    "BLEU-3": 0.1567,
    "BLEU-4": 0.0987,
    "ROUGE-L": 0.3456,
    "METEOR": 0.2345,
    "CIDEr-D": 1.2345,
    "samples": 100
  },
  "vqa": {
    "Accuracy": 0.7500,
    "samples": 100
  }
}
```

### 7. 其他配置

```bash
BATCH_SIZE=1           # 推理 batch size（多模态模型建议保持 1）
DEVICE="cuda"          # 推理设备: cuda | cpu
```

## 命令行参数覆盖

脚本中的配置也可以通过命令行参数临时覆盖（会覆盖脚本内的默认值）：

```bash
bash run_vrsbench.sh --model_path /path/to/other/model --max_samples 200
bash run_vrsbench.sh --no_prompt_template           # 临时关掉 prompt 模板
bash run_vrsbench.sh --tasks caption                # 临时只跑 caption
```

## 环境依赖

```bash
pip install nltk rouge_score torch transformers qwen-vl-utils tqdm numpy

# 首次使用还需下载 NLTK 数据（METEOR 指标依赖 WordNet）
python -m nltk.downloader wordnet punkt punkt_tab
```
