"""
Dataset Structure Analysis Script
Analyzes all 4 remote sensing datasets and outputs a markdown report.
"""

import json
import os
from rsmllm.config import DATA_ROOT
from collections import Counter
from pathlib import Path

from PIL import Image


DATA_ROOT = Path("$DATA_ROOT")
REPORT_PATH = Path("REPO_ROOT/data/dataset_analysis_report.md")


def _image_resolution_counter(image_paths, max_samples=200):
    """Read image resolutions from a list of paths, return Counter of (w,h)."""
    counter = Counter()
    for i, p in enumerate(image_paths):
        if i >= max_samples:
            break
        try:
            with Image.open(p) as img:
                counter[img.size] += 1
        except Exception:
            pass
    return counter


def analyze_levir_cc():
    print("=" * 60)
    print("[1/4] Analyzing LEVIR-CC (Change Captioning)...")
    print("=" * 60)

    root = DATA_ROOT / "LEVIR-CC"
    ann_path = root / "LevirCCcaptions.json"

    with open(ann_path) as f:
        data = json.load(f)

    images = data["images"]
    total_pairs = len(images)
    all_sentences = []
    split_counts = Counter()
    changeflag_counts = Counter()
    sent_lengths = []

    for img in images:
        split_counts[img["split"]] += 1
        changeflag_counts[img["changeflag"]] += 1
        for s in img["sentences"]:
            all_sentences.append(s["raw"])
            sent_lengths.append(len(s["tokens"]))

    img_dirs = {
        "train": root / "images" / "train",
        "val": root / "images" / "val",
        "test": root / "images" / "test",
    }
    actual_images = {}
    resolution_samples = []
    for split, d in img_dirs.items():
        dir_a = d / "A"
        dir_b = d / "B"
        a_count = len(list(dir_a.glob("*.png"))) if dir_a.exists() else 0
        b_count = len(list(dir_b.glob("*.png"))) if dir_b.exists() else 0
        actual_images[split] = {"A": a_count, "B": b_count}
        if a_count > 0:
            resolution_samples.extend(sorted(dir_a.glob("*.png"))[:50])
        if b_count > 0:
            resolution_samples.extend(sorted(dir_b.glob("*.png"))[:50])

    res_counter = _image_resolution_counter(resolution_samples)
    res_str = "×".join(map(str, list(res_counter.keys())[0])) if len(res_counter) == 1 else \
        ", ".join(f"{w}×{h}" for (w, h), c in res_counter.most_common(5))

    report = f"""## LEVIR-CC (遥感变化描述)

**路径**: `{root}`

### 数据集概览

| 指标 | 数值 |
|------|------|
| 图像对总数 (标注) | {total_pairs} |
| 总描述句子数 | {len(all_sentences)} |
| 每图像对描述数 | 5 |

### 数据集划分

| Split | 标注对数 | 图像 A (变化前) | 图像 B (变化后) |
|-------|---------|----------------|----------------|
| train | {split_counts.get('train', 0)} | {actual_images.get('train', {}).get('A', 0)} | {actual_images.get('train', {}).get('B', 0)} |
| val   | {split_counts.get('val', 0)} | {actual_images.get('val', {}).get('A', 0)} | {actual_images.get('val', {}).get('B', 0)} |
| test  | {split_counts.get('test', 0)} | {actual_images.get('test', {}).get('A', 0)} | {actual_images.get('test', {}).get('B', 0)} |

### 变化标志分布

| changeflag | 含义 | 数量 | 占比 |
|-----------|------|------|------|
| 0 | 无变化 | {changeflag_counts.get(0, 0)} | {changeflag_counts.get(0, 0)/total_pairs*100:.1f}% |
| 1 | 有变化 | {changeflag_counts.get(1, 0)} | {changeflag_counts.get(1, 0)/total_pairs*100:.1f}% |

### 描述文本统计

| 指标 | 数值 |
|------|------|
| 平均描述长度 (tokens) | {sum(sent_lengths)/len(sent_lengths):.1f} |
| 最短描述 (tokens) | {min(sent_lengths)} |
| 最长描述 (tokens) | {max(sent_lengths)} |

### 标注结构示例

```json
{json.dumps(images[0], indent=2, ensure_ascii=False)[:800]}
```

### 图像分辨率

| 指标 | 数值 |
|------|------|
| 采样数 | {sum(res_counter.values())} |
| 分辨率种类 | {len(res_counter)} |
| 分辨率明细 | {res_str} |

### 任务格式

- **输入**: 两张图像 (A: 变化前, B: 变化后) + 指令 "请描述两时相图像之间的变化"
- **输出**: 描述两时相之间的变化 (5句之一或综合)
- **图像格式**: PNG

"""
    return report


def analyze_mme_real_rs():
    print("=" * 60)
    print("[2/4] Analyzing MME-RealWorld-RS (VQA)...")
    print("=" * 60)

    root = DATA_ROOT / "MME-RealWorld-RS"
    ann_path = root / "MME_RealWorld.json"

    with open(ann_path) as f:
        data = json.load(f)

    all_items = data if isinstance(data, list) else list(data.values())
    total = len(all_items)

    subtask_counts = Counter()
    category_counts = Counter()
    question_type_counts = Counter()
    task_counts = Counter()
    rs_items = []

    for item in all_items:
        subtask = item.get("Subtask", "N/A")
        subtask_counts[subtask] += 1
        category_counts[item.get("Category", "N/A")] += 1
        question_type_counts[item.get("Question Type", "N/A")] += 1
        task_counts[item.get("Task", "N/A")] += 1
        if subtask == "Remote Sensing":
            rs_items.append(item)

    img_dir = root / "remote_sensing"
    actual_images = list(img_dir.glob("*.png")) if img_dir.exists() else []

    rs_categories = Counter(item.get("Category", "N/A") for item in rs_items)
    rs_types = Counter(item.get("Question Type", "N/A") for item in rs_items)

    sample_rs = [item for item in rs_items if item.get("Category") == "count"]
    sample_item = sample_rs[0] if sample_rs else (rs_items[0] if rs_items else {})

    # Resolution analysis
    res_counter = _image_resolution_counter(actual_images, max_samples=295)

    # Build resolution string grouped by similarity
    from collections import defaultdict
    res_groups = defaultdict(int)
    for (w, h), c in res_counter.most_common():
        key = f"{w}×{h}"
        res_groups[key] = c
    res_detail = ", ".join(f"{k} ({v}张)" for k, v in res_groups.items())

    report = f"""## MME-RealWorld-RS (遥感多模态评测)

**路径**: `{root}`

### 数据集概览

| 指标 | 数值 |
|------|------|
| 总问题数 (全部任务) | {total} |
| 遥感子任务问题数 | {len(rs_items)} |
| 遥感图像数量 (实际文件) | {len(actual_images)} |

### 全部任务分布

| Subtask | 数量 |
|---------|------|
"""

    for subtask, cnt in subtask_counts.most_common():
        report += f"| {subtask} | {cnt} |\n"

    report += f"""
### 遥感子任务详情

#### 问题类型分布

| Category | 数量 |
|----------|------|
"""
    for cat, cnt in rs_categories.most_common():
        report += f"| {cat} | {cnt} |\n"

    report += """
#### 问题格式分布

| Question Type | 数量 |
|---------------|------|
"""
    for qt, cnt in rs_types.most_common():
        report += f"| {qt} | {cnt} |\n"

    report += f"""
### 标注结构示例 (遥感Count任务)

```json
{json.dumps(sample_item, indent=2, ensure_ascii=False)}
```

### 图像分辨率

| 指标 | 数值 |
|------|------|
| 图像总数 | {len(actual_images)} |
| 分辨率种类 | {len(res_counter)} |
| 分辨率明细 | {res_detail} |

### 任务格式

- **输入**: 一张遥感图像 + 选择题问题
- **输出**: 选择题答案 (A/B/C/D/E)
- **所有问题均为多项选择题**
- **图像**: 高分辨率遥感图像 (DOTA-v2 和 Toronto 区域)

"""
    return report


def analyze_vrsbench():
    print("=" * 60)
    print("[3/4] Analyzing VRSBench (对话/VQA/Refer/Caption)...")
    print("=" * 60)

    root = DATA_ROOT / "VRSBench"

    img_train_dir = root / "images" / "Images_train"
    img_val_dir = root / "images" / "val"
    train_imgs = list(img_train_dir.glob("*.png")) if img_train_dir.exists() else []
    val_imgs = list(img_val_dir.glob("*.png")) if img_val_dir.exists() else []

    # Resolution analysis
    res_samples = []
    res_samples.extend(sorted(train_imgs)[:100])
    res_samples.extend(sorted(val_imgs)[:100])
    res_counter = _image_resolution_counter(res_samples)
    res_str = "×".join(map(str, list(res_counter.keys())[0])) if len(res_counter) == 1 else \
        ", ".join(f"{w}×{h}" for (w, h), c in res_counter.most_common(5))

    report = f"""## VRSBench (遥感多模态对话/VQA/Referring/Caption)

**路径**: `{root}`

### 数据集概览

| 指标 | 数值 |
|------|------|
| 训练图像 | {len(train_imgs)} |
| 验证图像 | {len(val_imgs)} |
| 总计图像 | {len(train_imgs) + len(val_imgs)} |

"""

    # Train conversations
    train_path = root / "VRSBench_train.json"
    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        train_entries = len(train_data)
        conv_turns = []
        conv_lengths = []
        for entry in train_data:
            convs = entry.get("conversations", [])
            conv_turns.append(len(convs) // 2)
            for c in convs:
                conv_lengths.append(len(c.get("value", "").split()))

        unique_train_images = len(set(e.get("image", "") for e in train_data if "image" in e))

        report += f"""### 训练对话数据

| 指标 | 数值 |
|------|------|
| 总对话条目数 | {train_entries} |
| 唯一图像数 | {unique_train_images} |
| 平均对话轮次 | {sum(conv_turns)/len(conv_turns):.1f} |
| 平均文本长度 (words) | {sum(conv_lengths)/len(conv_lengths):.1f} |

"""
        # Sample train entry
        report += f"""### 训练标注示例
```json
{json.dumps(train_data[0], indent=2, ensure_ascii=False)[:600]}
```
"""

    # EVAL files
    eval_configs = [
        ("VQA", "VRSBench_EVAL_vqa.json"),
        ("Referring", "VRSBench_EVAL_referring.json"),
        ("Caption", "VRSBench_EVAL_Cap.json"),
    ]

    for eval_name, eval_file in eval_configs:
        eval_path = root / eval_file
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            eval_entries = len(eval_data)

            # Collect types
            type_counter = Counter()
            if isinstance(eval_data, list) and len(eval_data) > 0:
                if "type" in eval_data[0]:
                    for item in eval_data:
                        type_counter[item.get("type", "N/A")] += 1

            report += f"""### EVAL {eval_name}

| 指标 | 数值 |
|------|------|
| 条目数 | {eval_entries} |
"""

            if type_counter:
                report += "| 子类型分布 | "
                for t, c in type_counter.most_common():
                    report += f"{t}: {c} "
                report += "|\n"

            report += f"""
**示例**:
```json
{json.dumps(eval_data[0], indent=2, ensure_ascii=False)[:500]}
```
"""

    report += f"""
### 任务格式总结

| 任务 | 输入 | 输出 |
|------|------|------|
| Caption | 图像 + "描述图像内容" | 自然语言描述 |
| VQA | 图像 + 自然语言问题 | 答案短语 |
| Referring | 图像 + 描述性短语 | 边界框坐标 |

### 图像分辨率

| 指标 | 数值 |
|------|------|
| 采样数 | {sum(res_counter.values())} |
| 分辨率种类 | {len(res_counter)} |
| 分辨率明细 | {res_str} |

### 图像

- 来源: GoogleEarth
- 格式: PNG
- 路径: `images/Images_train/` 和 `images/val/`

"""
    return report


def analyze_xlrs_bench():
    print("=" * 60)
    print("[4/4] Analyzing XLRS-Bench-lite (多选VQA)...")
    print("=" * 60)

    root = DATA_ROOT / "XLRS-Bench-lite"

    report = f"""## XLRS-Bench-lite (遥感多选题评测)

**路径**: `{root}`

### 数据集概览

| 指标 | 数值 |
|------|------|
| 格式 | HuggingFace Datasets (Arrow) |
| 总样本数 | 3,080 |
| 划分 | train only |

### 特征信息

根据 `dataset_info.json`:

| 字段 | 类型 | 说明 |
|------|------|------|
| path | string | 图像路径引用 |
| index | int32 | 样本索引 |
| question | string | 问题文本 |
| multi-choice options | list[string] | 多选题选项 |
| answer | string | 标准答案 |
| category | string | L1 任务类别 |
| l2-category | string | L2 子类别 |
| image | Image (Sequence) | 嵌入式图像 |

### 任务类别分布

"""
    try:
        from datasets import load_dataset

        ds = load_dataset(str(root), split="train", trust_remote_code=True)

        total = len(ds)
        report = f"""## XLRS-Bench-lite (遥感多选题评测)

**路径**: `{root}`

### 数据集概览

| 指标 | 数值 |
|------|------|
| 格式 | HuggingFace Datasets (Arrow) |
| 总样本数 | {total} |
| 划分 | train only |

### 特征信息

根据数据集加载:

| 字段 | 类型 |
|------|------|
"""
        for feat_name, feat_type in ds.features.items():
            report += f"| {feat_name} | {feat_type} |\n"

        l1_counter = Counter()
        l2_counter = Counter()
        q_lens = []
        option_counts = []

        # Use column-based access to avoid PIL decode issues
        categories = ds["category"]
        l2_categories = ds["l2-category"]
        questions = ds["question"]
        options_list = ds["multi-choice options"]

        for i in range(total):
            l1_counter[str(categories[i] or "N/A")] += 1
            l2_counter[str(l2_categories[i] or "N/A")] += 1
            q_lens.append(len(str(questions[i] or "").split()))
            opts = options_list[i]
            option_counts.append(len(opts) if opts else 0)

        report += """
### 一级类别 (L1) 分布

| Category | 数量 | 占比 |
|----------|------|------|
"""
        for cat, cnt in l1_counter.most_common():
            report += f"| {cat} | {cnt} | {cnt/total*100:.1f}% |\n"

        report += """
### 二级类别 (L2) 分布

| Sub-category | 数量 | 占比 |
|-------------|------|------|
"""
        for cat, cnt in l2_counter.most_common():
            report += f"| {cat} | {cnt} | {cnt/total*100:.1f}% |\n"

        report += f"""
### 文本统计

| 指标 | 数值 |
|------|------|
| 平均问题长度 (words) | {sum(q_lens)/len(q_lens):.1f} |
| 最短问题 | {min(q_lens)} words |
| 最长问题 | {max(q_lens)} words |
| 平均选项数 | {sum(option_counts)/len(option_counts):.1f} |

"""

        # Show 2 samples using column access (avoid PIL decode bug)
        report += """
### 标注示例

**示例 1**:
```json
"""
        sample = {
            "index": int(ds["index"][0]),
            "path": ds["path"][0],
            "question": ds["question"][0],
            "multi-choice options": ds["multi-choice options"][0],
            "answer": ds["answer"][0],
            "category": ds["category"][0],
            "l2-category": ds["l2-category"][0],
        }
        report += json.dumps(sample, indent=2, ensure_ascii=False)

        report += """
```

**示例 2**:
```json
"""
        sample = {
            "index": int(ds["index"][1]),
            "path": ds["path"][1],
            "question": ds["question"][1],
            "multi-choice options": ds["multi-choice options"][1],
            "answer": ds["answer"][1],
            "category": ds["category"][1],
            "l2-category": ds["l2-category"][1],
        }
        report += json.dumps(sample, indent=2, ensure_ascii=False)
        report += """
```

### 图像分辨率

| 指标 | 数值 |
|------|------|
"""
        try:
            import pyarrow as pa
            from io import BytesIO

            arrow_dir = root / "train"
            arrow_files = sorted(arrow_dir.glob("data-*.arrow"))
            xlrs_widths, xlrs_heights = [], []
            for af in arrow_files[:3]:
                with open(af, "rb") as f:
                    reader = pa.ipc.open_stream(f)
                    for batch in reader:
                        col_idx = batch.schema.get_field_index("image")
                        for i in range(min(len(batch), 50)):
                            raw = batch.column(col_idx)[i].as_py()
                            if raw and raw[0]:
                                img = Image.open(BytesIO(raw[0]["bytes"]))
                                xlrs_widths.append(img.width)
                                xlrs_heights.append(img.height)

            xlrs_res = Counter(zip(xlrs_widths, xlrs_heights))
            report += f"| 采样数 | {len(xlrs_widths)} |\n"
            report += f"| 分辨率种类 | {len(xlrs_res)} |\n"
            report += "| 分辨率明细 | "
            for (w, h), c in xlrs_res.most_common(5):
                report += f"{w}×{h} ({c}张) "
            report += "|\n"
        except Exception as e:
            report += "| 分辨率明细 | 需通过 Arrow 原始字节读取 (采样错误) |\n"

        report += """
### 任务格式

- **输入**: 一张遥感图像 + 多项选择题问题
- **输出**: 选择题答案
- **覆盖13个子任务**: 计数、空间关系、地物分类、复杂推理等

"""
    except ImportError:
        report += """
> ⚠️ 需要 `datasets` 库以加载 Arrow 数据。
> 请安装: `pip install datasets`

文件结构:
```
train/
├── dataset_info.json
├── state.json
└── data-00000-of-00074.arrow  (共74个Arrow文件)
```

"""
    except Exception as e:
        report += f"""
> ⚠️ 加载数据集时出错: {e}

"""

    return report


def main():
    os.makedirs(REPORT_PATH.parent, exist_ok=True)

    sections = [
        "# 遥感数据集结构分析报告\n",
        "> 生成时间: 自动生成\n",
        "> 用途: 为多模态遥感大模型微调提供数据格式参考\n",
    ]

    sections.append(analyze_levir_cc())
    sections.append(analyze_mme_real_rs())
    sections.append(analyze_vrsbench())
    sections.append(analyze_xlrs_bench())

    # Cross-dataset summary
    summary = """
---

# 附录: 跨数据集汇总对比

| 维度 | LEVIR-CC | MME-RealWorld-RS | VRSBench | XLRS-Bench-lite |
|------|----------|-------------------|----------|-----------------|
| **任务类型** | 变化描述 | 多选VQA | 对话/VQA/Refer/Caption | 多选VQA |
| **图像数量** | ~10,077对 | ~295张 | ~19,802张 | 3,080 (内嵌) |
| **标注量** | 50,385句描述 | 3,738 QA | 142K对话+63K评测 | 3,080 QA |
| **输入格式** | 双时相图像对 | 图像+选择题 | 图像+自然语言 | 图像+多选题 |
| **输出格式** | 自然语言描述 | 选项字母(A-E) | 文本/坐标 | 选项 |
| **图像位置** | `images/{split}/{A,B}/` | `remote_sensing/` | `images/{split}/` | Arrow内嵌 |
| **图像分辨率** | 256×256 | 多种 (4096×4096, 7360×4912, ...) | 512×512 | 多种 (10000×10000为主) |
| **划分** | train/val/test | 无 | train/val | train |
| **挑战点** | 双时相对齐 | 细粒度感知 | 多任务统一 | 复杂推理 |

## 统一微调格式建议

所有数据集均可转换为 Qwen3-VL 的 `messages` 格式:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/image.png"},
        {"type": "text", "text": "指令内容"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "标准答案"}
      ]
    }
  ]
}
```

### 各数据集转换要点

| 数据集 | 图像处理 | 指令构造 | 答案构造 |
|--------|---------|---------|---------|
| LEVIR-CC | A图+B图 水平拼接 | "请描述这两张图像之间的变化" | 5句描述中选1句 |
| MME-RealWorld-RS | 单图 | 使用原问题文本 | 标准答案字母 |
| VRSBench | 单图 | 使用原对话中的 human 消息 | 使用原对话中的 gpt 消息 |
| XLRS-Bench-lite | 从Arrow提取图像 | 使用原问题文本 | 标准答案 |

"""
    sections.append(summary)

    full_report = "\n".join(sections)
    with open(REPORT_PATH, "w") as f:
        f.write(full_report)

    print(f"\n{'=' * 60}")
    print(f"报告已保存至: {REPORT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
