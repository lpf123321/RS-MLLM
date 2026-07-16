# 评测加速日志

## 问题

初始评测使用 `batch_size=4` + `torch.compile`，但：
1. `torch.compile(mode="default")` 在模型加载后卡死（疑似 dynamic shape tracing 与视觉输入不兼容），无法启用
2. `batch_size=4` 时 VRSBench 全量预估约 20 小时

## 加速手段

### 1. 增大 Batch Size（4 → 32）

Qwen3-VL-2B 在 bfloat16 下约 4GB，A100 显存 40GB，远未饱和。

| batch_size | VRSBench 100条耗时 | 每条耗时 | 全量预估 |
|-----------|-------------------|---------|---------|
| 4 | ~113s | 1.13s | ~20h |
| 32 | ~5s | 0.03s | ~40min |

**改动**：`evaluation/run_eval.slurm` 中 `EVAL_BATCH_SIZE=32`，直接传入 python `--eval_batch_size`。

### 2. 左填充（Left Padding）

Decoder-only 架构的 batch inference 需要左填充，否则生成结果错位。

```python
self.processor.tokenizer.padding_side = "left"
```

在 `Qwen3VLAdapter.__init__` 中设置，确保全局生效。

### 3. System Prompt 压缩输出

模型默认输出完整句子（VQA 输出 "The large vehicles seen in the image are yellow."），导致每 batch 推理时间变长 + 准确率为 0（格式不匹配）。

添加 dataset-specific system prompt 后：
- VRSBench VQA → 短词/短语（"yellow"），生成 token 数大幅减少。VQA 准确率从 0% → 41%
- MCQ（MME/XLRS）→ 格式 "X. (X) Text"，匹配评测标准。MME: 0% → 31%（exact）/ 45%（letter），XLRS: 0% → 32%（exact）/ 70%（letter）
- 对 CAP 和 REF 任务也加了解释性指引

Effect：VRSBench batch 耗时从 7.31s → 1.18s（纯 VQA 场景 ~6x 提速）。全量数据含 CAP/REF 长输出任务，平均 ~3-4s/batch。

### 3b. MCQ_Accuracy 指标

对 MCQ 任务增加 `MCQAccuracy` 指标（提取答案字母比较），比 exact match 更合理。MME 从 31% → 45%，XLRS 从 32% → 70%。

### 4. Python 无缓冲输出

```bash
export PYTHONUNBUFFERED=1
```

SLURM 下 stdout 默认全缓冲，无法实时看到进度。加此环境变量后 tqdm 进度条立即刷新。

### 5. 修正数据路径

原代码在 `output/` 下找数据文件，但实际数据在共享文件夹 `/users/u2024311136/shared/shared_datasets/`。将 `DATASETS` 路径改为绝对路径，避免找不到文件。

### 6. tqdm 进度条

```python
for i in tqdm(range(0, len(batch), batch_size), desc="Inference", unit="batch"):
```

在 `batch_generate` 中加入，实时显示当前 batch 进度和速度。

## 当前速度（全量）

batch_size=32, 实测全量推理速度（Job 1935 运行中数据）：

| 数据集 | 样本数 | Batches | 平均速度 | 预估全量 |
|-------|--------|---------|---------|---------|
| VRSBench | 62,918 | 1,967 | 2-6s/batch（VQA 快 / CAP 慢） | ~2h |
| MME | 3,736 | 117 | ~4s/batch | ~8min |
| XLRS | 3,080 | 96 | ~6.6s/batch | ~11min |
| LEVIR-CC | 1,929 | 60 | ~5.5s/batch | ~6min |
| **总计** | **71,663** | **2,240** | | **~2-3h** |

VRSBench 速度波动大（VQA ~2s/batch，CAP/REF ~4-6s/batch），因数据按任务类型顺序排列。

## 待尝试

- `torch.compile` 改用 `mode="reduce-overhead"` 或 `fullgraph=False` 排查 hang 问题（可再提速 1.5-2x）
- vLLM 推理加速
- 异步图像加载（当前 `process_vision_info` 同步加载磁盘图像）
