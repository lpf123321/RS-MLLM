# Train-only SAM3 grounding pipeline

本目录把历史实验使用的确定性 SAM3 数据流程放回仓库，覆盖：train 图选择、固定类别
Codex 筛选、SAM3 推理、score/geometry/NMS/同类官方框 IoU 过滤、固定框语言审核和
native finalize。所有入口都只接收 train 输入，不接收或打开 test 路径。

SAM3 代码直接使用仓库内 `training/self_evolution/cvsearch/`，checkpoint 由
ModelScope 下载脚本提供；本目录不包含权重或数据。

## VRSBench-new

```bash
python sam3_pipeline/select_vrsbench_train.py \
  --train /data/VRSBench/train.json \
  --image-root /data/VRSBench/Images_train \
  --num-images 20 --output-dir runs/vrs/selection

python prepare_class_jobs.py \
  --selection runs/vrs/selection/selected_images.jsonl \
  --image-root /data/VRSBench/Images_train \
  --dataset-profile vrsbench --split train \
  --output runs/vrs/class_jobs.jsonl

python codex_runner.py validate --jobs runs/vrs/class_jobs.jsonl \
  --schema schemas/grounding_class_prediction.schema.json \
  --run-root runs/vrs/classify --stage smoke
python codex_runner.py run --jobs runs/vrs/class_jobs.jsonl \
  --schema schemas/grounding_class_prediction.schema.json \
  --run-root runs/vrs/classify --stage smoke
# 人工批准后再运行 full；批准步骤见上级 README。

python collect_class_results.py --jobs runs/vrs/class_jobs.jsonl \
  --results-dir runs/vrs/classify/full/results --dataset-profile vrsbench \
  --output runs/vrs/class_predictions.jsonl

python sam3_pipeline/run_sam3.py \
  --selection runs/vrs/selection/selected_images.jsonl \
  --class-predictions runs/vrs/class_predictions.jsonl \
  --dataset-profile vrsbench --image-root /data/VRSBench/Images_train \
  --sam3-repo ../../self_evolution/cvsearch --checkpoint /models/sam3.pt \
  --output-dir runs/vrs/sam3_raw

python sam3_pipeline/postprocess_vrsbench.py \
  --detections runs/vrs/sam3_raw/detections.jsonl --dataset-profile vrsbench \
  --train /data/VRSBench/train.json \
  --annotations-root /data/VRSBench/Annotations_train \
  --image-root /data/VRSBench/Images_train --output-dir runs/vrs/candidates

python prepare_grounding_jobs.py \
  --candidates runs/vrs/candidates/candidates.jsonl \
  --image-root /data/VRSBench/Images_train \
  --official-train-records runs/vrs/selection/official_records.jsonl \
  --dataset-profile vrsbench --split train --boxes-are-frozen-postprocessed \
  --output runs/vrs/language_jobs.jsonl
# 对 language_jobs 执行同样的 validate -> smoke -> 人工批准 -> full。
python collect_results.py --jobs runs/vrs/language_jobs.jsonl \
  --results-dir runs/vrs/language/full/results --kind grounding \
  --output runs/vrs/question_decisions.jsonl

python sam3_pipeline/finalize_vrsbench.py \
  --train /data/VRSBench/train.json --dataset-profile vrsbench \
  --candidates runs/vrs/candidates/candidates.jsonl \
  --question-decisions runs/vrs/question_decisions.jsonl \
  --sam3-manifest runs/vrs/sam3_raw/manifest.json \
  --selection runs/vrs/selection/selected_images.jsonl \
  --image-root /data/VRSBench/Images_train --output-dir runs/vrs/final
```

## XLRS-grounding-new

```bash
python sam3_pipeline/select_xlrs_train.py \
  --train-dir /data/XLRS-Bench_visual_grounding_en/train \
  --num-images 20 --output-dir runs/xlrs/selection

# prepare_class_jobs / Codex 分类 / collect_class_results / run_sam3 与上面相同，
# dataset-profile 改为 xlrs，image-root 使用 runs/xlrs/selection/images。

python sam3_pipeline/postprocess_xlrs.py \
  --detections runs/xlrs/sam3_raw/detections.jsonl \
  --official-records runs/xlrs/selection/official_records.jsonl \
  --image-root runs/xlrs/selection/images --output-dir runs/xlrs/candidates

# prepare_grounding_jobs / Codex 语言审核 / collect_results 同上，profile 改为 xlrs。
python sam3_pipeline/finalize_xlrs.py \
  --selection runs/xlrs/selection/selected_images.jsonl \
  --official-records runs/xlrs/selection/official_records.jsonl \
  --candidates runs/xlrs/candidates/candidates.jsonl \
  --question-decisions runs/xlrs/question_decisions.jsonl \
  --image-root runs/xlrs/selection/images --output-dir runs/xlrs/final

python sam3_pipeline/normalize_xlrs_fixed5.py \
  --synthetic-jsonl runs/xlrs/final/synthetic_grounding.jsonl \
  --output runs/xlrs/final/xlrs_grounding_new_fixed5.json
```

Fixture 测试只验证代码连通性和确定性门禁。训练数据必须来自真实 SAM3 manifest
（`fixture_only=false`），并完成两轮 20 图人工 smoke；fixture 产物不能用于训练。
