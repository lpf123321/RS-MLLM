# OPD / OPSD 数据与实验流水线

本目录只包含源码。默认产物写入 `pipeline/artifacts/`，顶层 `.gitignore` 会排除
该目录；正式运行仍建议把 `ARTIFACT_ROOT` 设置到仓库外部。

主要入口：

1. `scripts/prepare_subsets.py`：构建固定 train/eval 子集和统一 SFT 数据；
2. `scripts/prepare_opd_formal.py`：构建 on-policy OPD parquet；
3. `scripts/run_formal_fullteacher_opd.sh`：冻结 Vision-OPD-9B teacher 的 OPD；
4. `scripts/run_offpolicy_teacher_trajectory_generation.sh`：生成并校验固定教师轨迹；
5. `scripts/run_formal_offpolicy_distillation.sh`：在教师轨迹上训练 student。

所有入口通过环境变量接收机器路径：

```bash
export PIPELINE_ROOT="$PWD/training/distillation/pipeline"
export ARTIFACT_ROOT=/path/to/opd-artifacts
export VISION_OPD_ROOT=/path/to/full/Vision-OPD
export VISION_OPD_PYTHON=/path/to/vision-opd/python
export STUDENT_MODEL=/path/to/Qwen3.5-4B
export TEACHER_MODEL=/path/to/Vision-OPD-9B
export DATA_ROOT=/path/to/datasets
```

`offpolicy/qwen35_teacher_adapter.py` 内置 PEFT adapter 加载，因此教师轨迹生成不再
依赖 RS-MLLM 的旧版 `evaluation/main.py`。数据、模型与输出均不会被提交。
