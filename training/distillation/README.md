# Distillation: OPD / OPSD（报告 5.2）

本目录提交可复现的在线策略蒸馏与区域–全局在线自蒸馏代码，不包含模型权重、
训练数据、checkpoint、rollout 或日志。

## 目录

| 目录 | 内容 |
|---|---|
| `expert_lora/` | 实际保留的 General Exp3、Grounding bootstrap/Exp1/Exp4/Exp5 LoRA 训练、base+delta 合并和 ModelScope 下载入口 |
| `flashopd_mm/` | 轻量多模态 OPD：student rollout、local/API teacher、逐 token KL/JSD、LoRA 与分布式入口 |
| `opsd/vision_opd/` | 固定上游版本的完整 VERL/Vision-OPD 框架、Region–Global OPSD 改动、配置与训练/合并入口 |
| `pipeline/` | 正式数据构建、固定 Vision-OPD-9B 教师 on-policy/off-policy 实验及审计脚本 |

两条实现互补：`flashopd_mm` 便于阅读和独立验证 teacher–student OPD；
`opsd/vision_opd` 是正式 Region–Global OPSD 实验所使用的完整可安装源码快照。

## 快速检查

```bash
python -m compileall -q training/distillation

cd training/distillation/flashopd_mm
pip install -e .
STUDENT_MODEL=/path/to/student \
TEACHER_MODEL=/path/to/teacher \
VRSBENCH_TRAIN_JSONL=/path/to/vrsbench_train.jsonl \
VRSBENCH_IMAGE_ROOT=/path/to/Images_train \
bash scripts/run_vrsbench.sh configs/vrsbench_smoke_200.yaml
```

正式 OPSD 需要完整 Vision-OPD/VERL 环境。入口、所需环境变量和数据 schema 见
[`opsd/README.md`](opsd/README.md) 与 [`pipeline/README.md`](pipeline/README.md)。
General/Grounding Expert 的已验证 SFT 参数和运行入口见
[`expert_lora/README.md`](expert_lora/README.md)。

## 来源快照

- FlashOPD 基线：`china10s/flash-opd@f2485a646dbddac997396cd36e36ee2e41d3e52e`，包含本项目的多模态扩展。
- Vision-OPD 基线：`VisionOPD/Vision-OPD@c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471`，包含本项目固定教师和显存修复。
- 本项目流水线来自本机 `OPD-pipeline` 的已完成正式实验源码；产物未提交。
