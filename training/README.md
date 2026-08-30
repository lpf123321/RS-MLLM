# Training —— 模型训练方法（报告第 5 章）

本目录存放训练方法相关的**蒸馏与自进化**模块。LoRA 微调框架见顶层 `finetune_framework/`，
训练提交脚本见顶层 `slurm_scripts/`。

目录结构：

| 子目录 | 对应报告章节 | 应放入内容 |
|--------|--------------|------------|
| `distillation/` | 5.2 SFT 与在线策略自蒸馏 | OPD（在线策略蒸馏）、OPSD（在线自蒸馏）的教师-学生蒸馏实现、训练/数据构建脚本 |
| `self_evolution/` | 5.3 CVSearch 驱动的 OPSD 自进化训练框架 | CVSearch 视觉搜索框架、Self-evolution 流程、问题级有效轨迹选择等实现 |

## distillation/（5.2）

- OPD：在线策略蒸馏（teacher-student 逐 token 分布对齐）
- OPSD：区域–全局（Region-Global）自蒸馏改进
- 基于在线大模型的 SFT 与 OPSD 数据构造
- 教师模型：Vision-OPD-9B