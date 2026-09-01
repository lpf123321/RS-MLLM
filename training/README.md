# Training —— 模型训练方法（报告第 5 章）

本目录存放训练方法相关的**蒸馏与自进化**模块。LoRA 微调框架见顶层 `finetune_framework/`，
训练提交脚本见顶层 `slurm_scripts/`。

模型与数据不进入 Git 仓库。各模块 README 和 `scripts/download_assets.sh` 记录
ModelScope 资产 ID、下载布局、运行入口及已验证边界。

## Expert 续训快捷入口

在仓库根目录运行：

```bash
# General Expert：MME-new + XLRS-new（保留的 General Exp3 配置，4 GPU）
bash training/train_general_expert.sh --gpus 0,1,2,3

# Grounding Expert：washed VRSBench + VRSBench-new + XLRS official/new
# （保留的 Grounding Exp5 联合配置，8 GPU）
bash training/train_grounding_expert.sh --gpus 0,1,2,3,4,5,6,7
```

两个入口都会自动下载缺失的 ModelScope 资产，分别合并
`Qwen3.5-4B + general_expert_delta` 和
`Qwen3.5-4B + grounding_expert_delta`，然后启动 LoRA 续训。Grounding Exp5
还会加载 942 条 VRSBench-new 训练得到的 bootstrap LoRA。使用本地资产时设置
`RS_MLLM_ASSET_ROOT`；已准备完毕且不希望联网时再设置
`RS_MLLM_SKIP_DOWNLOAD=1`。可先追加 `--prepare-only` 或 `--dry-run` 检查环境。
需要验证真实 forward/backward 而不启动完整一轮训练时，可使用
`--max-updates 1`；正式复现实验不要设置该参数。

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
