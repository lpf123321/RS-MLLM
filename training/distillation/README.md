# distillation/ —— 在线策略自蒸馏（报告 5.2）

对应报告 5.2「SFT 与在线策略自蒸馏」。

上传内容：
- `opd/` 或 `*.py`：在线策略蒸馏 OPD（在线大模型教师，region–global 逐 token 蒸馏）
- `opsd/` 或 `*.py`：在线自蒸馏改进 OPSD
- 数据构造脚本：基于在线大模型的 SFT 与 OPSD 数据构建
- 训练入口与配置（`*.slurm` / `*.sh`）

教师模型：Vision-OPD-9B（参见报告「模型参数量与权重存储实测」表）。