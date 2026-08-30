# 待上传代码清单（供协作成员对照填写）

仓库 `~/RS-MLLM-release`（GitHub: `lpf123321/RS-MLLM`, 分支 `main`）已按技术报告章节搭好目录骨架，
以下模块为**待上传**。各目录当前仅有 README 占位说明，请把对应代码放入指定目录后提交。

---

## 1. 训练侧脚本（四专家 SFT + delta 构建）

**归属**：`slurm_scripts/`（训练提交脚本）、`scripts/`（数据构建与 delta 工具）

本地旧仓库 `~/RS-MLLM/` 已有这些脚本，但 release 分支未包含，需整理后上传：

### SFT 训练提交脚本 → `slurm_scripts/`
- `sft_stage1.slurm` / `sft_stage1_clean.slurm`（统一 SFT 主干）
- `sft_expert_general.slurm`, `sft_ga2_general.slurm`, `sft_ga3_general.slurm`（General 专家）
- `sft_expert_grounding.slurm`, `sft_a1_grounding.slurm`（Grounding 专家 / 域对齐）
- `sft_expert_change.slurm`, `sft_a2_change.slurm`, `sft_a2b_change.slurm`（Change 专家续训）
- `sft_caption_expert.slurm`（Caption 专家）
- `sft_stage2.slurm`

### 数据构建与 delta 工具 → `scripts/`
- `build_caption_expert_data.py`（Caption 双域数据构建）
- `generate_mcq_data.py`（合成 MCQ 样本）
- `split_expert_data.py`（专家数据切分）
- `convert_manifest_to_sft.py`（manifest → SFT 格式）
- `split_eval_data.py`（评测数据集切分）
- `gen_expert_deltas.py`（四专家 full-rank delta 生成，产出 `lora_expert/lora/{expert}/delta_model.pt`）
- `load_base_with_delta.py`（W0 + delta 装配加载）
- `merge_lora_for_eval.py`（LoRA 合并用于评测）

---

## 2. 在线策略自蒸馏 OPD / OPSD（报告 5.2）

**归属**：`training/distillation/`

应上传内容：
- OPD（在线策略蒸馏）教师-学生蒸馏实现
- OPSD（在线自蒸馏，region–global 逐 token 蒸馏）改进
- 基于在线大模型的 SFT 与 OPSD 数据构造脚本
- 训练入口与配置（`*.slurm` / `*.sh`）
- 教师模型：Vision-OPD-9B

---

## 3. CVSearch 驱动的 OPSD 自进化训练（报告 5.3）

**归属**：`training/self_evolution/`

应上传内容：
- CVSearch 视觉搜索框架（在完整图像上提取视觉目标并调用 search/定位）
- Self-evolution 流程（问题级有效轨迹筛选、学生模型重训练）
- 实验配置与效果脚本（`*.slurm` / `*.sh`）

---

## 4. 模型量化（报告 6.2）

**归属**：`quantization/`

应上传内容：
- INT8（W8A8）离线量化流程（LLM Compressor）
- GPTQ W4A16 离线量化流程（LLM Compressor）
- 运行时量化：bitsandbytes int8 / nf4 推理集成
- 量化对比与实测结论脚本（量化对比子集 770 条、运行时量化配对子集 590 条）
- 转换工具：LLM Compressor、bitsandbytes

---

## 5. 星载容错与故障恢复

**归属**：`tolerance/`

应上传内容：
- 故障模型与覆盖边界定义
- 不可变运行封套与完整性检查（哈希绑定、启动校验）
- 运行时安全状态机（启动 / 运行 / 降级 / 回滚状态流转）
- 容错收益实验验证脚本（注入故障、对比基线）

---

## 提交约定
- 放对目录；`*.slurm` 中的个人硬编码路径（`/home/u2024311149/...`）请改为共享路径或环境变量默认值
- LLM Compressor / bitsandbytes 依赖请注明版本（报告附录「模型与版本」）
- README 占位可被实际代码补齐，不要删除目录结构