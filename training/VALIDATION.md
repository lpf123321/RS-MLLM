# 本地复现验证记录（2026-08-31）

本文件记录发布前的轻量验证，不把 smoke 结果冒充为报告中的完整训练复跑。

## 自动化测试

共 50 项单元/契约测试通过：Expert 8、Codex-login 数据生成 16、FlashOPD 5、
Self-evolution 20、Vision-OPD/OPSD 1。此外，`compileall`、所有 Shell 脚本
`bash -n` 与 `git diff --check` 均通过。

## 真实模型 smoke

- General full-rank delta：Qwen3.5-4B + delta 成功合并，校验 723 个 tensor，delta
  SHA-256 为 `97d38a1c786015342dd309219fff6158fc870412eef8aca6ca126a29704b65ab`；
  单图 LoRA 完成 1 个 optimizer update，loss `0.73974985`。
- Grounding full-rank delta：Qwen3.5-4B + delta 成功合并，校验 723 个 tensor，delta
  SHA-256 为 `0e76042ba819b6164d6041b8fe13758041a145600a3226ff311e3d3cf45a7c94`；
  加载 942 bootstrap adapter 后完成 1 个 optimizer update，loss `0.04778161`。
- FlashOPD：Qwen3-VL-2B 单样本完成 student rollout、local teacher 打分、reverse-KL、
  CE+KL 反传和 LoRA 保存；KL `0.006089`，总 loss `3.894`，GPU 峰值约 `4.17 GiB`。
- OPSD：完整 Vision-OPD Hydra 入口可组合；region–global top-k JSD 在 CUDA 上完成
  前向/反向，梯度有限。
- CVSearch：Qwen3-VL-2B + stock SAM3 分别对 tree-node weak box、strict local
  recovery、fine trace-only 生成 1 条真实 GPU 轨迹。

## 数据与历史产物重放

- 五套 Expert 配置逐图校验通过：6,816 / 942 / 36,285 / 6,611 / 43,838 条。
- 三套历史 self-evolution train 轨迹均为 999 条，其中各 583 条通过问题级 OPSD
  过滤；当前构建器分别重建 583 条 parquet，所有 full image 与 teacher crop 可解析。
- XLRS-new 的 Codex-login 真实 smoke 为 20/20 schema/identity/collector 通过；尚需用户
  人工批准后才能作为正式生成运行。

## 尚未由本轮 smoke 证明的内容

- 报告中的完整 epoch、多卡最终指标没有在本轮重新训练；仓库保留的是历史配置、
  checkpoint、审计产物与轻量启动证据。
- MME 3,736 的历史中间 accepted/style/audit 文件已经缺失，只能发布最终训练 JSON；
  不能声称逐条重建。
- VRSBench-new 942 与 XLRS-grounding-new 303 的确定性选择、格式和 fixture 已验证，
  若要声称与历史接受集合完全等价，仍需重新执行独立视觉复核和完整评测。

## ModelScope 回读

五个私有资产仓库均已上传并回读清单校验。Expert Checkpoints 最后完成，20 个文件、
7 个 LFS 上传、失败 0；远端与本地 `ASSET_MANIFEST.json` 和 `SHA256SUMS` 均逐字节一致。
