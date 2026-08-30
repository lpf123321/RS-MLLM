# Compression —— 视觉 Token 压缩与模型轻量化（报告第 6 章）

本目录为**导航层**：剪枝与 token 压缩实现保留在顶层包 `prune/` 与 `token_compression/`
（维持 Python import 不被破坏），此处仅汇总本章的量化与轻量化模块。

| 子目录 | 对应报告章节 | 应放入内容 |
|--------|--------------|------------|
| `quantization/` | 6.2 模型量化 | INT8（W8A8）量化、GPTQ W4A16 离线量化、运行时 bitsandbytes int8/nf4 流程 |

## prune/ / token_compression/（6.1，位于顶层）

- 免训练剪枝：`prune/`（L2、SCOPE）
- token 压缩方法集：`token_compression/`（uniform/random/mmtok/l2norm/scope_l2/divprune/fourier）
- 剪枝推理/可视化：`scripts/prune*.py`、`scripts/viz*.py`
- 剪枝 Router 设计：`evaluation/adapters/router_pruned.py`、`evaluation/router/task_prune_config.py`

## quantization/（6.2，待上传）