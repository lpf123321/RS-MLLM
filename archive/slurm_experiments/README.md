# archive/slurm_experiments

原始实验的 SLURM 提交脚本归档（**证据保留，不参与运行**）。

## 用途

这些脚本是《技术报告》各实验（训练/评测/量化配对/剪枝/冒烟）在 HITSZ 超算集群上的
原始提交记录，包含当时的作业参数、数据路径与评测配置，供报告审阅与追溯使用。

**独立仓库的运行链路不依赖它们**：评测/推理统一走 `./rsmllm.sh`（vLLM 评测器 +
WebUI），训练见 `scripts/train.sh` 的本地直跑说明。归档脚本中的 slurm 参数
（`--partition`/`--gres` 等）与个人硬编码路径（`/home/u2024311149/...`、
`/users/u2024311136/...`）在集群外无效。

## 内容

```
slurm_scripts/                 # 训练/delta 转换/评测/冒烟提交脚本(28 个)
run_eval.slurm                 # 评测入口提交
run_eval_3.5_2b.slurm          # Qwen3.5-2B 评测
run_eval_geoeyes.slurm         # GeoEyes 评测
run_eval_qwen35vl.slurm        # Qwen3.5-VL 评测(硬编码 conda 路径)
run_memory_smoke.slurm         # 显存冒烟
run_xlrs_memory.slurm          # XLRS 显存评测
smoke_xlrs_memory.slurm        # XLRS 显存预检
run_divprune_1000.slurm        # DivPrune 1000 样本评测
rerun_caption.slurm            # caption CIDEr 重算
quant_pair_eval.sbatch         # 量化配对评测(A-B-B-A)
```

如需在集群上复跑任一实验，先按报告附录「模型与版本」恢复环境，
再按脚本内注释修正数据/模型路径。
