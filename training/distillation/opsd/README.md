# Region–Global OPSD

`vision_opd/` 是固定到上游提交
`c8a8fdd1f88eef1b5ef4fe6a8d64eb0272917471` 的完整可安装源码快照，并叠加本项目
OPSD 改动。关键文件包括：

- `verl/trainer/ppo/core_algos.py`：逐 token 自蒸馏损失；
- `verl/workers/actor/dp_actor.py`：student/teacher forward、mask、top-k 分布与反向传播；
- `verl/workers/fsdp_workers.py`：固定教师装载与 FSDP/LoRA 隔离；
- `verl/trainer/ppo/ray_trainer.py`、`verl/utils/dataset/rl_dataset.py`：全图与区域图输入流；
- `verl/trainer/config/{vopd,opsd}.yaml`：OPD/OPSD 配置；
- `scripts/prepare_data.py`、`run_vision_opd.sh`、`merge_checkpoint.sh`：数据、训练和权重合并入口。

不再需要额外 checkout 才能获得缺失的 VERL 模块。安装依赖后可直接在
`vision_opd/` 中执行训练；`pipeline/scripts/run_formal_fullteacher_opd.sh` 也可以把
`VISION_OPD_ROOT` 指向此目录。

正式实验增加了两项必要修复：固定 9B teacher 不套用 4B student LoRA；teacher
log-prob 先计算并转存 CPU，再执行 student forward/backward，以降低峰值显存。

上游许可证保存在 `vision_opd/LICENSE`。

轻量检查：

```bash
cd training/distillation/opsd/vision_opd
PYTHONPATH=. python -m unittest discover -s tests -v
python -m verl.trainer.main_ppo --help
```
