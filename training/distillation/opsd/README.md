# Region–Global OPSD

`vision_opd/` 保留 OPSD 的关键、原路径兼容源码切片：

- `verl/trainer/ppo/core_algos.py`：逐 token 自蒸馏损失；
- `verl/workers/actor/dp_actor.py`：student/teacher forward、mask、top-k 分布与反向传播；
- `verl/workers/fsdp_workers.py`：固定教师装载与 FSDP/LoRA 隔离；
- `verl/trainer/ppo/ray_trainer.py`、`verl/utils/dataset/rl_dataset.py`：全图与区域图输入流；
- `verl/trainer/config/{vopd,opsd}.yaml`：OPD/OPSD 配置；
- `scripts/prepare_data.py`、`run_vision_opd.sh`、`merge_checkpoint.sh`：数据、训练和权重合并入口。

该目录用于代码审阅和与完整 Vision-OPD checkout 对照。实际训练应安装完整的
Vision-OPD/VERL 依赖，然后用这里的同路径文件覆盖对应源码，或直接使用包含这些
改动的项目 checkout。`pipeline/scripts/run_formal_fullteacher_opd.sh` 会调用完整环境。

正式实验增加了两项必要修复：固定 9B teacher 不套用 4B student LoRA；teacher
log-prob 先计算并转存 CPU，再执行 student forward/backward，以降低峰值显存。

上游许可证保存在 `vision_opd/LICENSE`。
