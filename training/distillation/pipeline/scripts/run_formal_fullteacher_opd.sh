#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIPELINE_ROOT=${PIPELINE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
PROJECT_ROOT=${VISION_OPD_ROOT:?Set VISION_OPD_ROOT to a complete Vision-OPD checkout}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PIPELINE_ROOT}/artifacts}
PYTHON=${VISION_OPD_PYTHON:-python}
STUDENT_PATH=${STUDENT_MODEL:?Set STUDENT_MODEL to the student checkpoint}
TEACHER_PATH=${TEACHER_MODEL:-${ARTIFACT_ROOT}/sft/full-vision-opd-9b-sft-merged}
TEACHER_SFT_ADAPTER=${TEACHER_SFT_ADAPTER:-${ARTIFACT_ROOT}/sft/full-vision-opd-9b-lora}
TRAIN_FILE=${OPD_TRAIN_FILE:-${ARTIFACT_ROOT}/opd/data/formal_train_18413.parquet}
OUTPUT_ROOT=${ARTIFACT_ROOT}/opd
EXPERIMENT=formal-fullsft9b-to-base4b-opd-lora-r8
CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints/$EXPERIMENT"
ROLLOUT_DIR="$OUTPUT_ROOT/rollouts/$EXPERIMENT"
LOGPROB_DIR="$OUTPUT_ROOT/logprob_dumps/$EXPERIMENT"
TEMPLATE="$PROJECT_ROOT/chat_templates/perception_chat_template_qwen35.jinja"
TRAIN_BATCH_SIZE=8
SAVE_FREQ=50

for path in "$STUDENT_PATH" "$TEACHER_PATH" \
  "$TEACHER_SFT_ADAPTER" "$TRAIN_FILE" "$TEMPLATE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
done

mkdir -p "$CHECKPOINT_DIR" "$ROLLOUT_DIR" "$LOGPROB_DIR"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export USER="${USER:-$(id -un 2>/dev/null || echo cpy)}"
unset VLLM_ATTENTION_BACKEND
ulimit -c 0

echo "OPD student initialization (original, not SFT): $STUDENT_PATH"
echo "OPD trainable parameters: fresh rank-8 LoRA on gate/up/down projections"
echo "Frozen full-data-SFT teacher: $TEACHER_PATH"
echo "Teacher full-data SFT adapter provenance: $TEACHER_SFT_ADAPTER"
echo "Training data: $TRAIN_FILE (18,413 records; 2,301 batches)"
echo "Global batch size: $TRAIN_BATCH_SIZE"
echo "Checkpoint directory: $CHECKPOINT_DIR"

"$PYTHON" -m verl.trainer.main_ppo --config-name vopd \
  "data.train_files=[\"$TRAIN_FILE\"]" \
  'data.val_files=[]' \
  data.filter_overlong_prompts=False \
  data.max_prompt_length=2048 \
  data.max_response_length=64 \
  data.truncation=error \
  data.shuffle=True \
  data.seed=20260809 \
  data.trust_remote_code=True \
  data.return_multi_modal_inputs=True \
  data.image_key=images \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.dataloader_num_workers=2 \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$STUDENT_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.custom_chat_template_file="$TEMPLATE" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  "actor_rollout_ref.model.target_modules='.*(gate_proj|up_proj|down_proj)$'" \
  actor_rollout_ref.model.lora_adapter_path=null \
  actor_rollout_ref.actor.freeze_vision_tower=True \
  actor_rollout_ref.actor.data_loader_seed=20260809 \
  actor_rollout_ref.actor.fsdp_config.seed=20260809 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2112 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
  actor_rollout_ref.actor.calculate_entropy=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
  actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
  actor_rollout_ref.actor.self_distillation.distillation_add_tail=True \
  actor_rollout_ref.actor.self_distillation.alpha=0.5 \
  actor_rollout_ref.actor.self_distillation.gamma=1.0 \
  actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
  actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed \
  actor_rollout_ref.actor.self_distillation.teacher_model_path="$TEACHER_PATH" \
  actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0 \
  actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
  actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher=False \
  actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
  actor_rollout_ref.actor.self_distillation.log_prob_dump_dir="$LOGPROB_DIR" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2112 \
  actor_rollout_ref.rollout.max_model_len=2112 \
  actor_rollout_ref.rollout.response_length=64 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs=1 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  reward_model.enable=False \
  reward_model.use_reward_loop=False \
  custom_reward_function.path=null \
  trainer.project_name=OPD-pipeline \
  trainer.group_name=formal-fullteacher \
  trainer.experiment_name="$EXPERIMENT" \
  'trainer.logger=[console,tensorboard]' \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DIR"

latest_file="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
if [[ ! -f "$latest_file" ]]; then
  echo "Training ended without a latest-checkpoint marker: $latest_file" >&2
  exit 1
fi
latest_step=$(tr -d '[:space:]' < "$latest_file")
adapter_dir="$CHECKPOINT_DIR/global_step_${latest_step}/actor/lora_adapter"
"$PYTHON" "${PIPELINE_ROOT}/scripts/repair_verl_lora_config.py" "$adapter_dir"
echo "Full-teacher formal OPD finished at global step $latest_step"
echo "Loadable LoRA adapter: $adapter_dir"
