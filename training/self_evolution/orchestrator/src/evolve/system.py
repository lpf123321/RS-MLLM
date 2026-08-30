from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import workspace
from .io import atomic_write_text, write_json


def runtime_env(config: dict[str, Any]) -> dict[str, str]:
    root = workspace(config)
    cache = root / "cache"
    paths = {
        "PYTHONPYCACHEPREFIX": cache / "pycache",
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "TMPDIR": cache / "tmp",
        "MPLCONFIGDIR": cache / "matplotlib",
        "XDG_CACHE_HOME": cache / "xdg",
    }
    env = os.environ.copy()
    for key, value in paths.items():
        value.mkdir(parents=True, exist_ok=True)
        env[key] = str(value)
    env.update({"TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "offline", "PYTHONUNBUFFERED": "1"})
    return env


def run_logged(command: list[str], log_path: Path, env: dict[str, str], cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(log_path.with_suffix(log_path.suffix + ".command"), shlex.join(command) + "\n")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def write_sam_config(config: dict[str, Any], checkpoint: str, data_root: str, output_dir: Path) -> Path:
    import yaml
    sam = config["sam_training"]
    payload = {
        "model": {"checkpoint": checkpoint},
        "lora": {
            "rank": sam["rank"], "alpha": sam["alpha"], "dropout": sam["dropout"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "apply_to_vision_encoder": False, "apply_to_text_encoder": False,
            "apply_to_geometry_encoder": False, "apply_to_detr_encoder": True,
            "apply_to_detr_decoder": True, "apply_to_mask_decoder": False,
        },
        "training": {
            "data_root": data_root, "batch_size": sam["batch_size_per_gpu"],
            "num_workers": int(sam.get("num_workers", 2)),
            "learning_rate": sam["learning_rate"], "weight_decay": sam["weight_decay"],
            "max_epochs": sam["epochs_per_round"], "min_epochs": sam["epochs_per_round"],
            "early_stopping_patience": 1, "seed": config["seed"], "num_negatives": 0,
            "amp_dtype": "bfloat16",
        },
        "evaluation": {"batch_size": 2, "nms_iou": 0.7, "max_detections": 100, "bootstrap_repeats": 10000},
        "output": {"output_dir": str(output_dir)},
    }
    path = output_dir.parent / "sam_training.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
    return path


def train_sam(
    config: dict[str, Any], stage: str, round_id: int, current_checkpoint: str,
    data_root: str, negative_manifest: str, valid_negative_manifest: str, devices: list[int],
) -> dict[str, str]:
    root = workspace(config)
    bridge = Path(config["validated_sam_method"]["root"])
    output = root / "models" / stage / f"round_{round_id:02d}" / "sam_candidate"
    output.mkdir(parents=True, exist_ok=True)
    existing_adapter = output / "best_lora_weights.pt"
    existing_merged = output / "sam3_merged.pt"
    if existing_adapter.is_file() and existing_merged.is_file():
        return {"adapter": str(existing_adapter), "merged_checkpoint": str(existing_merged), "output_dir": str(output)}
    sam_config = write_sam_config(config, current_checkpoint, data_root, output)
    env = runtime_env(config)
    env["SAM3_REPO"] = config["repositories"]["sam3_lora"]
    env["SAM3_CHECKPOINT"] = current_checkpoint
    sam_cfg = config["sam_training"]
    command = [
        "conda", "run", "-n", "Sam3_lora", "python", str(bridge / "scripts/train_vopd_bbox_lora.py"),
        "--config", str(sam_config), "--output-dir", str(output), "--data-root", data_root,
        "--devices", *map(str, devices), "--max-epochs", str(sam_cfg["epochs_per_round"]),
        "--min-epochs", str(sam_cfg["epochs_per_round"]), "--negative-manifest", negative_manifest,
        "--valid-negative-manifest", valid_negative_manifest,
        "--negative-ratio", str(sam_cfg["negative_ratio"]), "--presence-weight", str(sam_cfg["presence_weight"]),
        "--rank-loss-weight", str(sam_cfg["ranking_weight"]), "--rank-margin", str(sam_cfg["ranking_margin"]),
    ]
    if float(sam_cfg["distill_weight"]) > 0:
        cache = output / "base_teacher_replay.pt"
        cache_command = [
            "conda", "run", "-n", "Sam3_lora", "python", str(bridge / "scripts/build_teacher_cache.py"),
            "--manifest", negative_manifest, "--output", str(cache), "--device", str(devices[0]),
            "--accepted-only", "--one-per-image", "--batch-size", "2",
        ]
        run_logged(cache_command, output / "teacher_cache.log", env, bridge)
        command.extend(["--distill-cache", str(cache), "--distill-weight", str(sam_cfg["distill_weight"])])
    run_logged(command, output / "train.log", env, bridge)
    adapter = output / "best_lora_weights.pt"
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    merged = output / "sam3_merged.pt"
    merge_command = [
        "conda", "run", "-n", "Sam3_lora", "python", str(bridge / "scripts/merge_vopd_lora_checkpoint.py"),
        "--base-checkpoint", current_checkpoint, "--adapter", str(adapter),
        "--config", str(sam_config), "--output", str(merged),
    ]
    run_logged(merge_command, output / "merge.log", env, bridge)
    return {"adapter": str(adapter), "merged_checkpoint": str(merged), "output_dir": str(output)}


def train_mllm(
    config: dict[str, Any], stage: str, round_id: int, current_model: str, parquet: str,
) -> dict[str, str]:
    root = workspace(config)
    repo = Path(config["repositories"]["vision_opd"])
    run_root = root / "runs" / stage / f"round_{round_id:02d}" / "opd"
    checkpoint_root = root / "models" / stage / f"round_{round_id:02d}" / "mllm_candidate_raw"
    merged_root = root / "models" / stage / f"round_{round_id:02d}" / "mllm_candidate"
    reuse_model = config.get("opd_training", {}).get("reuse_models", {}).get(stage)
    if reuse_model:
        reuse_path = Path(reuse_model).resolve()
        if not (reuse_path / "config.json").is_file():
            raise FileNotFoundError(f"Configured reusable MLLM is incomplete: {reuse_path}")
        run_root.mkdir(parents=True, exist_ok=True)
        provenance = {
            "model": str(reuse_path),
            "reason": "VOPD teacher inputs are invariant for this controlled variant",
            "stage": stage,
            "round": round_id,
            "source_parquet": str(Path(parquet).resolve()),
        }
        write_json(run_root / "reuse_model.json", provenance)
        return {
            "model": str(reuse_path), "raw_checkpoint": "reused_controlled_variant",
            "output_dir": str(run_root),
        }
    if (merged_root / "config.json").is_file():
        return {"model": str(merged_root), "raw_checkpoint": "already_merged", "output_dir": str(run_root)}
    run_root.mkdir(parents=True, exist_ok=True)
    opd = config["opd_training"]
    smoke = stage == "smoke"
    accepted = None
    training_records = None
    summary_path = Path(parquet).with_suffix(".summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        accepted = int(summary["accepted"])
        training_records = int(summary.get("training_records", accepted))
    batch = int(opd["smoke_train_batch_size"] if smoke else opd["train_batch_size"])
    gpu_count = int(opd["gpus"])
    if smoke and training_records is not None:
        batch = max(1, min(batch, training_records))
        gpu_count = max(1, min(gpu_count, training_records))
    rollout_n = int(opd["smoke_rollout_n"] if smoke else opd["rollout_n"])
    max_len = int(opd["max_prompt_length"]) + int(opd["max_response_length"])
    sequence_parallel = int(opd.get("sequence_parallel_size", 1))
    ppo_tokens_per_gpu = (max_len + sequence_parallel - 1) // sequence_parallel
    experiment = f"self-evolve-{stage}-r{round_id:02d}"
    overrides = [
        f'data.train_files=["{parquet}"]', 'data.val_files=[]', 'data.filter_overlong_prompts=False',
        f'data.max_prompt_length={opd["max_prompt_length"]}', f'data.max_response_length={opd["max_response_length"]}',
        'data.truncation=error', 'data.shuffle=True', 'data.trust_remote_code=True',
        'data.return_multi_modal_inputs=True', 'data.image_key=images', f'data.train_batch_size={batch}',
        'data.dataloader_num_workers=8', f'actor_rollout_ref.model.path={current_model}',
        'actor_rollout_ref.model.trust_remote_code=True', 'actor_rollout_ref.model.use_remove_padding=True',
        'actor_rollout_ref.model.enable_gradient_checkpointing=True', f'actor_rollout_ref.rollout.n={rollout_n}',
        f'actor_rollout_ref.actor.optim.lr={opd["learning_rate"]}', f'actor_rollout_ref.actor.ppo_mini_batch_size={batch}',
        'actor_rollout_ref.actor.use_dynamic_bsz=True', f'actor_rollout_ref.actor.ppo_max_token_len_per_gpu={ppo_tokens_per_gpu}',
        f'actor_rollout_ref.actor.ulysses_sequence_parallel_size={sequence_parallel}',
        'actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16',
        'actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16',
        'actor_rollout_ref.model.enable_activation_offload=True',
        'actor_rollout_ref.actor.fsdp_config.param_offload=True', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=True',
        'actor_rollout_ref.actor.policy_loss.loss_mode=vopd', 'actor_rollout_ref.actor.calculate_entropy=False',
        f'actor_rollout_ref.actor.self_distillation.distillation_topk={opd["distillation_topk"]}',
        'actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240',
        f'actor_rollout_ref.actor.self_distillation.is_clip={opd["is_clip"]}',
        'actor_rollout_ref.actor.self_distillation.teacher_always_on=True',
        'actor_rollout_ref.actor.self_distillation.teacher_model_source=fixed',
        f'actor_rollout_ref.actor.self_distillation.teacher_model_path={current_model}',
        'actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images',
        f'actor_rollout_ref.actor.self_distillation.alpha={opd["alpha"]}',
        'actor_rollout_ref.actor.self_distillation.include_environment_feedback=False',
        'actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True',
        'algorithm.rollout_correction.rollout_is=token', f'algorithm.rollout_correction.rollout_is_threshold={opd["is_clip"]}',
        'algorithm.adv_estimator=grpo', 'algorithm.norm_adv_by_std_in_grpo=False', 'algorithm.use_kl_in_reward=False',
        'actor_rollout_ref.rollout.name=vllm',
        f'actor_rollout_ref.rollout.tensor_model_parallel_size={opd.get("rollout_tensor_parallel_size", 1)}',
        f'actor_rollout_ref.rollout.gpu_memory_utilization={opd.get("rollout_gpu_memory_utilization", 0.7)}',
        'actor_rollout_ref.rollout.max_num_seqs=16',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1', f'actor_rollout_ref.rollout.max_model_len={max_len}',
        f'actor_rollout_ref.rollout.max_num_batched_tokens={max_len}',
        f'actor_rollout_ref.rollout.response_length={opd["max_response_length"]}',
        f'++actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len={max_len}',
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False',
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False',
        'actor_rollout_ref.rollout.calculate_log_probs=True', f'actor_rollout_ref.rollout.agent.num_workers={gpu_count}',
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1', 'actor_rollout_ref.ref.fsdp_config.param_offload=True',
        'reward_model.enable=False', f'critic.model.path={current_model}', 'reward_model.use_reward_loop=False',
        'custom_reward_function.path=null', 'trainer.project_name=CVSearch-Self-Evolve',
        f'trainer.experiment_name={experiment}', 'trainer.logger=["console"]',
        f'trainer.n_gpus_per_node={gpu_count}', 'trainer.nnodes=1',
        f'trainer.save_freq={int(opd.get("checkpoint_save_freq", 1))}',
        f'trainer.max_actor_ckpt_to_keep={int(opd.get("max_actor_ckpt_to_keep", 1))}',
        'trainer.test_freq=-1', 'trainer.total_epochs=1', 'trainer.val_before_train=False',
        f'trainer.default_local_dir={checkpoint_root}', f'trainer.rollout_data_dir={run_root / "rollouts"}',
        f'hydra.run.dir={run_root / "hydra"}',
    ]
    custom_template = opd.get("custom_chat_template_file")
    if custom_template:
        overrides.append(f'actor_rollout_ref.model.custom_chat_template_file={custom_template}')
    command = ["conda", "run", "-n", "vision-opd-rtx5090", "python", "-m", "verl.trainer.main_ppo", "--config-name", "vopd", *overrides]
    env = runtime_env(config)
    # Ray embeds its session name below TMPDIR; nested experiment paths can
    # exceed Linux's 107-byte AF_UNIX socket limit. Keep this short and still
    # fully contained in the independent orchestration repository.
    ray_tmp = Path(__file__).resolve().parents[2] / "r"
    ray_tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(ray_tmp)
    env.update({"PYTHONPATH": f"{repo}:{env.get('PYTHONPATH', '')}", "WORLD_SIZE": "1", "VLLM_USE_V1": "1"})
    run_logged(command, run_root / "train.log", env, repo)
    step_dirs = sorted(checkpoint_root.glob("global_step_*"), key=lambda path: int(path.name.rsplit("_", 1)[1]))
    if not step_dirs:
        raise FileNotFoundError(f"No Vision-OPD checkpoint under {checkpoint_root}")
    merge_command = [
        "conda", "run", "-n", "vision-opd-rtx5090", "python", "-m", "verl.model_merger", "merge",
        "--backend", "fsdp", "--local_dir", str(step_dirs[-1] / "actor"), "--target_dir", str(merged_root),
    ]
    run_logged(merge_command, run_root / "merge.log", env, repo)
    return {"model": str(merged_root), "raw_checkpoint": str(step_dirs[-1]), "output_dir": str(run_root)}


def evaluate_sam_selectivity(
    config: dict[str, Any], stage: str, round_id: int, checkpoint: str,
    data_root: str, negative_manifest: str, tag: str, device: int = 0,
) -> dict[str, Any]:
    root = workspace(config)
    bridge = Path(config["validated_sam_method"]["root"])
    output = root / "runs" / stage / f"round_{round_id:02d}" / "sam_eval" / tag
    metrics_path = output / "selectivity_metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    sam_config = write_sam_config(config, checkpoint, data_root, output)
    env = runtime_env(config)
    env["SAM3_REPO"] = config["repositories"]["sam3_lora"]
    env["SAM3_CHECKPOINT"] = checkpoint
    command = [
        "conda", "run", "-n", "Sam3_lora", "python", str(bridge / "scripts/evaluate_vopd_selectivity.py"),
        "--config", str(sam_config), "--split-dir", str(Path(data_root) / "valid"),
        "--negative-manifest", negative_manifest, "--output-dir", str(output),
        "--mode", "base", "--device", str(device), "--batch-size", "2",
        "--num-workers", str(int(config["sam_training"].get("num_workers", 2))),
    ]
    run_logged(command, output / "evaluate.log", env, bridge)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    return json.loads(metrics_path.read_text(encoding="utf-8"))
