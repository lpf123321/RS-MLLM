#!/usr/bin/env python3
"""Stage retained checkpoints for ModelScope without base weights or local paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ADAPTER_FILES = (
    "adapter_model.safetensors", "adapter_config.json", "training_metrics.json",
    "trainer_state.json", "non_lora_state_dict.bin",
)
MLLM_FILES = (
    "model.safetensors", "config.json", "generation_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def scrub(value: Any) -> Any:
    if isinstance(value, str) and Path(value).is_absolute():
        return f"redacted-local-path://{Path(value).name}"
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_adapter(source: Path, destination: Path, base_reference: str) -> dict[str, Any]:
    copied = []
    for name in ADAPTER_FILES:
        path = source / name
        if not path.is_file():
            continue
        target = destination / name
        if name == "adapter_config.json":
            config = json.loads(path.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = base_reference
            write_json(target, scrub(config))
        elif name.endswith(".json"):
            write_json(target, scrub(json.loads(path.read_text(encoding="utf-8"))))
        else:
            copy_file(path, target)
        copied.append(name)
    if "adapter_model.safetensors" not in copied or "adapter_config.json" not in copied:
        raise FileNotFoundError(f"incomplete adapter: {source}")
    return {
        "files": copied,
        "adapter_sha256": sha256(destination / "adapter_model.safetensors"),
        "base_reference": base_reference,
    }


def checksums(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    with (root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")


def stage_experts(experiment_root: Path, output: Path) -> None:
    empty_output(output)
    delta_root = experiment_root / "expert/lora_expert/lora"
    checkpoint_root = experiment_root / "sft/checkpoints"
    deltas = {}
    for expert in ("general", "grounding"):
        source = delta_root / expert / "delta_model.pt"
        target = output / "deltas" / expert / "delta_model.pt"
        copy_file(source, target)
        deltas[expert] = {"sha256": sha256(target), "bytes": target.stat().st_size}

    adapters = {}
    specs = {
        "general_exp3": checkpoint_root / "mme_xlrs_matrix/exp3_old_mme_plus_old_xlrs_lora",
        "grounding_bootstrap_942": checkpoint_root / "vrsbench_referring/small_grounding_test_only_grounding_lora",
        "grounding_exp1": checkpoint_root / "grounding_942_continuations/exp1_942_then_washed_vrs36285_lora",
        "grounding_exp4": checkpoint_root / "grounding_942_continuations/exp4_942_then_xlrs6308_plus_sam3clean303_fixed5_lora",
        "grounding_exp5": checkpoint_root / "grounding_942_continuations/exp5_942_then_washed_vrs36285_plus_942_plus_xlrs6611_lora",
    }
    for name, source in specs.items():
        expert = "general" if name.startswith("general") else "grounding"
        adapters[name] = copy_adapter(
            source, output / "adapters" / name,
            f"Qwen/Qwen3.5-4B + deltas/{expert}/delta_model.pt",
        )
    (output / "README.md").write_text(
        "# RS-MLLM Expert Checkpoints\n\n"
        "Full-rank General/Grounding expert deltas and the five retained LoRA outputs. "
        "The Qwen3.5-4B base model is not duplicated; compose it with the delta before "
        "loading an adapter. See `training/distillation/expert_lora` in RS-MLLM.\n",
        encoding="utf-8",
    )
    write_json(output / "ASSET_MANIFEST.json", {
        "schema_version": 1, "base_model": "Qwen/Qwen3.5-4B",
        "contains_base_weights": False, "deltas": deltas, "adapters": adapters,
        "contains_absolute_paths": False,
    })
    checksums(output)


def stage_distillation(opd_root: Path, output: Path) -> None:
    empty_output(output)
    specs = {
        "sft_student_qwen35_4b": (
            opd_root / "sft/formal-qwen3.5-4b-lora", "Qwen/Qwen3.5-4B"
        ),
        "sft_teacher_vision_opd_9b": (
            opd_root / "sft/formal-vision-opd-9b-lora", "VisionOPD/Vision-OPD-9B"
        ),
        "opd_online_teacher_final": (
            opd_root / "opd/checkpoints/formal-fullsft9b-to-base4b-opd-lora-r8/global_step_2301/actor/lora_adapter",
            "Qwen/Qwen3.5-4B",
        ),
        "opd_offpolicy_teacher_final": (
            opd_root / "offpolicy/checkpoints/formal-fullsft9b-to-base4b-offpolicy-lora-r8/global_step_2301/actor/lora_adapter",
            "Qwen/Qwen3.5-4B",
        ),
    }
    adapters = {
        name: copy_adapter(source, output / "adapters" / name, base)
        for name, (source, base) in specs.items()
    }
    (output / "README.md").write_text(
        "# RS-MLLM Distillation Checkpoints\n\n"
        "Retained SFT, online-policy OPD, and frozen-trajectory comparison adapters. "
        "Base model weights and optimizer/FSDP shards are intentionally excluded.\n",
        encoding="utf-8",
    )
    write_json(output / "ASSET_MANIFEST.json", {
        "schema_version": 1, "contains_base_weights": False,
        "contains_optimizer_state": False, "adapters": adapters,
        "contains_absolute_paths": False,
    })
    checksums(output)


def copy_mllm(source: Path, destination: Path) -> dict[str, Any]:
    copied = []
    for name in MLLM_FILES:
        path = source / name
        if not path.is_file():
            continue
        target = destination / name
        if name.endswith(".json"):
            write_json(target, scrub(json.loads(path.read_text(encoding="utf-8"))))
        else:
            copy_file(path, target)
        copied.append(name)
    if "model.safetensors" not in copied or "config.json" not in copied:
        raise FileNotFoundError(f"incomplete merged MLLM checkpoint: {source}")
    return {
        "format": "merged_huggingface_checkpoint",
        "files": copied,
        "model_sha256": sha256(destination / "model.safetensors"),
        "bytes": (destination / "model.safetensors").stat().st_size,
    }


def copy_sam(source: Path, destination: Path) -> dict[str, Any]:
    copy_file(source / "best_lora_weights.pt", destination / "best_lora_weights.pt")
    copied = ["best_lora_weights.pt"]
    for name in ("resolved_config.json", "training_summary.json", "training_history.json"):
        path = source / name
        if path.is_file():
            write_json(destination / name, scrub(json.loads(path.read_text(encoding="utf-8"))))
            copied.append(name)
    return {
        "format": "sam3_lora_adapter", "files": copied,
        "adapter_sha256": sha256(destination / "best_lora_weights.pt"),
    }


def stage_self_evolution(runs_root: Path, output: Path) -> None:
    empty_output(output)
    methods = {
        "tree_node_weak_box": runs_root / "qwen3vl_2b_bbox_free_one_fifth",
        "strict_local_recovery": runs_root / "qwen3vl_2b_sam_recovered_multi_one_fifth",
        "fine_trace_only": runs_root / "qwen3vl_2b_fine_sam_trace_one_fifth",
    }
    manifest: dict[str, Any] = {}
    for method, root in methods.items():
        sam_source = root / "models/full/round_00/sam_candidate"
        sam = copy_sam(sam_source, output / "methods" / method / "sam3_adapter")
        if method == "fine_trace_only":
            mllm = {
                "format": "reuse_reference",
                "path": "../strict_local_recovery/mllm_candidate",
                "reason": "controlled variant reuses the strict-local-recovery OPSD candidate",
            }
            write_json(output / "methods" / method / "mllm_reuse.json", mllm)
        else:
            mllm = copy_mllm(
                root / "models/full/round_00/mllm_candidate",
                output / "methods" / method / "mllm_candidate",
            )
        manifest[method] = {"mllm": mllm, "sam3": sam}
    (output / "README.md").write_text(
        "# RS-MLLM Self-Evolution Checkpoints\n\n"
        "Two merged Qwen3-VL-2B OPSD candidates and three SAM3 LoRA candidates from the "
        "retained single-round experiments. Fine trace-only deliberately reuses the strict "
        "MLLM candidate. Official Qwen3-VL-2B and SAM3 bases are not duplicated.\n",
        encoding="utf-8",
    )
    write_json(output / "ASSET_MANIFEST.json", {
        "schema_version": 1,
        "base_models": ["Qwen/Qwen3-VL-2B-Instruct", "facebook/sam3"],
        "contains_sam3_base": False, "contains_optimizer_state": False,
        "methods": manifest, "contains_absolute_paths": False,
    })
    checksums(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-experiment-root", type=Path, required=True)
    parser.add_argument("--opd-root", type=Path, required=True)
    parser.add_argument("--self-evolution-runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.expanduser().resolve()
    stage_experts(args.expert_experiment_root.expanduser().resolve(), root / "expert_checkpoints")
    stage_distillation(args.opd_root.expanduser().resolve(), root / "distillation_checkpoints")
    stage_self_evolution(args.self_evolution_runs_root.expanduser().resolve(), root / "self_evolution_checkpoints")
    print(json.dumps({
        "expert_checkpoints": str(root / "expert_checkpoints"),
        "distillation_checkpoints": str(root / "distillation_checkpoints"),
        "self_evolution_checkpoints": str(root / "self_evolution_checkpoints"),
    }, indent=2))


if __name__ == "__main__":
    main()
