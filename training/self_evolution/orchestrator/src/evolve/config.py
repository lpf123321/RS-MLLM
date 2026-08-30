from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


def validate_config(config: dict[str, Any]) -> None:
    trajectory_workers = int(config.get("trajectory_workers", 8))
    if trajectory_workers < 1:
        raise ValueError("trajectory_workers must be at least 1")

    required_repositories = {"cvsearch", "sam3_lora", "vision_opd"}
    repositories = config.get("repositories", {})
    missing = required_repositories.difference(repositories)
    if missing:
        raise ValueError(f"Missing required repositories: {sorted(missing)}")
    source_markers = {
        "cvsearch": ("cvsearch/CVSearch.py", "sam3/model_builder.py"),
        "sam3_lora": ("sam3/model_builder.py", "lora_layers.py"),
        "vision_opd": ("verl/trainer/main_ppo.py",),
    }
    for name in sorted(required_repositories):
        path = Path(repositories[name]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{name} source directory is missing: {path}")
        missing_markers = [relative for relative in source_markers[name] if not (path / relative).is_file()]
        if missing_markers:
            raise FileNotFoundError(
                f"{name} source directory is incomplete ({', '.join(missing_markers)}): {path}"
            )

    method = config.get("validated_sam_method", {})
    method_root = Path(method.get("root", "")).expanduser().resolve()
    required_method_files = (
        "scripts/train_vopd_bbox_lora.py",
        "scripts/vopd_negative_common.py",
        "scripts/evaluate_vopd_selectivity.py",
        "scripts/merge_vopd_lora_checkpoint.py",
    )
    for relative in required_method_files:
        if not (method_root / relative).is_file():
            raise FileNotFoundError(f"Validated SAM method file is missing: {method_root / relative}")
    split_manifest = method.get("split_manifest")
    if split_manifest and not Path(split_manifest).expanduser().resolve().is_file():
        raise FileNotFoundError(f"Validated SAM split_manifest is missing: {split_manifest}")
    for key in ("train_negative_manifest", "valid_negative_manifest"):
        path = method.get(key)
        if path and not Path(path).expanduser().resolve().is_file():
            raise FileNotFoundError(f"Validated SAM method {key} is missing: {path}")

    sam = config.get("sam_training", {})
    expected = {
        "rank": 16, "alpha": 32, "dropout": 0.1, "presence_weight": 5.0,
        "negative_ratio": 1.0, "ranking_weight": 2.0, "ranking_margin": 0.2,
        "distill_weight": 0.0,
    }
    mismatched = {key: (sam.get(key), value) for key, value in expected.items() if sam.get(key) != value}
    if mismatched:
        raise ValueError(f"SAM config diverges from validated generic_random_e3 method: {mismatched}")

    cvsearch = config.get("cvsearch", {})
    if cvsearch.get("force_visual_search"):
        lower = float(cvsearch.get("answering_confidence_threshold_lower", 0.0))
        fast = float(cvsearch.get("fast_threshold", 0.0))
        if lower != 0.0 or lower + fast <= 1.0:
            raise ValueError(
                "force_visual_search requires answering_confidence_threshold_lower=0 and "
                "lower+fast_threshold>1 because root confidence is in [-1, 1]"
            )

    negative = config.get("negative_generation", {})
    verifier = Path(negative.get("verifier_model", "")).expanduser().resolve()
    if not verifier.is_dir():
        raise FileNotFoundError(f"Negative verifier model is missing: {verifier}")
    if float(negative.get("absent_probability_threshold", 0.0)) < 0.9:
        raise ValueError("Negative absent_probability_threshold must be >= 0.9")

    model_path = Path(config["models"]["mllm_t0"]).expanduser().resolve()
    model_config_path = model_path / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"MLLM config is missing: {model_config_path}")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    if "Qwen3VLForConditionalGeneration" not in model_config.get("architectures", []):
        raise ValueError(f"MLLM must be Qwen3-VL: {model_path}")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    config = expand(config)
    config["_config_path"] = str(path)
    config["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_config(config)
    return config


def workspace(config: dict[str, Any]) -> Path:
    return Path(config["workspace"]).expanduser().resolve()


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
