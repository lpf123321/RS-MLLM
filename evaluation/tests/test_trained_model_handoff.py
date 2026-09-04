from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "evaluation" / "vllm_eval"
for path in (REPO_ROOT, EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rsmllm import models, router, router_eval  # noqa: E402
from model_policy import load_derived_profile, verify_derived_model_content  # noqa: E402
from scripts import preflight_training35 as training_preflight  # noqa: E402
from scripts import publish_trained_experts as publish  # noqa: E402


def _fake_merged_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"test weights")
    (path / "composition_manifest.json").write_text(
        json.dumps({"operation": "base + peft_lora"}) + "\n",
        encoding="utf-8",
    )


def test_modelscope_cache_is_resolved_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / ".models"
    model_id = models.MODEL_REGISTRY["expert_caption"]
    snapshot = cache / "models" / model_id.replace("/", "--") / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(models, "MODELS_ROOT", tmp_path / "published")
    monkeypatch.setenv("MODELSCOPE_OFFLINE", "1")
    assert models.get_model("expert_caption", cache_dir=str(cache)) == str(snapshot.resolve())


def test_publish_profiles_are_accepted_by_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    five_stage = tmp_path / "five-stage"
    expert = tmp_path / "continued"
    sources = {
        "expert_general": expert / "expert_general_full",
        "expert_ground": expert / "expert_ground_full",
        "expert_change": five_stage / "a2b_change",
        "expert_caption": five_stage / "caption",
    }
    for source in sources.values():
        _fake_merged_model(source)

    models_root = tmp_path / "models"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_trained_experts.py",
            "--five-stage-root",
            str(five_stage),
            "--expert-root",
            str(expert),
            "--models-root",
            str(models_root),
        ],
    )
    publish.main()

    for name, source in sources.items():
        profile_path = source / "evaluation_profile.json"
        assert (models_root / name).resolve() == source.resolve()
        profile = load_derived_profile(profile_path)
        result = verify_derived_model_content(profile)
        assert result["model_path"] == str(source.resolve())
        assert profile.key.startswith(name + "_")
    report = json.loads((models_root / "TRAINED_EXPERTS_MANIFEST.json").read_text())
    assert set(report["evaluation_profiles"]) == set(sources)


def test_trained_composition_lineage_is_accepted(tmp_path: Path) -> None:
    _fake_merged_model(tmp_path)
    router_eval._assert_canonical_lineage(tmp_path, "general", "bf16")
    router_eval._assert_canonical_lineage(tmp_path, "grounding", "bf16")


def test_run_eval_selects_generated_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    profile = model / "evaluation_profile.json"
    profile.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(router_eval.subprocess, "run", fake_run)
    router_eval.run_eval(
        str(model),
        "expert_general",
        tmp_path / "manifest.jsonl",
        None,
        "none",
        output_dir=tmp_path / "results",
    )
    assert calls
    assert "--derived-profile" in calls[0]
    assert str(profile) in calls[0]
    assert "--model-profile" not in calls[0]


def test_three_gpu_router_reserves_for_two_experts_per_gpu() -> None:
    mapping = router.expert_device_map(["0", "1", "2"])
    assert mapping["general"] == "0"
    assert mapping["caption"] == "0"
    assert router.default_gpu_memory_utilization(3) == pytest.approx(0.42)


def test_continuation_preflight_requires_only_its_two_input_experts() -> None:
    assert set(training_preflight.TRAINING_MODELS) == {
        "Qwen3.5-4B",
        "expert_general",
        "expert_ground",
    }
