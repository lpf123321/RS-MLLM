from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from rsmllm.pruning_policy import (
    PRUNING_ENV_ENABLED,
    PRUNING_ENV_KEEP_RATIO,
    PRUNING_ENV_METHOD,
    configure_vllm_pruning_env,
    expert_pruning_spec,
    parse_keep_ratio,
    pruning_spec,
)
from rsmllm.config import vllm_mm_processor_kwargs
from token_compression import build_token_pruner
from rsmllm import console, data, eval_reporting, router, router_eval


def test_task_method_is_preserved_when_user_overrides_keep_ratio() -> None:
    assert pruning_spec("vqa", 0.6).method == "l2norm"
    assert pruning_spec("mcq", 0.6).method == "l2norm"
    assert pruning_spec("change", 0.6).method == "l2norm"
    assert pruning_spec("caption", 0.6).method == "l2norm"
    grounding = pruning_spec("referring", 0.6)
    assert grounding.method == "l2norm"
    assert grounding.keep_ratio == pytest.approx(0.6)


def test_adaptive_and_disabled_policies() -> None:
    assert pruning_spec("vqa", "adaptive").keep_ratio == pytest.approx(0.25)
    assert pruning_spec("referring", "adaptive").keep_ratio == pytest.approx(0.75)
    assert not expert_pruning_spec("caption", 1.0).enabled
    with pytest.raises(ValueError):
        parse_keep_ratio(0)


def test_pruning_environment_is_fail_closed() -> None:
    env = dict(os.environ)
    enabled = pruning_spec("referring", 0.5)
    env["VLLM_PLUGINS"] = "existing_plugin"
    configure_vllm_pruning_env(enabled, env)
    assert env[PRUNING_ENV_ENABLED] == "1"
    assert env[PRUNING_ENV_METHOD] == "l2norm"
    assert env[PRUNING_ENV_KEEP_RATIO] == "0.5"
    assert env["VLLM_PLUGINS"] == "existing_plugin,rs_mllm_token_pruning"
    configure_vllm_pruning_env(pruning_spec("referring", 1.0), env)
    assert PRUNING_ENV_ENABLED not in env
    assert PRUNING_ENV_METHOD not in env
    assert PRUNING_ENV_KEEP_RATIO not in env


def test_route_eval_passes_pruning_to_vllm(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: object) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr(router_eval.subprocess, "run", fake_run)
    spec = pruning_spec("referring", 0.6)
    router_eval.run_eval(
        str(tmp_path / "model"),
        "expert_ground",
        tmp_path / "manifest.jsonl",
        None,
        "bf16",
        output_dir=tmp_path / "results",
        pruning=spec,
    )
    command = commands[0]
    assert command[command.index("--prune-method") + 1] == "l2norm"
    assert command[command.index("--prune-keep-ratio") + 1] == "0.6"


@pytest.mark.parametrize("quantization", ["bf16", "w8a8", "gptq"])
@pytest.mark.parametrize("pruning_enabled", [False, True])
def test_xlrs_grounding_keeps_4096_runtime_for_every_quantization(
    quantization: str,
    pruning_enabled: bool,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: object) -> Result:
        commands.append(command)
        return Result()

    monkeypatch.setattr(router_eval.subprocess, "run", fake_run)
    router_eval.run_eval(
        str(tmp_path / "model"),
        f"expert_ground_{quantization}",
        tmp_path / "xlrs_grounding_test.jsonl",
        None,
        quantization,
        output_dir=tmp_path / "results",
        pruning=(
            pruning_spec("referring", "adaptive")
            if pruning_enabled
            else pruning_spec("referring", 1.0)
        ),
    )

    command = commands[0]
    assert command[command.index("--quantization") + 1] == quantization
    assert command[command.index("--min-pixels") + 1] == "65536"
    assert command[command.index("--max-pixels") + 1] == "16777216"
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[command.index("--max-num-seqs") + 1] == "4"
    assert command[command.index("--prune-method") + 1] == "l2norm"
    assert command[command.index("--prune-keep-ratio") + 1] == (
        "0.75" if pruning_enabled else "1"
    )


@pytest.mark.parametrize("quantization", ["bf16", "w8a8", "gptq"])
@pytest.mark.parametrize("pruning_enabled", [False, True])
@pytest.mark.parametrize(
    ("task_name", "manifest_name", "expected_method", "adaptive_keep_ratio"),
    [
        ("vrsbench-vqa", "vrsbench_eval__vrsbench_vqa.jsonl", "l2norm", 0.25),
        ("mme", "mme_rs.jsonl", "l2norm", 0.25),
        ("xlrs-bench-lite", "xlrs.jsonl", "l2norm", 0.25),
        (
            "vrsbench-referring",
            "vrsbench_eval__vrsbench_referring.jsonl",
            "l2norm",
            0.75,
        ),
        (
            "xlrs-bench-grounding-en",
            "xlrs_grounding_test.jsonl",
            "l2norm",
            0.75,
        ),
        ("levir-cc", "levircc_test.jsonl", "l2norm", 0.50),
        (
            "vrsbench-caption",
            "vrsbench_eval__vrsbench_caption.jsonl",
            "l2norm",
            0.50,
        ),
        ("xlrs-bench-caption", "xlrs_caption_en.jsonl", "l2norm", 0.50),
    ],
)
def test_all_48_eval_combinations_share_the_reference_protocol(
    quantization: str,
    pruning_enabled: bool,
    task_name: str,
    manifest_name: str,
    expected_method: str,
    adaptive_keep_ratio: float,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    class Result:
        returncode = 0

    def fake_run(command: list[str], **_kwargs: object) -> Result:
        commands.append(command)
        return Result()

    spec = pruning_spec(
        router_eval.ROUTE_TASK[task_name],
        "adaptive" if pruning_enabled else 1.0,
    )
    monkeypatch.setattr(router_eval.subprocess, "run", fake_run)
    router_eval.run_eval(
        str(tmp_path / "model"),
        f"profile_{quantization}",
        tmp_path / manifest_name,
        None,
        quantization,
        output_dir=tmp_path / "results",
        pruning=spec,
    )

    command = commands[0]
    expected_runtime = router_eval.evaluation_runtime_config(
        tmp_path / manifest_name,
        pruning=spec,
    )
    assert command[command.index("--quantization") + 1] == quantization
    assert command[command.index("--prune-method") + 1] == expected_method
    assert float(command[command.index("--prune-keep-ratio") + 1]) == pytest.approx(
        adaptive_keep_ratio if pruning_enabled else 1.0
    )
    for option, key in (
        ("--min-pixels", "min_pixels"),
        ("--max-pixels", "max_pixels"),
        ("--max-model-len", "max_model_len"),
        ("--batch-size", "batch_size"),
        ("--max-num-seqs", "max_num_seqs"),
    ):
        assert int(command[command.index(option) + 1]) == expected_runtime[key]
    if task_name == "xlrs-bench-grounding-en":
        assert expected_runtime["max_pixels"] == 16_777_216
        assert expected_runtime["max_model_len"] == 32_768
    else:
        assert expected_runtime["max_pixels"] == router_eval.REPORT_CONF["max_pixels"]


def test_single_console_yes_uses_task_default_pruning(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(
        ["single", "grounding", "bf16", "yes", "vrsbench-referring"]
    )
    captured = []
    manifest = tmp_path / "vrsbench_referring.jsonl"

    monkeypatch.setattr(console, "_ask", lambda *_args, **_kwargs: next(answers))
    monkeypatch.setattr(console, "resolve_model", lambda *_args: str(tmp_path))
    monkeypatch.setattr(data, "prepare_eval", lambda *_args: None)
    monkeypatch.setattr(
        console, "build_subtask_manifest", lambda *_args: manifest
    )
    monkeypatch.setattr(
        console, "new_evaluation_output_dir", lambda *_args: tmp_path / "out"
    )
    monkeypatch.setattr(
        console,
        "run_eval",
        lambda *_args, **kwargs: captured.append(kwargs["pruning"]) or 0,
    )
    monkeypatch.setattr(console, "print_combined_key_metrics", lambda *_args: None)

    console._cmd_eval()
    assert len(captured) == 1
    assert captured[0].method == "l2norm"
    assert captured[0].keep_ratio == pytest.approx(0.75)
    assert captured[0].policy == "task_adaptive"


def test_console_pruning_switch_accepts_only_yes_or_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(["0.5", "adaptive", "yes", "no"])
    monkeypatch.setattr(console, "_ask", lambda *_args, **_kwargs: next(answers))
    assert console._ask_pruning_enabled() is True
    assert console._ask_pruning_enabled() is False


def test_route_console_no_disables_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(["route", "bf16", "no", "10"])
    captured: list[list[str]] = []
    monkeypatch.setattr(console, "_ask", lambda *_args, **_kwargs: next(answers))
    monkeypatch.setattr(
        console,
        "_run_repo",
        lambda command, **_kwargs: captured.append(command)
        or SimpleNamespace(returncode=0),
    )
    console._cmd_eval()
    command = captured[0]
    assert command[command.index("--prune-keep-ratio") + 1] == "1.0"
    assert command[command.index("--limit") + 1] == "10"


def test_metric_summary_includes_recorded_inference_time(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "clean_summary.json").write_text(
        json.dumps(
            {
                "prediction_rows": 1,
                "generation_errors": 0,
                "generation_truncations": 0,
                "groups": {},
            }
        )
    )
    (tmp_path / "inference_time.json").write_text(
        json.dumps({"wall_seconds": 65.25})
    )
    eval_reporting.print_key_metrics(tmp_path)
    assert "推理耗时=1分5秒 (65.25 s)" in capsys.readouterr().out


def test_xlrs_grounding_l2_pruning_preserves_4096_profile(tmp_path) -> None:
    manifest = tmp_path / "xlrs_grounding_test.jsonl"
    baseline = router_eval.evaluation_runtime_config(manifest)
    pruned = router_eval.evaluation_runtime_config(
        manifest, pruning=pruning_spec("referring", 0.75)
    )
    assert baseline["max_pixels"] == 16_777_216
    assert pruned == baseline
    assert pruned["max_model_len"] == 32_768
    assert pruned["batch_size"] == 4
    assert pruned["max_num_seqs"] == 4


@pytest.mark.parametrize("method", ["l2norm", "scope_l2"])
def test_vllm_plugin_selectors_match_shared_implementation(method: str) -> None:
    plugin = pytest.importorskip("rs_mllm_vllm_plugin.model")
    generator = torch.Generator().manual_seed(2026)
    features = torch.randn(12, 8, generator=generator)
    expected = build_token_pruner(method).select(features, 6)
    actual = plugin.select_token_indices(features, method, 0.5)
    assert torch.equal(actual, expected)


def test_vllm_plugin_builds_multi_image_mrope_from_pruned_spans() -> None:
    plugin = pytest.importorskip("rs_mllm_vllm_plugin.model")

    def feature(offset: int, length: int) -> SimpleNamespace:
        return SimpleNamespace(
            modality="image",
            mm_position=SimpleNamespace(offset=offset, length=length),
            data={
                "image_grid_thw": SimpleNamespace(
                    data=torch.tensor([1, 4, 4])
                )
            },
        )

    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    positions, _delta = plugin.pruned_image_mrope_input_positions(
        list(range(20)), [feature(2, 2), feature(8, 2)], config
    )
    assert positions.shape == (3, 20)


def test_vllm_plugin_rewinds_exact_image_prefill_boundary() -> None:
    plugin = pytest.importorskip("rs_mllm_vllm_plugin.model")
    vision_start = 99
    assert plugin.normalize_image_prefill_boundary(
        [1, vision_start, 7, 7], 2, vision_start
    ) == 1
    assert plugin.normalize_image_prefill_boundary(
        torch.tensor([1, vision_start, 7, 7]), 3, vision_start
    ) == 3
    assert plugin.normalize_image_prefill_boundary(
        [1, vision_start, 7, 7], 5, vision_start
    ) == 5


def test_online_router_passes_plugin_policy_to_vllm(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace()

    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(router, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(router, "require_vllm_plugin", lambda: None)
    monkeypatch.setattr(router.subprocess, "Popen", fake_popen)
    router.start_expert(
        "grounding",
        str(model),
        8002,
        python="python",
        pruning=pruning_spec("referring", 0.6),
    )
    command = captured["command"]
    env = captured["env"]
    assert isinstance(command, list)
    assert command[command.index("--video-pruning-rate") + 1] == "0.4"
    mm_kwargs = json.loads(command[command.index("--mm-processor-kwargs") + 1])
    assert mm_kwargs["truncation"] is False
    assert isinstance(env, dict)
    assert env[PRUNING_ENV_METHOD] == "l2norm"
    assert env[PRUNING_ENV_KEEP_RATIO] == "0.6"


def test_vllm_multimodal_kwargs_disable_embedded_tokenizer_truncation() -> None:
    assert vllm_mm_processor_kwargs(65_536, 16_777_216) == {
        "min_pixels": 65_536,
        "max_pixels": 16_777_216,
        "truncation": False,
    }
