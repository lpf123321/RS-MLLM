from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from rsmllm import config, console, data, models, quantize, router


def test_console_runs_children_from_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, *, check, cwd, env):
        captured.update(command=command, check=check, cwd=cwd, env=env)
        return console.subprocess.CompletedProcess(command, 0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(console.subprocess, "run", fake_run)

    result = console._run_repo(["example"])

    assert result.returncode == 0
    assert captured["cwd"] == str(console.REPO_ROOT)
    assert captured["check"] is False
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(console.REPO_ROOT)


def test_console_simple_forwards_extra_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return 0

    monkeypatch.setattr(console, "_run_repo", fake_run)
    console._cmd_simple("batch", "scripts/run_batch_scan.py", ["--model", "/m"])

    assert calls == [
        [
            console.TRAIN_PY,
            str(console.REPO_ROOT / "scripts" / "run_batch_scan.py"),
            "--model",
            "/m",
        ]
    ]


def test_quantize_resolves_exec_script_and_defaults_from_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class ExecIntercept(RuntimeError):
        pass

    def fake_execv(executable, argv):
        captured.update(executable=executable, argv=argv)
        raise ExecIntercept

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quantize, "get_model", lambda name: "/models/input")
    monkeypatch.setattr(quantize.os, "execv", fake_execv)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize",
            "--method",
            "w8a8-int8",
            "--model",
            "input",
            "--calibration",
            "calibration.jsonl",
        ],
    )

    with pytest.raises(ExecIntercept):
        quantize.main()

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[1] == str(console.REPO_ROOT / "scripts" / "quantize_qwen35_vlm.py")
    output_index = argv.index("--output")
    assert argv[output_index + 1] == str(
        console.REPO_ROOT / "quantized_models" / "out"
    )
    calibration_index = argv.index("--calibration")
    assert argv[calibration_index + 1] == str(
        console.REPO_ROOT / "calibration.jsonl"
    )


def test_router_default_runtime_paths_are_absolute() -> None:
    assert all(Path(path).is_absolute() for path in router.DEFAULT_MODELS.values())
    assert all(Path(path).is_absolute() for path in router.DEFAULT_LORAS.values())
    assert Path(router.DEFAULT_PYTHON).is_absolute()


def test_missing_cluster_fallback_does_not_escape_to_a_nonexistent_root(
    tmp_path: Path,
) -> None:
    local_default = tmp_path / "datasets"
    missing_cluster_root = tmp_path / "not-mounted"
    assert config._settle(
        "__RSMLLM_TEST_MISSING_ROOT__",
        local_default,
        str(missing_cluster_root),
    ) == local_default


def test_relative_model_and_dataset_paths_resolve_from_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    caller = tmp_path / "caller"
    local_model = repo / "models" / "local"
    local_dataset = repo / "datasets" / "local"
    local_model.mkdir(parents=True)
    local_dataset.mkdir(parents=True)
    caller.mkdir()

    monkeypatch.setattr(models, "REPO_ROOT", repo)
    monkeypatch.setattr(data, "REPO_ROOT", repo)
    monkeypatch.chdir(caller)

    assert models.get_model("models/local") == str(local_model.resolve())
    assert data.get_dataset("datasets/local") == str(local_dataset.resolve())
