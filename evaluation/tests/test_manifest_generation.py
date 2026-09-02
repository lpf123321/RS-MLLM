from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from rsmllm.data import validate_eval_manifest
from rsmllm import config as rsmllm_config
from rsmllm import data as rsmllm_data
from rsmllm import router_eval
from scripts import build_sample_manifest as builder


def _configure_builder(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    output = root / "evaluation" / "vllm_eval" / "manifests"
    monkeypatch.setattr(builder, "REPO_ROOT", root)
    monkeypatch.setattr(builder, "OUTPUT_DIR", output)
    return output


def _write_vrs_source(root: Path) -> Path:
    source = root / "datasets_data" / "vrsbench_eval.jsonl"
    source.parent.mkdir(parents=True)
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "../../datasets/shared_datasets/VRSBench/images/val/sample.png",
                    },
                    {"type": "text", "text": "[VQA] What is visible?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "A field"}]},
        ]
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    image = root / "datasets" / "shared_datasets" / "VRSBench" / "images" / "val" / "sample.png"
    image.parent.mkdir(parents=True)
    with Image.new("RGB", (32, 24), color="white") as im:
        im.save(image)
    return source


def test_builder_resolves_paths_and_writes_real_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _configure_builder(monkeypatch, tmp_path)
    _write_vrs_source(tmp_path)

    manifest = builder.build(
        "vrsbench", None, None, "../../../datasets/shared_datasets"
    )
    row = json.loads(manifest.read_text(encoding="utf-8").strip())
    image = row["images"][0]

    assert image["width"] == 32
    assert image["height"] == 24
    assert image["path"] == "../../../datasets/shared_datasets/VRSBench/images/val/sample.png"
    assert validate_eval_manifest(manifest) == 1
    assert not list(output.glob(".*.tmp.*"))

    manifest_without_images_root = builder.build("vrsbench", "vqa", None)
    row_without_images_root = json.loads(
        manifest_without_images_root.read_text(encoding="utf-8").strip()
    )
    assert row_without_images_root["images"][0]["width"] == 32
    assert row_without_images_root["images"][0]["height"] == 24


def test_missing_pillow_fails_without_replacing_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _configure_builder(monkeypatch, tmp_path)
    _write_vrs_source(tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "vrsbench_eval.jsonl"
    existing.write_text("old manifest\n", encoding="utf-8")
    monkeypatch.setattr(builder, "Image", None)
    monkeypatch.setattr(builder, "PIL_IMPORT_ERROR", ImportError("test missing PIL"))

    with pytest.raises(builder.ManifestBuildError, match="Pillow is required"):
        builder.build("vrsbench", None, None, "../../../datasets/shared_datasets")

    assert existing.read_text(encoding="utf-8") == "old manifest\n"
    assert not list(output.glob(".*.tmp.*"))


def test_unreadable_image_fails_without_replacing_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _configure_builder(monkeypatch, tmp_path)
    source = tmp_path / "datasets_data" / "vrsbench_eval.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "missing.png"},
                            {"type": "text", "text": "[VQA] What is visible?"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "nothing"}],
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "vrsbench_eval.jsonl"
    existing.write_text("old manifest\n", encoding="utf-8")

    with pytest.raises(builder.ManifestBuildError, match="does not exist"):
        builder.build("vrsbench", None, None)

    assert existing.read_text(encoding="utf-8") == "old manifest\n"
    assert not list(output.glob(".*.tmp.*"))


def test_prepare_eval_rebuilds_invalid_existing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _configure_builder(monkeypatch, tmp_path)
    _write_vrs_source(tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "vrsbench_eval.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "path": "../../../datasets/shared_datasets/VRSBench/images/val/sample.png",
                        "width": 0,
                        "height": 0,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rsmllm_config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        rsmllm_data,
        "ensure_benchmark_data",
        lambda dataset, **kwargs: tmp_path / "datasets" / "shared_datasets" / "VRSBench",
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        builder.build("vrsbench", None, None, "../../../datasets/shared_datasets")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    rebuilt = rsmllm_data.prepare_eval("vrsbench")

    assert rebuilt == manifest
    assert commands and "--images-root" in commands[0]
    row = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert row["images"][0]["width"] == 32
    assert row["images"][0]["height"] == 24


def test_subtask_builder_rejects_zero_dimension_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    image = tmp_path / "sample.png"
    with Image.new("RGB", (8, 8), color="white") as im:
        im.save(image)
    source = manifests / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "1",
                "dataset": "test",
                "subtask": "vqa",
                "task_type": "open_vqa",
                "prompt": "question",
                "images": [
                    {
                        "path": "../sample.png",
                        "role": "none",
                        "width": 0,
                        "height": 8,
                    }
                ],
                "references": ["answer"],
                "schema_version": "1.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(router_eval, "MANIFESTS", manifests)

    with pytest.raises(ValueError, match="invalid dimensions"):
        router_eval.build_subtask_manifest("source.jsonl", "open_vqa", "source_vqa")
    assert not (manifests / "source__source_vqa.jsonl").exists()
