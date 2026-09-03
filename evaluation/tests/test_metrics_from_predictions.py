from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.vllm_eval import metrics_from_predictions as metrics


def _sample(
    sample_id: str | int,
    *,
    task_type: str = "open_vqa",
    dataset: str = "vrsbench",
    subtask: str = "vqa",
    prediction_reference: str = "red",
    clean_status: str = "keep",
    references: list[str] | None = None,
) -> dict:
    image_count = 2 if task_type == "change_caption" else 1
    images = [
        {
            "path": f"image_{index}.png",
            "role": "before" if image_count == 2 and index == 0 else (
                "after" if image_count == 2 else "image"
            ),
            "width": 8,
            "height": 8,
        }
        for index in range(image_count)
    ]
    row = {
        "id": sample_id,
        "dataset": dataset,
        "subtask": subtask,
        "task_type": task_type,
        "prompt": f"[{task_type}] prompt {sample_id}",
        "images": images,
        "references": references or [prediction_reference],
        "clean_status": clean_status,
        "schema_version": "1.0",
    }
    if task_type == "single_choice":
        row.update(
            choices={"A": "red", "B": "blue"},
            answer_labels=["A"],
            accepted_labels=["A"],
        )
        row["references"] = []
    return row


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction_row(sample: dict, prediction: str, **extra: object) -> dict:
    return {
        "sample": sample,
        "prediction": prediction,
        "score": {"official_correct": True, "stale": True},
        "error": None,
        "generation_truncated": False,
        **extra,
    }


def test_loader_recomputes_score_instead_of_trusting_embedded_score(tmp_path: Path) -> None:
    sample = _sample("vqa-1", prediction_reference="red")
    path = tmp_path / "predictions.jsonl"
    _write_rows(path, [_prediction_row(sample, "blue")])

    loaded = metrics.load_predictions(path)
    assert loaded.rows[0]["score"]["official_correct"] is False
    report = metrics.build_metric_report(loaded, include_coco=False)
    assert report["protocols"]["source_answer"]["groups"]["vrsbench/vqa"]["correct"] == 0


def test_attempts_choose_latest_successful_row_and_accept_integer_ids(tmp_path: Path) -> None:
    sample = _sample(7, prediction_reference="red")
    path = tmp_path / "prediction_attempts.jsonl"
    _write_rows(
        path,
        [
            _prediction_row(sample, "", error="first failure"),
            _prediction_row(sample, "red", attempt=2),
            _prediction_row(sample, "", error="later failure", attempt=3),
        ],
    )

    loaded = metrics.load_predictions(path)
    assert loaded.selection.startswith("latest_successful_attempt")
    assert len(loaded.rows) == 1
    assert loaded.rows[0]["prediction"] == "red"
    assert loaded.rows[0]["score"]["official_correct"] is True


def test_attempts_reject_sample_definition_drift(tmp_path: Path) -> None:
    first = _sample("same", prediction_reference="red")
    second = _sample("same", prediction_reference="blue")
    path = tmp_path / "prediction_attempts.jsonl"
    _write_rows(path, [_prediction_row(first, "red"), _prediction_row(second, "blue")])

    with pytest.raises(ValueError, match="definition changed"):
        metrics.load_predictions(path)


def test_caption_report_contains_all_metric_families_and_scales(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample(
        "cap-1",
        task_type="caption",
        dataset="vrsbench_caption",
        subtask="caption",
        references=["A red vehicle near a road", "A red car beside the road"],
    )
    path = tmp_path / "predictions.jsonl"
    _write_rows(path, [_prediction_row(sample, "A red vehicle near a road")])
    calls: list[tuple[list[str], list[list[str]]]] = []

    def fake_coco(predictions, references, **kwargs):
        calls.append((predictions, references))
        return {
            "status": "ok",
            "protocol": "pycocoevalcap==1.2",
            "raw": {"CIDEr": 1.25},
            "percent": {"CIDEr": 125.0},
            "spice_requested": kwargs["include_spice"],
        }

    monkeypatch.setattr(metrics, "compute_official_coco_caption", fake_coco)
    output = tmp_path / "metrics"
    report = metrics.write_metric_report(path, output, include_coco=True)

    item = report["caption_metric_groups"]["vrsbench_caption/caption"]["source_answer"]
    assert item["eligible_samples"] == 1
    assert item["legacy_proxy"]["protocol"].endswith("not_official_coco_caption")
    assert "raw" in item["project_report"] and "percent" in item["project_report"]
    assert item["official_coco"]["raw"]["CIDEr"] == 1.25
    assert len(calls) == 2  # source_answer and clean are reported separately
    assert all(
        call == (["A red vehicle near a road"], [sample["references"]])
        for call in calls
    )
    assert (output / "metrics.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "metrics.md").is_file()
    csv_text = (output / "metrics.csv").read_text(encoding="utf-8")
    assert "official_coco" in csv_text
    with pytest.raises(FileExistsError):
        metrics.write_metric_report(path, output, include_coco=False)


def test_xlrs_caption_intentional_truncation_remains_eligible(tmp_path: Path) -> None:
    sample = _sample(
        "xlrs-1",
        task_type="caption",
        dataset="xlrs_caption",
        subtask="caption",
        references=["A rural landscape"],
    )
    path = tmp_path / "predictions.jsonl"
    _write_rows(
        path,
        [_prediction_row(sample, "A rural landscape", generation_truncated=True, max_new_tokens=550)],
    )

    loaded = metrics.load_predictions(path)
    report = metrics.build_metric_report(loaded, include_coco=False)
    assert report["counts"]["intentional_length_constraints"] == 1
    item = report["caption_metric_groups"]["xlrs_caption/caption"]["source_answer"]
    assert item["eligible_samples"] == 1
    assert item["excluded"]["generation_truncations_excluded"] == 0


def test_skip_coco_does_not_import_or_run_coco_scorer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample(
        "cap-skip",
        task_type="caption",
        dataset="vrsbench_caption",
        subtask="caption",
        references=["a road"],
    )
    path = tmp_path / "predictions.jsonl"
    _write_rows(path, [_prediction_row(sample, "a road")])

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("COCO scorer must not run with include_coco=False")

    monkeypatch.setattr(metrics, "compute_official_coco_caption", fail_if_called)
    report = metrics.build_metric_report(metrics.load_predictions(path), include_coco=False)
    coco = report["caption_metric_groups"]["vrsbench_caption/caption"]["source_answer"]["official_coco"]
    assert coco["status"] == "not_requested"
