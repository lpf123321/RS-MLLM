"""Recompute every supported metric family from raw inference JSONL.

The evaluator stores the raw sample and generated text in each prediction row.
This module deliberately recomputes scores from those two fields instead of
trusting a possibly stale embedded ``score`` object.  It writes one explicit
report containing source-answer, clean, legacy proxy, project-report, and
optional official COCO Caption conventions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    # Existing evaluator modules use sibling absolute imports and support
    # direct path execution. Keep those imports working for ``python -m`` too.
    sys.path.insert(0, str(_EVAL_DIR))
if __package__:
    from .caption_metrics import (
        summarize_caption_proxies,
        summarize_reported_caption_metrics,
    )
    from .coco_caption_metrics import compute_official_coco_caption
    from .schema import Sample
    from .scoring import (
        prediction_letter_distribution,
        score_prediction,
        summarize_predictions,
    )
else:  # direct ``python evaluation/vllm_eval/metrics_from_predictions.py``
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from caption_metrics import (  # type: ignore[no-redef]
        summarize_caption_proxies,
        summarize_reported_caption_metrics,
    )
    from coco_caption_metrics import compute_official_coco_caption  # type: ignore[no-redef]
    from schema import Sample  # type: ignore[no-redef]
    from scoring import (  # type: ignore[no-redef]
        prediction_letter_distribution,
        score_prediction,
        summarize_predictions,
    )


REPORT_FILES = ("metrics.json", "metrics.csv", "metrics.md")
CAPTION_TASK_TYPES = frozenset({"caption", "change_caption"})
CLEAN_STATUSES = frozenset({"keep", "corrected"})


@dataclass(frozen=True)
class LoadedPredictions:
    path: Path
    rows: list[dict[str, Any]]
    selection: str
    source_run_dir: Path


def _sample_key(value: object) -> str:
    """Return the stable ID key used while selecting attempt rows."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"sample.id must be a non-empty string or integer: {value!r}")
    key = str(value)
    if not key:
        raise ValueError("sample.id must be non-empty")
    return key


def _intentional_length_constraint(row: dict[str, Any]) -> bool:
    """Match the documented XLRS Caption 550-token reporting exception."""
    sample = row.get("sample", {})
    return bool(
        row.get("generation_truncated")
        and sample.get("dataset") == "xlrs_caption"
        and sample.get("task_type") == "caption"
        and row.get("max_new_tokens") == 550
    )


def _successful_row(row: dict[str, Any]) -> bool:
    return not row.get("error") and (
        not row.get("generation_truncated") or _intentional_length_constraint(row)
    )


def _caption_eligible(row: dict[str, Any]) -> bool:
    return _successful_row(row)


def _resolve_path(path: Path, *, must_exist: bool = False) -> Path:
    path = Path(path)
    path = path.expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        # Relative CLI paths are repository-relative in every invocation mode;
        # this prevents a caller's cwd from changing which result is scored.
        resolved = (_REPO_ROOT / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield line_number, value


def _validate_and_rescore(
    raw_row: dict[str, Any], *, path: Path, line_number: int
) -> dict[str, Any]:
    raw_sample = raw_row.get("sample")
    if not isinstance(raw_sample, dict):
        raise ValueError(f"{path}:{line_number}: missing object field 'sample'")
    try:
        sample = Sample.from_dict(raw_sample, manifest_dir=path.parent)
        sample.validate(check_images=False)
    except Exception as exc:  # noqa: BLE001
        identifier = raw_sample.get("id", "<missing>")
        raise ValueError(
            f"{path}:{line_number}: invalid sample {identifier}: {exc}"
        ) from exc

    prediction = raw_row.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError(
            f"{path}:{line_number}: sample {sample.id}: 'prediction' must be a string"
        )
    error = raw_row.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError(
            f"{path}:{line_number}: sample {sample.id}: 'error' must be null or string"
        )
    truncated = raw_row.get("generation_truncated", False)
    if not isinstance(truncated, bool):
        raise ValueError(
            f"{path}:{line_number}: sample {sample.id}: 'generation_truncated' must be bool"
        )

    # Do not trust a score serialized by an older scorer or prompt convention.
    row = dict(raw_row)
    row["sample"] = sample.to_dict()
    row["score"] = score_prediction(sample, prediction)
    return row


def load_predictions(input_path: Path) -> LoadedPredictions:
    """Load a finalized predictions file or select latest rows from attempts."""
    input_path = _resolve_path(input_path, must_exist=True)
    if input_path.is_dir():
        predictions = input_path / "predictions.jsonl"
        attempts = input_path / "prediction_attempts.jsonl"
        if predictions.is_file():
            path = predictions
        elif attempts.is_file():
            path = attempts
        else:
            raise FileNotFoundError(
                f"No predictions.jsonl or prediction_attempts.jsonl in {input_path}"
            )
    elif input_path.is_file():
        path = input_path
    else:
        raise ValueError(
            f"Prediction input is neither a file nor directory: {input_path}"
        )

    if path.name == "prediction_attempts.jsonl":
        # Keep first-seen order (the manifest order in evaluator attempts), but
        # select the latest successful attempt. A later failed retry must not
        # erase an earlier valid generation; if no valid attempt exists, retain
        # the latest failed/truncated row so the report exposes the failure.
        order: list[str] = []
        all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sample_fingerprints: dict[str, str] = {}
        for line_number, raw_row in _read_jsonl(path):
            raw_sample = raw_row.get("sample")
            identifier = raw_sample.get("id") if isinstance(raw_sample, dict) else None
            try:
                key = _sample_key(identifier)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if key not in all_rows:
                order.append(key)
            row = _validate_and_rescore(raw_row, path=path, line_number=line_number)
            fingerprint = json.dumps(
                row["sample"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            previous_fingerprint = sample_fingerprints.get(key)
            if previous_fingerprint is not None and previous_fingerprint != fingerprint:
                raise ValueError(
                    f"{path}:{line_number}: sample definition changed across attempts: {key}"
                )
            sample_fingerprints[key] = fingerprint
            all_rows[key].append(row)
        rows = []
        for key in order:
            candidates = all_rows[key]
            valid = [row for row in candidates if _successful_row(row)]
            rows.append(valid[-1] if valid else candidates[-1])
        selection = "latest_successful_attempt_per_sample_id_from_prediction_attempts"
    else:
        rows = []
        seen: set[str] = set()
        for line_number, raw_row in _read_jsonl(path):
            row = _validate_and_rescore(raw_row, path=path, line_number=line_number)
            identifier = _sample_key(row["sample"]["id"])
            if identifier in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample ID {identifier}")
            seen.add(identifier)
            rows.append(row)
        selection = "finalized_predictions_rows"

    if not rows:
        raise ValueError(f"Prediction input is empty: {path}")
    return LoadedPredictions(
        path=path,
        rows=rows,
        selection=selection,
        source_run_dir=path.parent,
    )


def _load_adjacent_json(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name in ("run_manifest.json", "run_config.json"):
        candidate = path.parent / name
        if not candidate.is_file():
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            value = {"read_error": f"{type(exc).__name__}: {exc}"}
        loaded[name] = value
    return loaded


def _provenance(input_data: LoadedPredictions) -> dict[str, Any]:
    def digest(value: object) -> str:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    samples = [row["sample"] for row in input_data.rows]
    files: dict[str, Any] = {
        "predictions_path": str(input_data.path),
        "predictions_sha256": _sha256(input_data.path),
        "selection": input_data.selection,
        # These hashes make prompt/reference drift observable when comparing
        # parallel teammate runs, even if the raw file is later relocated.
        "selected_sample_definition_sha256": digest(samples),
        "selected_prompt_sha256": digest(
            [{"id": sample["id"], "prompt": sample["prompt"]} for sample in samples]
        ),
        "selected_references_sha256": digest(
            [
                {"id": sample["id"], "references": sample["references"]}
                for sample in samples
            ]
        ),
    }
    for name, value in _load_adjacent_json(input_data.path).items():
        files[name] = value
        files[f"{name}_sha256"] = _sha256(input_data.path.parent / name)
    return files


def _caption_groups(
    rows: list[dict[str, Any]],
    *,
    include_coco: bool,
    include_spice: bool,
    require_coco: bool,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = row["sample"]
        if sample.get("task_type") in CAPTION_TASK_TYPES:
            groups[f"{sample['dataset']}/{sample['subtask']}"].append(row)

    output: dict[str, Any] = {}
    for group_name in sorted(groups):
        group_rows = groups[group_name]
        source_eligible = [
            row
            for row in group_rows
            if _caption_eligible(row)
        ]
        clean_eligible = [
            row
            for row in source_eligible
            if row["sample"].get("clean_status") in CLEAN_STATUSES
        ]
        scopes: dict[str, Any] = {}
        for scope_name, eligible in (
            ("source_answer", source_eligible),
            ("clean", clean_eligible),
        ):
            exclusions = {
                "generation_errors": sum(bool(row.get("error")) for row in group_rows),
                "generation_truncations": sum(
                    bool(row.get("generation_truncated")) for row in group_rows
                ),
                "intentional_length_constraints": sum(
                    _intentional_length_constraint(row) for row in group_rows
                ),
                "generation_truncations_excluded": sum(
                    bool(row.get("generation_truncated"))
                    and not _intentional_length_constraint(row)
                    for row in group_rows
                ),
                "clean_status_excluded": (
                    sum(
                        row["sample"].get("clean_status") not in CLEAN_STATUSES
                        for row in source_eligible
                    )
                    if scope_name == "clean"
                    else 0
                ),
            }
            item: dict[str, Any] = {
                "samples": len(group_rows),
                "eligible_samples": len(eligible),
                "excluded": exclusions,
            }
            if not eligible:
                item["status"] = "not_applicable"
                item["legacy_proxy"] = {}
                item["project_report"] = {}
                item["official_coco"] = {
                    "status": "not_applicable",
                    "protocol": "pycocoevalcap==1.2",
                    "raw": {},
                    "percent": {},
                    "spice_requested": include_spice if include_coco else False,
                }
            else:
                predictions = [row["prediction"] for row in eligible]
                references = [row["sample"]["references"] for row in eligible]
                item["status"] = "ok"
                item["legacy_proxy"] = summarize_caption_proxies(
                    predictions, references
                )
                item["project_report"] = summarize_reported_caption_metrics(
                    predictions, references
                )
                if include_coco:
                    item["official_coco"] = compute_official_coco_caption(
                        predictions,
                        references,
                        include_spice=include_spice,
                        strict=require_coco,
                    )
                else:
                    item["official_coco"] = {
                        "status": "not_requested",
                        "protocol": "pycocoevalcap==1.2",
                        "raw": {},
                        "percent": {},
                        "spice_requested": False,
                    }
            scopes[scope_name] = item
        output[group_name] = scopes
    return output


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [row["sample"] for row in rows]
    return {
        "prediction_rows": len(rows),
        "generation_errors": sum(bool(row.get("error")) for row in rows),
        "generation_truncations": sum(
            bool(row.get("generation_truncated")) for row in rows
        ),
        "intentional_length_constraints": sum(
            _intentional_length_constraint(row) for row in rows
        ),
        "clean_status": dict(
            sorted(Counter(s["clean_status"] for s in samples).items())
        ),
        "task_types": dict(sorted(Counter(s["task_type"] for s in samples).items())),
        "groups": dict(
            sorted(Counter(f"{s['dataset']}/{s['subtask']}" for s in samples).items())
        ),
    }


def _code_hashes() -> dict[str, str]:
    names = (
        "metrics_from_predictions.py",
        "coco_caption_metrics.py",
        "scoring.py",
        "caption_metrics.py",
    )
    return {name: _sha256(Path(__file__).resolve().parent / name) for name in names}


def build_metric_report(
    input_data: LoadedPredictions,
    *,
    include_coco: bool = True,
    include_spice: bool = False,
    require_coco: bool = False,
) -> dict[str, Any]:
    rows = input_data.rows
    official, clean = summarize_predictions(rows)
    official["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    clean["prediction_letter_distribution"] = prediction_letter_distribution(rows)
    return {
        "schema_version": "rsmllm.metrics.v1",
        "provenance": _provenance(input_data),
        "code_sha256": _code_hashes(),
        "counts": _counts(rows),
        "protocols": {
            "source_answer": official,
            "clean": clean,
        },
        "caption_metric_groups": _caption_groups(
            rows,
            include_coco=include_coco,
            include_spice=include_spice,
            require_coco=require_coco,
        ),
        "metric_conventions": {
            "source_answer": "Recomputed task parser with original references/labels; includes all rows in existing summary semantics.",
            "clean": "Recomputed task parser after excluding generation failures/truncations and non-keep/corrected samples.",
            "project_report": "evaluation.metrics BLEU/METEOR/ROUGE-L/CIDEr-D implementation; report percent fields multiply raw values by 100.",
            "legacy_proxy": "Dependency-light lexical smoke metrics; cider_tfidf_proxy is not official COCO CIDEr.",
            "official_coco": "pycocoevalcap 1.2 with Stanford PTBTokenizer; raw CIDEr scale is 0-10, all percent fields are explicit display units.",
        },
    }


def _add_csv_row(
    output: list[dict[str, Any]],
    *,
    scope: str,
    group: str,
    family: str,
    metric: str,
    raw_value: object,
    display_value: object,
    display_unit: str,
    samples: int,
    eligible_samples: int,
    status: str = "ok",
) -> None:
    if raw_value is None:
        return
    output.append(
        {
            "scope": scope,
            "group": group,
            "family": family,
            "metric": metric,
            "raw_value": f"{float(raw_value):.12g}",
            "display_value": f"{float(display_value):.12g}",
            "display_unit": display_unit,
            "samples": samples,
            "eligible_samples": eligible_samples,
            "status": status,
        }
    )


def _flatten_csv(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, summary_key in (("source_answer", "source_answer"), ("clean", "clean")):
        summary = report["protocols"][summary_key]
        for group, item in sorted(summary.get("groups", {}).items()):
            samples = int(item.get("samples", 0))
            eligible = int(
                item.get("scoreable_samples", item.get("eligible_scoreable_samples", 0))
            )
            if item.get("accuracy") is not None:
                _add_csv_row(
                    rows,
                    scope=scope,
                    group=group,
                    family="discrete",
                    metric="accuracy",
                    raw_value=item["accuracy"],
                    display_value=float(item["accuracy"]) * 100,
                    display_unit="percent",
                    samples=samples,
                    eligible_samples=eligible,
                )
            bbox = item.get("bbox_metrics")
            if bbox:
                for metric in (
                    "mean_iou",
                    "accuracy_at_0_5",
                    "accuracy_at_0_7",
                    "format_validity",
                ):
                    if bbox.get(metric) is not None:
                        _add_csv_row(
                            rows,
                            scope=scope,
                            group=group,
                            family="bbox",
                            metric=metric,
                            raw_value=bbox[metric],
                            display_value=float(bbox[metric]) * 100,
                            display_unit="percent",
                            samples=samples,
                            eligible_samples=int(bbox.get("samples", eligible)),
                        )

    # All legacy proxy values are fractions except CIDEr, which follows the
    # old project's 0--10 proxy scale.  Both are displayed as percentages by
    # multiplying by 100; raw_value remains untouched.
    proxy_scales = {
        "bleu_1_proxy": 100,
        "bleu_2_proxy": 100,
        "bleu_3_proxy": 100,
        "bleu_4_proxy": 100,
        "meteor_exact_proxy": 100,
        "rouge_l_proxy": 100,
        "cider_tfidf_proxy": 100,
    }
    for group, scopes in sorted(report["caption_metric_groups"].items()):
        for scope, item in sorted(scopes.items()):
            samples = int(item.get("samples", 0))
            eligible = int(item.get("eligible_samples", 0))
            status = str(item.get("status", "unknown"))
            proxy = item.get("legacy_proxy", {})
            for metric, value in proxy.items():
                if metric not in proxy_scales or not isinstance(value, (int, float)):
                    continue
                _add_csv_row(
                    rows,
                    scope=scope,
                    group=group,
                    family="legacy_proxy",
                    metric=metric,
                    raw_value=value,
                    display_value=float(value) * proxy_scales[metric],
                    display_unit="percent",
                    samples=samples,
                    eligible_samples=eligible,
                    status=status,
                )
            project = item.get("project_report", {})
            for metric, value in project.get("raw", {}).items():
                percent = project.get("percent", {}).get(metric)
                _add_csv_row(
                    rows,
                    scope=scope,
                    group=group,
                    family="project_report",
                    metric=metric,
                    raw_value=value,
                    display_value=percent,
                    display_unit="percent",
                    samples=samples,
                    eligible_samples=eligible,
                    status=status,
                )
            coco = item.get("official_coco", {})
            for metric, value in coco.get("raw", {}).items():
                percent = coco.get("percent", {}).get(metric)
                _add_csv_row(
                    rows,
                    scope=scope,
                    group=group,
                    family="official_coco",
                    metric=metric,
                    raw_value=value,
                    display_value=percent,
                    display_unit="percent",
                    samples=samples,
                    eligible_samples=eligible,
                    status=coco.get("status", status),
                )
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# RS-MLLM 自动指标报告",
        "",
        "该文件由原始 `predictions.jsonl` 重算生成；不会信任输入行中已有的 `score` 字段。",
        "",
        f"- 输入：`{report['provenance']['predictions_path']}`",
        f"- 输入 SHA-256：`{report['provenance']['predictions_sha256']}`",
        f"- prediction 行数：{report['counts']['prediction_rows']}",
        f"- generation errors：{report['counts']['generation_errors']}",
        f"- generation truncations：{report['counts']['generation_truncations']}",
        "",
        "## 离散 / 定位指标",
        "",
        "| scope | group | metric | raw fraction | display (%) |",
        "|---|---|---|---:|---:|",
    ]
    for row in _flatten_csv(report):
        if row["family"] not in {"discrete", "bbox"}:
            continue
        lines.append(
            f"| {row['scope']} | {row['group']} | {row['family']}.{row['metric']} | "
            f"{row['raw_value']} | {row['display_value']} |"
        )

    lines.extend(
        [
            "",
            "## Caption 指标",
            "",
            "| scope | group | family | metric | raw | display | unit | status |",
            "|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in _flatten_csv(report):
        if row["family"] not in {
            "legacy_proxy",
            "project_report",
            "official_coco",
        }:
            continue
        lines.append(
            f"| {row['scope']} | {row['group']} | {row['family']} | {row['metric']} | "
            f"{row['raw_value']} | {row['display_value']} | {row['display_unit']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- `legacy_proxy.cider_tfidf_proxy` 是旧 lexical TF-IDF proxy，不是官方 COCO CIDEr。",
            "- `project_report` 使用仓库 `evaluation.metrics` 实现；`official_coco` 使用 `pycocoevalcap==1.2` 与 PTBTokenizer。",
            "- 官方 COCO 的 CIDEr 原始范围为 0–10；表中 `display` 是为报告统一展示而乘 100 后的值。",
            "- `source_answer` 与 `clean` 同时输出，不能把不同清单、prompt、reference 或 semantic judge 协议的数值直接相减。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    report: dict[str, Any], output_dir: Path, *, refuse_existing: bool
) -> dict[str, Any]:
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in REPORT_FILES]
    if refuse_existing and any(path.exists() for path in targets):
        raise FileExistsError(
            "Refusing to overwrite metric report files: "
            + ", ".join(str(path) for path in targets if path.exists())
        )

    report["output"] = {
        "directory": str(output_dir),
        "files": [str(path) for path in targets],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_rows = _flatten_csv(report)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope",
                "group",
                "family",
                "metric",
                "raw_value",
                "display_value",
                "display_unit",
                "samples",
                "eligible_samples",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    _write_markdown(report, output_dir / "metrics.md")
    return report


def write_metric_report(
    predictions: Path,
    output_dir: Path | None = None,
    *,
    include_coco: bool = True,
    include_spice: bool = False,
    require_coco: bool = False,
    refuse_existing: bool = True,
) -> dict[str, Any]:
    """Load raw results, recompute metrics, and write JSON/CSV/Markdown."""
    loaded = load_predictions(predictions)
    if output_dir is None:
        output_dir = loaded.source_run_dir / "metrics"
    report = build_metric_report(
        loaded,
        include_coco=include_coco,
        include_spice=include_spice,
        require_coco=require_coco,
    )
    return _write_report(report, output_dir, refuse_existing=refuse_existing)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从原始 predictions.jsonl 自动重算所有支持的指标口径"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predictions",
        type=Path,
        help="predictions.jsonl，或包含 predictions.jsonl/attempts 的结果目录",
    )
    group.add_argument(
        "--run-dir",
        type=Path,
        help="结果目录；等价于 --predictions <run-dir>",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--spice",
        action="store_true",
        help="额外运行官方 COCO SPICE（需要 Stanford CoreNLP/Java 资源）",
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="不运行 pycocoevalcap，报告中记录 official_coco=not_requested",
    )
    parser.add_argument(
        "--require-coco",
        action="store_true",
        help="COCO scorer 不可用或部分失败时以错误退出",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="存在不可接受的 generation error/truncation 时拒绝生成报告；XLRS Caption 550-token 例外保留",
    )
    args = parser.parse_args()
    if args.skip_coco and args.require_coco:
        parser.error("--skip-coco and --require-coco cannot be used together")
    input_path = args.predictions or args.run_dir
    assert input_path is not None
    loaded = load_predictions(input_path)
    if args.require_complete:
        bad = [
            row["sample"]["id"]
            for row in loaded.rows
            if not _successful_row(row)
        ]
        if bad:
            raise ValueError(
                f"Incomplete predictions ({len(bad)} rows); first IDs: {bad[:5]}"
            )
    report = build_metric_report(
        loaded,
        include_coco=not args.skip_coco,
        include_spice=args.spice and not args.skip_coco,
        require_coco=args.require_coco,
    )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = loaded.source_run_dir / "metrics"
    report = _write_report(report, output_dir, refuse_existing=True)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "prediction_rows": report["counts"]["prediction_rows"],
                "caption_groups": len(report["caption_metric_groups"]),
                "generation_errors": report["counts"]["generation_errors"],
                "generation_truncations": report["counts"]["generation_truncations"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
