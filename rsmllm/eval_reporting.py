"""Compact report-facing metric output for one or more evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _metric_percent(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒 ({seconds:.2f} s)"
    if minutes:
        return f"{minutes}分{secs}秒 ({seconds:.2f} s)"
    return f"{secs}秒 ({seconds:.2f} s)"


def inference_time_line(output_dir: Path) -> str | None:
    """Read the persisted wall time spent only inside inference calls."""
    timing_path = output_dir / "inference_time.json"
    if timing_path.is_file():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
            seconds = float(timing["wall_seconds"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return f"推理耗时={_format_duration(seconds)}"
    return None


def key_metric_lines(summary: dict[str, Any]) -> list[str]:
    """Render the key metrics for every dataset/task group in one summary."""
    lines = [
        "样本={samples}，生成错误={errors}，截断={truncations}".format(
            samples=summary.get("prediction_rows", 0),
            errors=summary.get("generation_errors", 0),
            truncations=summary.get("generation_truncations", 0),
        )
    ]
    reported_groups = 0
    for group_name, group in summary.get("groups", {}).items():
        bbox = group.get("bbox_metrics")
        if bbox:
            lines.append(
                f"{group_name}: Acc@0.5={_metric_percent(bbox['accuracy_at_0_5'])}, "
                f"Acc@0.7={_metric_percent(bbox['accuracy_at_0_7'])}, "
                f"mIoU={_metric_percent(bbox['mean_iou'])}, "
                f"格式有效率={_metric_percent(bbox['format_validity'])}"
            )
            reported_groups += 1
            continue

        caption = group.get("caption_metrics")
        if caption and caption.get("percent"):
            metrics = caption["percent"]
            ordered = (
                "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                "METEOR", "ROUGE-L", "CIDEr-D",
            )
            values = [
                f"{name}={float(metrics[name]):.2f}%"
                for name in ordered
                if metrics.get(name) is not None
            ]
            lines.append(f"{group_name}: " + ", ".join(values))
            reported_groups += 1
            continue

        accuracy = group.get("accuracy")
        if accuracy is not None:
            correct = group.get("correct", 0)
            total = group.get("eligible_scoreable_samples", 0)
            lines.append(
                f"{group_name}: Accuracy={_metric_percent(accuracy)} "
                f"({correct}/{total})"
            )
            reported_groups += 1

    if not reported_groups:
        lines.append("未找到可展示的离散、定位或描述指标。")
    return lines


def print_key_metrics(output_dir: Path) -> None:
    """Print one completed evaluation's key metrics."""
    summary_path = output_dir / "clean_summary.json"
    if not summary_path.is_file():
        print(f"[metrics] 未生成正式汇总: {summary_path}", flush=True)
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\n" + "=" * 60, flush=True)
    print("本次评测关键指标（报告/clean 口径）", flush=True)
    for line in key_metric_lines(summary):
        print(f"  {line}", flush=True)
    timing = inference_time_line(output_dir)
    if timing:
        print(f"  {timing}", flush=True)
    print(f"  结果目录: {output_dir}", flush=True)
    print("=" * 60, flush=True)


def print_combined_key_metrics(output_dirs: Iterable[Path]) -> None:
    """Print all successfully completed tasks again at the end of a run."""
    completed = [
        path for path in dict.fromkeys(output_dirs)
        if (path / "clean_summary.json").is_file()
    ]
    if not completed:
        return
    print("\n" + "=" * 72, flush=True)
    print("本轮全部已完成任务汇总（报告/clean 口径）", flush=True)
    for index, output_dir in enumerate(completed, start=1):
        summary = json.loads(
            (output_dir / "clean_summary.json").read_text(encoding="utf-8")
        )
        print(f"\n[{index}] {output_dir.name}", flush=True)
        for line in key_metric_lines(summary):
            print(f"  {line}", flush=True)
        timing = inference_time_line(output_dir)
        if timing:
            print(f"  {timing}", flush=True)
        print(f"  结果目录: {output_dir}", flush=True)
    print("=" * 72, flush=True)
