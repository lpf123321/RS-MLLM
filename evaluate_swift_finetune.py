#!/usr/bin/env python3
"""Compare paired ms-swift base and adapter inference JSONL outputs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_results import normalize_text, rouge_l_f1, token_f1

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def clean_response(text: str) -> str:
    return re.sub(r"\s+", " ", THINK_RE.sub(" ", str(text))).strip()


def user_prompt(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("row has no user message")


def image_paths(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(image.get("path", "")) for image in row.get("images", []))


def pair_key(row: dict[str, Any]) -> tuple[str, tuple[str, ...], str]:
    return user_prompt(row), image_paths(row), str(row.get("labels", ""))


def task_name(prompt: str) -> str:
    if "[VQA]" in prompt:
        return "vqa"
    if "[CAP]" in prompt:
        return "caption"
    if "[CCD]" in prompt or "[CD]" in prompt:
        return "change_caption"
    if "[CDVQA]" in prompt:
        return "change_vqa"
    raise ValueError(f"Unknown task marker in prompt: {prompt[:80]!r}")


def metric_values(task: str, prediction: str, reference: str) -> dict[str, float]:
    values = {"token_f1": token_f1(prediction, [reference])}
    if task in {"vqa", "change_vqa"}:
        values["exact_match"] = float(normalize_text(prediction) == normalize_text(reference))
    if task in {"caption", "change_caption"}:
        values["rouge_l_f1"] = rouge_l_f1(prediction, [reference])
    return values


def main() -> int:
    args = parse_args()
    base_rows = load_jsonl(args.base)
    adapter_rows = load_jsonl(args.adapter)
    if len(base_rows) != len(adapter_rows):
        raise ValueError(f"row-count mismatch: base={len(base_rows)}, adapter={len(adapter_rows)}")

    details: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for index, (base, adapter) in enumerate(zip(base_rows, adapter_rows, strict=True), 1):
        if pair_key(base) != pair_key(adapter):
            raise ValueError(f"paired sample mismatch at row {index}")
        prompt = user_prompt(base)
        task = task_name(prompt)
        reference = str(base.get("labels", ""))
        base_prediction = clean_response(base.get("response", ""))
        adapter_prediction = clean_response(adapter.get("response", ""))
        base_metrics = metric_values(task, base_prediction, reference)
        adapter_metrics = metric_values(task, adapter_prediction, reference)
        for metric in sorted(base_metrics):
            aggregates[(task, metric)].append((base_metrics[metric], adapter_metrics[metric]))
        details.append(
            {
                "row": index,
                "task": task,
                "prompt": prompt,
                "images": list(image_paths(base)),
                "reference": reference,
                "base_prediction": base_prediction,
                "adapter_prediction": adapter_prediction,
                "base_metrics": base_metrics,
                "adapter_metrics": adapter_metrics,
            }
        )

    summary = []
    for (task, metric), pairs in sorted(aggregates.items()):
        base_mean = statistics.fmean(pair[0] for pair in pairs)
        adapter_mean = statistics.fmean(pair[1] for pair in pairs)
        summary.append(
            {
                "task": task,
                "metric": metric,
                "samples": len(pairs),
                "base": base_mean,
                "adapter": adapter_mean,
                "delta": adapter_mean - base_mean,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base_path": str(args.base),
        "adapter_path": str(args.adapter),
        "paired_samples": len(details),
        "summary": summary,
        "details": details,
        "notes": [
            "All metrics are paired on identical prompts, image paths, and references.",
            "Exact match is normalized lowercase alphanumeric text equality.",
            "Caption token-F1 and ROUGE-L are lexical proxy metrics, not semantic judges.",
        ],
    }
    (args.out_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    markdown = [
        "# Qwen3.5-4B 遥感 LoRA 对照评测",
        "",
        f"- 配对样本数：{len(details)}",
        f"- 基座结果：`{args.base}`",
        f"- LoRA 结果：`{args.adapter}`",
        "",
        "| 任务 | 指标 | 样本数 | 基座 | LoRA | 差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        markdown.append(
            f"| {row['task']} | {row['metric']} | {row['samples']} | "
            f"{row['base']:.4f} | {row['adapter']:.4f} | {row['delta']:+.4f} |"
        )
    markdown.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 所有指标均在相同提示词、图像路径和参考答案上配对计算。",
            "- VQA exact match 为小写、去标点、压缩空白后的严格匹配。",
            "- caption 的 token-F1 与 ROUGE-L 仅是词面代理指标，不等价于语义质量评价。",
            "",
        ]
    )
    (args.out_dir / "comparison.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
