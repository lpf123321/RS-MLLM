#!/usr/bin/env python3
"""Summarize candidate registry, cache state, and evaluation summaries.

The summary is intentionally registry-driven: candidates never silently disappear
just because they have not been downloaded or evaluated yet.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from candidate_registry import CACHE_MANIFEST, RESULTS_DIR, ROOT, SOURCE_REPORT, iter_candidates


SUMMARY_CSV = RESULTS_DIR / "candidate_eval_summary.csv"
SUMMARY_MD = RESULTS_DIR / "candidate_eval_summary.md"

SUMMARY_COLUMNS = [
    "key", "model_id", "group", "license", "params", "quantization",
    "fit_status", "peak_mem_mb", "avg_infer_s",
    "vrs_vqa_acc", "vrs_refer_pass", "xlrs_acc", "mme_acc",
    "quality_avg", "efficiency_score", "final_score", "decision", "notes",
]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def latest_eval_summaries() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for path in sorted(RESULTS_DIR.glob("candidate_*_summary.json")):
        data = read_json(path, None)
        if not isinstance(data, dict):
            continue
        data["summary_file"] = str(path)
        found.setdefault(data.get("candidate", "unknown"), []).append(data)
    return found


def latest_for_dataset(rows: list[dict], dataset: str | None = None, stage: str | None = None) -> dict:
    filtered = rows
    if dataset is not None:
        filtered = [row for row in filtered if row.get("dataset") in {dataset, "all"}]
    if stage is not None:
        filtered = [row for row in filtered if row.get("stage") == stage]
    return filtered[-1] if filtered else {}


def metric_from(summary: dict, *needles: str) -> float | None:
    accs = summary.get("accuracy_by_task", {}) if isinstance(summary, dict) else {}
    if not isinstance(accs, dict):
        return None
    lowered_needles = tuple(n.lower() for n in needles)
    for task, stat in accs.items():
        task_l = str(task).lower()
        if all(n in task_l for n in lowered_needles) and isinstance(stat, dict) and "acc" in stat:
            return round(float(stat["acc"]), 4)
    return None


def pick_metric(rows: list[dict], dataset: str, *task_needles: str) -> float | None:
    # Prefer full > mini > fit, but accept any explicit latest summary.
    for stage in ("full", "mini", "fit", None):
        summary = latest_for_dataset(rows, dataset, stage)
        value = metric_from(summary, dataset, *task_needles)
        if value is not None:
            return value
        value = metric_from(summary, *task_needles)
        if value is not None:
            return value
    return None


def quality_avg(vrs_vqa: float | None, xlrs: float | None, mme: float | None) -> float | None:
    values = [v for v in (vrs_vqa, xlrs, mme) if v is not None]
    return round(mean(values), 4) if values else None


def legacy_runtime_warnings(summary: dict) -> list[str]:
    """Recover runtime-gated evidence for summaries produced before the field existed."""
    warnings = list(summary.get("runtime_warnings") or [])
    if warnings:
        return warnings
    summary_file = str(summary.get("summary_file", ""))
    match = re.search(r"_(\d+)_summary\.json$", summary_file)
    if not match:
        return warnings
    job_id = match.group(1)
    log_text = ""
    for log_path in ROOT.glob(f"*{job_id}.log"):
        try:
            log_text += log_path.read_text(errors="ignore") + "\n"
        except OSError:
            continue
    if "offloaded to the cpu" in log_text:
        warnings.append(
            "runtime_gated_offload: model completed inference but uses cpu offload, unsuitable for single-A100 efficient deployment"
        )
    if "FP8 quantized models is only supported on GPUs with compute capability >= 8.9" in log_text:
        warnings.append(
            "runtime_gated_fp8_fallback: A100 compute capability 8.0 cannot run this Transformers FP8 path natively"
        )
    return warnings


def fit_status_for(candidate, manifest: dict, rows: list[dict]) -> str:
    if candidate.cache_status == "baseline":
        return "BASELINE"
    entry = manifest.get(candidate.key, {}) if isinstance(manifest, dict) else {}
    if entry.get("status") == "FAILED_DOWNLOAD":
        return "SOURCE_FAILED"
    if not entry and candidate.cache_status == "download-first":
        return "NEEDS_DOWNLOAD"
    if rows:
        latest = rows[-1]
        if legacy_runtime_warnings(latest):
            return "RUNTIME_GATED"
        if latest.get("errors"):
            errors = "; ".join(str(e) for e in latest.get("errors", []))
            if "peak memory" in errors or "out of memory" in errors.lower() or "cuda" in errors.lower():
                return "BLOCKED_RUNTIME"
            return "EVAL_ERRORS"
        return "EVALUATED"
    return "CACHED" if candidate.cache_status == "cached" else "DOWNLOADED"


def deployability_score(status: str, peak_mem_mb: float | None) -> float:
    if status in {"SOURCE_FAILED", "BLOCKED_RUNTIME", "RUNTIME_GATED"}:
        return 0.0
    if peak_mem_mb is None:
        return 0.0
    if peak_mem_mb <= 8192:
        return 1.0
    if peak_mem_mb <= 16384:
        return 0.8
    if peak_mem_mb <= 24576:
        return 0.6
    if peak_mem_mb <= 39000:
        return 0.4
    return 0.0


def efficiency_score(peak_mem_mb: float | None, avg_infer_s: float | None,
                     baseline_peak: float | None, baseline_infer: float | None) -> float | None:
    if peak_mem_mb in (None, 0) or avg_infer_s in (None, 0):
        return None
    mem_term = min(1.0, baseline_peak / peak_mem_mb) if baseline_peak else 0.0
    speed_term = min(1.0, baseline_infer / avg_infer_s) if baseline_infer else 0.0
    if baseline_peak is None and baseline_infer is None:
        return None
    return round(0.5 * mem_term + 0.5 * speed_term, 4)


def final_score(quality: float | None, eff: float | None, deploy: float,
                baseline_quality: float | None) -> float | None:
    if quality is None and eff is None:
        return None
    if quality is None:
        quality_norm = 0.0
    elif baseline_quality and baseline_quality > 0:
        quality_norm = min(1.0, quality / baseline_quality)
    else:
        quality_norm = quality
    return round(0.4 * quality_norm + 0.4 * (eff or 0.0) + 0.2 * deploy, 4)


def decision_for(status: str, quality: float | None, baseline_quality: float | None,
                 peak_mem_mb: float | None, baseline_peak: float | None) -> str:
    if status == "SOURCE_FAILED":
        return "SOURCE_FAILED"
    if status in {"BLOCKED_RUNTIME", "RUNTIME_GATED"}:
        return "BLOCKED_RUNTIME"
    if status in {"NEEDS_DOWNLOAD", "CACHED", "DOWNLOADED"}:
        return "PENDING_GATE"
    if quality is None:
        return "PENDING_METRICS"
    if baseline_quality and quality >= 0.9 * baseline_quality:
        return "KEEP_FULL"
    if baseline_quality and baseline_peak and peak_mem_mb and peak_mem_mb <= 0.5 * baseline_peak and quality >= 0.7 * baseline_quality:
        return "KEEP_LIGHTWEIGHT"
    # If there is no baseline metric yet, keep evaluated candidates visible but do not overclaim.
    if baseline_quality is None:
        return "EVALUATED_NO_BASELINE"
    return "DROP_QUALITY"


def write_source_report(manifest: dict) -> None:
    rows = []
    for candidate in iter_candidates(include_baseline=True):
        rows.append({
            **candidate.to_dict(),
            "manifest": manifest.get(candidate.key, {}) if isinstance(manifest, dict) else {},
        })
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_REPORT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def md_escape(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def build_rows(manifest: dict, evals: dict[str, list[dict]]) -> list[dict]:
    baseline_rows = evals.get("qwen3vl_2b_baseline", [])
    baseline_latest = baseline_rows[-1] if baseline_rows else {}
    baseline_peak = baseline_latest.get("peak_mem_mb")
    baseline_infer = baseline_latest.get("avg_infer_s")
    baseline_vrs = pick_metric(baseline_rows, "vrsbench", "vqa")
    baseline_xlrs = pick_metric(baseline_rows, "xlrs_lite")
    baseline_mme = pick_metric(baseline_rows, "mme")
    baseline_quality = quality_avg(baseline_vrs, baseline_xlrs, baseline_mme)

    output: list[dict] = []
    for candidate in iter_candidates(include_baseline=True):
        rows = evals.get(candidate.key, [])
        latest = rows[-1] if rows else {}
        vrs_vqa = pick_metric(rows, "vrsbench", "vqa")
        vrs_refer = pick_metric(rows, "vrsbench", "refer")
        xlrs = pick_metric(rows, "xlrs_lite")
        mme = pick_metric(rows, "mme")
        quality = quality_avg(vrs_vqa, xlrs, mme)
        peak = latest.get("peak_mem_mb")
        infer = latest.get("avg_infer_s")
        status = fit_status_for(candidate, manifest, rows)
        deploy = deployability_score(status, peak)
        eff = efficiency_score(peak, infer, baseline_peak, baseline_infer)
        score = final_score(quality, eff, deploy, baseline_quality)
        decision = decision_for(status, quality, baseline_quality, peak, baseline_peak)

        notes = candidate.support_status
        runtime_warnings = legacy_runtime_warnings(latest) if latest else []
        if runtime_warnings:
            notes = "; ".join(runtime_warnings)
        elif latest.get("errors"):
            notes = "; ".join(str(e) for e in latest["errors"])
        elif candidate.key in manifest:
            entry = manifest[candidate.key]
            source = entry.get("source")
            cache_path = entry.get("cache_path")
            if source or cache_path:
                notes = f"{source or ''} {cache_path or ''}".strip()

        output.append({
            "key": candidate.key,
            "model_id": candidate.model_id,
            "group": candidate.group,
            "license": candidate.license,
            "params": candidate.params,
            "quantization": candidate.quantization,
            "fit_status": status,
            "peak_mem_mb": peak,
            "avg_infer_s": infer,
            "vrs_vqa_acc": vrs_vqa,
            "vrs_refer_pass": vrs_refer,
            "xlrs_acc": xlrs,
            "mme_acc": mme,
            "quality_avg": quality,
            "efficiency_score": eff,
            "final_score": score,
            "decision": decision,
            "notes": notes,
        })
    return output


def write_csv(rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def write_md(rows: list[dict]) -> None:
    lines = [
        "# Candidate Evaluation Summary",
        "",
        f"Source registry: `{SOURCE_REPORT}`",
        f"Cache manifest: `{CACHE_MANIFEST}`",
        f"CSV summary: `{SUMMARY_CSV}`",
        "",
    ]
    for group in ["baseline", "cached", "download-first"]:
        title = {"baseline": "Baseline reference", "cached": "Cached / immediately testable", "download-first": "Download-first candidates"}[group]
        lines.extend([
            f"## {title}",
            "",
            "| key | model | license | params | quant | status | quality_avg | peak_mem_mb | avg_infer_s | decision | notes |",
            "|---|---|---|---:|---|---|---:|---:|---:|---|---|",
        ])
        for row in rows:
            if row["group"] != group:
                continue
            lines.append(
                "| " + " | ".join([
                    md_escape(row["key"]),
                    md_escape(row["model_id"]),
                    md_escape(row["license"]),
                    md_escape(row["params"]),
                    md_escape(row["quantization"]),
                    md_escape(row["fit_status"]),
                    md_escape(row["quality_avg"]),
                    md_escape(row["peak_mem_mb"]),
                    md_escape(row["avg_infer_s"]),
                    md_escape(row["decision"]),
                    md_escape(row["notes"]),
                ]) + " |"
            )
        lines.append("")
    lines.extend([
        "## Selection gates",
        "",
        "1. Keep only rows with 2026-verified sources, except `qwen3vl_2b_baseline`.",
        "2. Mark any model needing CPU offload or >39GB peak memory as `BLOCKED_RUNTIME` / `RUNTIME_GATED` for A100 40GB.",
        "3. Promote no more than eight candidate keys to full evaluation after fit/mini gates.",
        "4. Caption-only rows are inspectable but excluded from `quality_avg` until caption metrics are implemented for every model.",
        "",
    ])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def selected_for_full(rows: list[dict]) -> list[str]:
    return [row["key"] for row in rows if row["decision"] in {"KEEP_FULL", "KEEP_LIGHTWEIGHT"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-full", action="store_true", help="Print candidate keys selected for full evaluation after summarizing.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_json(CACHE_MANIFEST, {})
    evals = latest_eval_summaries()
    write_source_report(manifest)
    rows = build_rows(manifest, evals)
    write_csv(rows)
    write_md(rows)

    print(f"Wrote {SUMMARY_MD}", flush=True)
    print(f"Wrote {SUMMARY_CSV}", flush=True)
    if args.select_full:
        print(" ".join(selected_for_full(rows)), flush=True)
    else:
        print("\n".join(SUMMARY_MD.read_text(encoding="utf-8").splitlines()[:80]), flush=True)


if __name__ == "__main__":
    main()
