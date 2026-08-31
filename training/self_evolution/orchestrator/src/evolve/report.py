from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import workspace
from .io import atomic_write_text, read_jsonl


def generate_report(config: dict[str, Any]) -> Path:
    root = workspace(config)
    metrics_files = sorted(root.glob("runs/**/metrics.json"))
    decision_files = sorted(root.glob("runs/**/promotion.json"))
    lines = [
        "# CVSearch × SAM3-LoRA × Vision-OPD 自进化报告", "",
        "模型、轨迹、数据和日志均写入独立 workspace；CVSearch 上游仅包含已批准的结构化 trace/视觉 artifact 最小接口修改，SAM3-LoRA 与 Vision-OPD 上游未改动。", "",
        "## 配置", "", f"- 配置文件：`{config['_config_path']}`", f"- seed：`{config['seed']}`", "",
        "## 每轮指标", "",
    ]
    if not metrics_files:
        lines.append("尚未产生完整轮次指标。")
    for path in metrics_files:
        lines.extend([f"### {path.parent.relative_to(root)}", "", "```json", path.read_text(encoding="utf-8").strip(), "```", ""])
    lines.extend(["## 晋升记录", ""])
    if not decision_files:
        lines.append("尚未产生晋升决定。")
    for path in decision_files:
        lines.extend([f"### {path.parent.relative_to(root)}", "", "```json", path.read_text(encoding="utf-8").strip(), "```", ""])
    output = root / "reports" / "README.md"
    atomic_write_text(output, "\n".join(lines) + "\n")
    return output
