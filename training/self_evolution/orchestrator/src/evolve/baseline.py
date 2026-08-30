from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .config import workspace
from .io import write_json


def repository_snapshot(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    return {"path": str(path), "commit": git("rev-parse", "HEAD"), "status_porcelain": git("status", "--porcelain=v1")}


def snapshot_repositories(config: dict[str, Any], label: str) -> dict[str, Any]:
    snapshots = {
        name: repository_snapshot(path)
        for name, path in config["repositories"].items()
        if name in {"cvsearch", "sam3_lora", "vision_opd"}
    }
    write_json(workspace(config) / "manifests" / f"repositories_{label}.json", snapshots)
    return snapshots
