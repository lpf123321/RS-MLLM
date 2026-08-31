from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: object, length: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
