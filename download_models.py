#!/usr/bin/env python3
"""Download RS-MLLM candidate models on the login node, never inside a GPU job."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from candidate_registry import CACHE_MANIFEST, RESULTS_DIR, iter_candidates, get_candidate


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def first_snapshot(path: Path) -> Path:
    snapshots = path / "snapshots"
    if snapshots.exists():
        entries = sorted(snapshots.iterdir())
        if entries:
            return entries[-1]
    return path


def load_manifest() -> dict:
    if not CACHE_MANIFEST.exists():
        return {}
    try:
        return json.loads(CACHE_MANIFEST.read_text())
    except json.JSONDecodeError:
        return {}


def save_manifest(manifest: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def download_modelscope(candidate, force: bool) -> Path | None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:
        print(f"  ModelScope unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None

    mid = candidate.default_modelscope_id
    if not mid:
        return None
    expected = candidate.expected_modelscope_path
    if expected and expected.exists() and not force:
        print(f"  ModelScope cache exists: {expected} ({dir_size_gb(expected)} GB)", flush=True)
        return first_snapshot(expected)
    try:
        path = Path(snapshot_download(mid, cache_dir=str(candidate.expected_modelscope_path.parents[1])))
        print(f"  ModelScope -> {path}", flush=True)
        return path
    except Exception as exc:
        print(f"  ModelScope failed for {mid}: {type(exc).__name__}: {exc}", flush=True)
        return None


def download_huggingface(candidate, force: bool) -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        print(f"  Hugging Face hub unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None

    repo_id = candidate.default_hf_id
    local_dir = candidate.expected_modelscope_path
    if local_dir is None:
        return None
    if local_dir.exists() and not force:
        print(f"  HF-compatible cache exists: {local_dir} ({dir_size_gb(local_dir)} GB)", flush=True)
        return first_snapshot(local_dir)
    try:
        path = Path(snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False))
        print(f"  Hugging Face -> {path}", flush=True)
        return path
    except Exception as exc:
        print(f"  Hugging Face failed for {repo_id}: {type(exc).__name__}: {exc}", flush=True)
        return None


def selected_candidates(keys: list[str] | None, include_baseline: bool):
    if keys:
        for key in keys:
            yield get_candidate(key)
        return
    for candidate in iter_candidates(include_baseline=include_baseline):
        if candidate.cache_status == "baseline":
            continue
        yield candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", help="Candidate key to download; repeatable.")
    parser.add_argument("--group", choices=["cached", "download-first", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report cache/manifest state without downloading.")
    args = parser.parse_args()

    manifest = load_manifest()
    failures: dict[str, str] = {}
    for candidate in selected_candidates(args.candidate, include_baseline=False):
        if args.group != "all" and candidate.group != args.group:
            continue
        print(f"\n=== {candidate.key}: {candidate.model_id} ===", flush=True)
        if args.dry_run:
            entry = manifest.get(candidate.key, {}) if isinstance(manifest, dict) else {}
            manifest_path = Path(entry["cache_path"]) if entry.get("cache_path") else None
            expected = candidate.expected_modelscope_path
            resolved = manifest_path if manifest_path and manifest_path.exists() else first_snapshot(expected) if expected else None
            status = "cached" if resolved and resolved.exists() else "missing"
            print(json.dumps({
                "key": candidate.key,
                "group": candidate.group,
                "model_id": candidate.model_id,
                "status": status,
                "manifest_status": entry.get("status"),
                "cache_path": str(resolved) if resolved else None,
                "expected_path": str(expected) if expected else None,
            }, ensure_ascii=False, indent=2), flush=True)
            continue
        start = time.time()
        path = download_modelscope(candidate, args.force)
        source = "modelscope"
        if path is None:
            path = download_huggingface(candidate, args.force)
            source = "huggingface"
        if path is None:
            failures[candidate.key] = "download failed from ModelScope and Hugging Face"
            manifest[candidate.key] = {
                "model_id": candidate.model_id,
                "status": "FAILED_DOWNLOAD",
                "source": None,
                "cache_path": None,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            size = dir_size_gb(path if path.exists() else path.parent)
            manifest[candidate.key] = {
                "model_id": candidate.model_id,
                "status": "OK",
                "source": source,
                "cache_path": str(path),
                "size_gb": size,
                "seconds": round(time.time() - start, 1),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  OK: {path} ({size} GB)", flush=True)
        save_manifest(manifest)

    print(f"\nManifest: {CACHE_MANIFEST}", flush=True)
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(2)
    print("All requested downloads are cached.", flush=True)


if __name__ == "__main__":
    main()
