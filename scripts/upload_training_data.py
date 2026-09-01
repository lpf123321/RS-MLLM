#!/usr/bin/env python3
"""Upload training data (transfer/rs_mllm_training_data) to a ModelScope dataset repo.

Keeps the relative structure ``finetune_framework/VRSbench/...`` so that
``scripts/fetch_training_data.sh`` can restore internal names on download.

Requirements: SDK token
  export MODELSCOPE_API_TOKEN=sk-xxx     (or pass --token)

Usage:
  python scripts/upload_training_data.py \
      --data-dir transfer/rs_mllm_training_data \
      --repo yasumi/rs-mllm-datasets [--dry-run] [--only file1,file2]
"""
import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="transfer/rs_mllm_training_data")
    ap.add_argument("--repo", default="yasumi/rs-mllm-datasets")
    ap.add_argument("--token", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="逗号分隔的相对路径子集（可选）")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not args.dry_run and not token:
        raise SystemExit("缺少 token：--token xxx 或 export MODELSCOPE_API_TOKEN=sk-xxx")

    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)

    root = Path(args.data_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    only = set(args.only.split(",")) if args.only else None

    print(f"将上传 {len(files)} 个文件到 dataset: {args.repo}")
    for f in files:
        rel = str(f.relative_to(root))
        if only is not None and rel not in only:
            continue
        size = f.stat().st_size / 1e6
        if args.dry_run:
            print(f"[dry-run] {rel}  ({size:.1f} MB)")
            continue
        print(f"[upload] {rel}  ({size:.1f} MB)", flush=True)
        api.upload_file(
            repo_id=args.repo,
            repo_type="dataset",
            path_or_fileobj=str(f),
            path_in_repo=rel,
        )
    print("== 上传完成 ==")


if __name__ == "__main__":
    main()