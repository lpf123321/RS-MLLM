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

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "transfer" / "rs_mllm_training_data"))
    ap.add_argument("--repo", default="yasumi/rs-mllm-datasets")
    ap.add_argument("--token", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="逗号分隔的相对路径子集（可选）")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--flatten", action="store_true",
                    help="去掉 finetune_framework/VRSbench 前缀，VRSbench 顶层文件上传到根目录，"
                         "expert_data/expert_data_v2 子目录保留（避免同名覆盖）")
    ap.add_argument("--clean-nested", action="store_true",
                    help="删除远程 finetune_framework/ 旧嵌套结构（配合 --flatten 使用）")
    args = ap.parse_args()

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if not args.dry_run and not token:
        raise SystemExit("缺少 token：--token xxx 或 export MODELSCOPE_API_TOKEN=sk-xxx")

    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)

    root = resolve_path(args.data_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    only = set(args.only.split(",")) if args.only else None

    if args.clean_nested:
        # 删除远程旧嵌套结构（finetune_framework/VRSbench/...）
        old_pairs = []
        for f in files:
            s = str(f.relative_to(root))
            if s.startswith("finetune_framework/VRSbench/"):
                old_pairs.append(s)
        if old_pairs:
            print(f"[clean] 删除远程旧嵌套 {len(old_pairs)} 个文件", flush=True)
            api.delete_files(repo_id=args.repo, repo_type="dataset",
                             file_paths=old_pairs)
            print("[clean] 完成", flush=True)
        return

    print(f"将上传 {len(files)} 个文件到 dataset: {args.repo}"
          + ("（根目录拍平）" if args.flatten else ""))
    for f in files:
        rel = str(f.relative_to(root))
        if only is not None and rel not in only:
            continue
        target = rel
        if args.flatten:
            # 去掉 finetune_framework/VRSbench/ 前缀，保留 expert_data 等子目录层级
            target = rel.removeprefix("finetune_framework/VRSbench/")
        size = f.stat().st_size / 1e6
        if args.dry_run:
            print(f"[dry-run] -> {target}  ({size:.1f} MB)  <{rel}>")
            continue
        print(f"[upload] -> {target}  ({size:.1f} MB)", flush=True)
        api.upload_file(
            repo_id=args.repo,
            repo_type="dataset",
            path_or_fileobj=str(f),
            path_in_repo=target,
        )
    print("== 上传完成 ==")


if __name__ == "__main__":
    main()