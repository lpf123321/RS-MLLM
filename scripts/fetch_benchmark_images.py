#!/usr/bin/env python3
"""从 HuggingFace 官方源按需下载评测图片 -> assets/<dataset>/ (可复现布局).

数据集 -> HF 仓库映射(官方源):
  vrsbench      : xiang709/VRSBench            (Images_val.zip)
  levircc       : lcybuaa/LEVIR-CC             (Levir-CC-dataset.zip)
  xlrs          : initiacms/XLRS-Bench-lite     (arrow, 内嵌图片)
  xlrs_caption  : initiacms/XLRS-Bench_caption_en
  xlrs_grounding: initiacms/XLRS-Bench_visual_grounding_en
  mme           : yifanzhang114/MME-RealWorld   (tar.gz)

布局契约(与评测清单一致):
  assets/<dataset>/<清单中的文件名>          # zip 类: 解压后按清单文件名放置
  assets/xlrs_bench_lite/xlrs_%05d.png      # arrow: 按 index 顺序导出重编号
  assets/xlrs_caption/xlrs_caption_%05d.jpg
  assets/xlrs_grounding/xlrs_vg_%05d.jpg

用法(在 xd1/评测机, 有代理时设 HTTPS_PROXY):
  python scripts/fetch_benchmark_images.py --dataset vrsbench
  python scripts/fetch_benchmark_images.py --all
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"

# dataset -> (HF repo, kind)
SOURCES = {
    "vrsbench": ("xiang709/VRSBench", "zip"),
    "levircc": ("lcybuaa/LEVIR-CC", "zip"),
    "xlrs": ("initiacms/XLRS-Bench-lite", "arrow"),
    "xlrs_caption": ("initiacms/XLRS-Bench_caption_en", "arrow"),
    "xlrs_grounding": ("initiacms/XLRS-Bench_visual_grounding_en", "arrow"),
    "mme": ("yifanzhang114/MME-RealWorld", "tar"),
}

# zip 内需要解压的图片目录前缀(按评测清单实际引用)
ZIP_IMAGE_PREFIX = {
    "vrsbench": "images/",   # Images_val.zip 内 images/val/P0003_0002.png
    "levircc": "images/",    # Levir-CC-dataset.zip 内结构待确认
}


def fetch_zip(dataset: str, repo: str) -> int:
    """下载官方 zip, 按清单文件名解压到 assets/<dataset>/."""
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # 先读清单确定需要的文件名
    manifest = REPO_ROOT / "datasets_data" / {
        "vrsbench": "vrsbench_eval.jsonl",
        "levircc": "levircc_test.jsonl",
    }[dataset]
    needed: set[str] = set()
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            import json
            d = json.loads(line)
            for m in d.get("messages", []):
                for c in m.get("content", []):
                    if c.get("type") == "image":
                        needed.add(Path(c["image"]).name)
    zip_name = {"vrsbench": "Images_val.zip", "levircc": "Levir-CC-dataset.zip"}[dataset]
    print(f"[{dataset}] 下载 {repo}/{zip_name} (仅解压 {len(needed)} 个清单引用文件)...")
    local = hf_hub_download(repo, zip_name, repo_type="dataset", token=None)
    out_dir = ASSETS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(local) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            fname = Path(name).name
            if fname in needed:
                (out_dir / fname).write_bytes(z.read(name))
                written += 1
    print(f"[{dataset}] 解压完成: {written}/{len(needed)} 张 -> {out_dir}")
    return written


def fetch_arrow(dataset: str, repo: str) -> int:
    """下载 arrow 分片, 按 index 顺序导出图片重编号."""
    import pyarrow.ipc as ipc
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".arrow"))
    out_dir = ASSETS / {
        "xlrs": "xlrs_bench_lite",
        "xlrs_caption": "xlrs_caption",
        "xlrs_grounding": "xlrs_grounding",
    }[dataset]
    out_dir.mkdir(parents=True, exist_ok=True)
    # 文件名模板(与评测清单一致)
    tpl = {
        "xlrs": "xlrs_{:05d}.png",
        "xlrs_caption": "xlrs_caption_{:05d}.jpg",
        "xlrs_grounding": "xlrs_vg_{:05d}.jpg",
    }[dataset]
    written = 0
    for f in files:
        local = hf_hub_download(repo, f, repo_type="dataset", token=None)
        with open(local, "rb") as fh:
            reader = ipc.open_stream(fh)
            for batch in reader:
                idxs = batch.column("index").to_pylist()
                imgs = batch.column("image").to_pylist()
                for idx, img in zip(idxs, imgs):
                    data = img[0]["bytes"] if isinstance(img, list) else img["bytes"]
                    (out_dir / tpl.format(int(idx))).write_bytes(data)
                    written += 1
    print(f"[{dataset}] arrow 导出完成: {written} 张 -> {out_dir}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(SOURCES))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all:
        for ds, (repo, kind) in SOURCES.items():
            (fetch_zip if kind == "zip" else fetch_arrow)(ds, repo)
        return 0
    if not args.dataset:
        ap.error("需 --dataset 或 --all")
    repo, kind = SOURCES[args.dataset]
    (fetch_zip if kind == "zip" else fetch_arrow)(args.dataset, repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
