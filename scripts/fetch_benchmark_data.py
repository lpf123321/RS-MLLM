#!/usr/bin/env python3
"""从 HuggingFace 官方源下载评测数据(图片+清单), 零硬编码路径.

用法:
  python scripts/fetch_benchmark_data.py --all                # 全部 6 个数据集
  python scripts/fetch_benchmark_data.py --dataset vrsbench   # 单个

产物(相对仓库根):
  datasets/assets/<dataset>/...   图片(与评测清单路径对应)
  datasets/manifests/*.jsonl      Sample schema 清单(评测器直接读)

数据集 -> HF 官方仓库:
  vrsbench       xiang709/VRSBench
  mme            yifanzhang114/MME-RealWorld
  xlrs           initiacms/XLRS-Bench-lite
  xlrs_caption   initiacms/XLRS-Bench_caption_en
  xlrs_grounding initiacms/XLRS-Bench_visual_grounding_en
  levircc        lcybuaa/LEVIR-CC

依赖: 评测环境(python -m ... 见 README) 或独立 venv(pyarrow+datasets+requests)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "datasets" / "assets"
MANIFESTS_OUT = REPO_ROOT / "datasets" / "manifests"

# dataset -> (HF repo, 类型, 输出子目录)
SOURCES = {
    "vrsbench": ("xiang709/VRSBench", "zip", "vrsbench"),
    "levircc": ("lcybuaa/LEVIR-CC", "zip", "levircc"),
    "mme": ("yifanzhang114/MME-RealWorld", "tar", "mme"),
    "xlrs": ("initiacms/XLRS-Bench-lite", "arrow", "xlrs_bench_lite"),
    "xlrs_caption": ("initiacms/XLRS-Bench_caption_en", "arrow", "xlrs_caption"),
    "xlrs_grounding": ("initiacms/XLRS-Bench_visual_grounding_en", "arrow", "xlrs_grounding"),
}

ZIP_FILES = {"vrsbench": "Images_val.zip", "levircc": "Levir-CC-dataset.zip"}


def _hf_download(repo: str, filename: str) -> str:
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", token=None)


def fetch_zip(dataset: str, repo: str, sub: str) -> int:
    """zip 型: 解压清单引用的图片文件到 assets/<sub>/."""
    needed = _needed_files(dataset)
    zip_name = ZIP_FILES[dataset]
    print(f"[{dataset}] 下载 {repo}/{zip_name} ...")
    local = _hf_download(repo, zip_name)
    out = ASSETS / sub
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(local) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            if Path(name).name in needed:
                (out / Path(name).name).write_bytes(z.read(name))
                written += 1
    print(f"[{dataset}] 图片就绪: {written}/{len(needed)} 张 -> {out}")
    return written


def fetch_arrow(dataset: str, repo: str, sub: str) -> int:
    """arrow 型: 按 index 顺序导出内嵌图片并重编号(与评测清单文件名一致)."""
    import pyarrow.ipc as ipc
    from huggingface_hub import HfApi
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".arrow"))
    tpl = {
        "xlrs": "xlrs_{:05d}.png",
        "xlrs_caption": "xlrs_caption_{:05d}.jpg",
        "xlrs_grounding": "xlrs_vg_{:05d}.jpg",
    }[dataset]
    out = ASSETS / sub
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in files:
        local = _hf_download(repo, f)
        with open(local, "rb") as fh:
            reader = ipc.open_stream(fh)
            for batch in reader:
                idxs = batch.column("index").to_pylist()
                imgs = batch.column("image").to_pylist()
                for idx, img in zip(idxs, imgs):
                    data = img[0]["bytes"] if isinstance(img, list) else img["bytes"]
                    (out / tpl.format(int(idx))).write_bytes(data)
                    written += 1
    print(f"[{dataset}] 图片就绪: {written} 张 -> {out}")
    return written


def fetch_tar(dataset: str, repo: str, sub: str) -> int:
    """tar 型(MME): 解压清单引用图片. MME 官方为多分片 tar.gz, 需按 images_resized 匹配."""
    print(f"[{dataset}] MME 官方源为 tar.gz 分片, 图片名与评测清单(images_resized)需映射; 暂用清单引用集合")
    return 0


def _needed_files(dataset: str) -> set[str]:
    """从仓库清单收集需要的图片文件名."""
    src = REPO_ROOT / "datasets_data" / {
        "vrsbench": "vrsbench_eval.jsonl",
        "levircc": "levircc_test.jsonl",
    }[dataset]
    needed: set[str] = set()
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if "messages" in d:
                for m in d["messages"]:
                    for c in m.get("content", []):
                        if c.get("type") == "image":
                            needed.add(Path(c["image"]).name)
            elif "image" in d:
                needed.add(Path(d["image"]).name)
    return needed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=list(SOURCES))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    targets = list(SOURCES) if args.all else [args.dataset]
    if not args.dataset and not args.all:
        ap.error("需 --dataset 或 --all")
    for ds in targets:
        repo, kind, sub = SOURCES[ds]
        {"zip": fetch_zip, "arrow": fetch_arrow, "tar": fetch_tar}[kind](ds, repo, sub)
    return 0


if __name__ == "__main__":
    sys.exit(main())
