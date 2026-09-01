#!/usr/bin/env python3
"""下载评测图片到 datasets/shared_datasets/<dataset>/ (与清单路径一致).

两种来源:
  默认: ModelScope 数据集仓库 Fun10165/rs-mllm-benchmark-data-clean (清洗裁剪包, ~8.8GB)
  --source hf: HuggingFace 官方源(全量复现, 数据量大: XLRS 系列 38~127GB)

用法:
  python scripts/fetch_benchmark_data.py --all                # 全部 6 数据集(默认 ModelScope)
  python scripts/fetch_benchmark_data.py --dataset vrsbench   # 单个
  python scripts/fetch_benchmark_data.py --all --source hf    # HF 官方源(全量)

产物:
  datasets/shared_datasets/<dataset>/...   图片(布局与评测清单路径一致)

依赖: modelscope (评测环境已含); --source hf 需 pyarrow+datasets(量化环境含)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "datasets" / "shared_datasets"
MS_REPO = "Fun10165/rs-mllm-benchmark-data-clean"

# dataset -> (输出目录, ModelScope tar 文件名, HF 仓库, HF 类型)
SOURCES = {
    "vrsbench": ("VRSBench", "images_VRSBench.tar", "xiang709/VRSBench", "zip"),
    "mme": ("MME-RealWorld-RS", "images_MME-RealWorld-RS.tar", "yifanzhang114/MME-RealWorld", "tar"),
    "xlrs": ("XLRS-Bench-lite", "images_XLRS-Bench-lite.tar", "initiacms/XLRS-Bench-lite", "arrow"),
    "xlrs_caption": ("XLRS-Bench_caption_en", "images_XLRS-Bench_caption_en.tar", "initiacms/XLRS-Bench_caption_en", "arrow"),
    "xlrs_grounding": ("XLRS-Bench_visual_grounding_en", "images_XLRS-Bench_visual_grounding_en.tar", "initiacms/XLRS-Bench_visual_grounding_en", "arrow"),
    "levircc": ("LEVIR-CC", "images_LEVIR-CC.tar", "lcybuaa/LEVIR-CC", "zip"),
}


def fetch_ms(dataset: str, subdir: str, tar_name: str) -> int:
    """从 ModelScope 数据集仓库下载裁剪图片包."""
    from modelscope.hub.api import HubApi
    api = HubApi()
    out = DEST / subdir
    out.mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / ".cache" / tar_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"[{dataset}] 从 ModelScope 下载 {tar_name} ...")
        api.upload_file  # noqa: B018 (确保 api 已加载)
        from modelscope.hub.file_download import dataset_file_download
        cache = Path(dataset_file_download(MS_REPO, tar_name, local_dir=str(cache.parent)))
    print(f"[{dataset}] 解压到 {out} ...")
    with tarfile.open(cache, "r") as tar:
        tar.extractall(out, filter="data")
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"[{dataset}] 就绪: {n} 文件 -> {out}")
    return n


def fetch_hf_zip(dataset: str, repo: str, subdir: str) -> int:
    """HF 官方 zip: 下载并按清单文件名解压."""
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    needed = _needed_files(dataset)
    zip_name = {"vrsbench": "Images_val.zip", "levircc": "Levir-CC-dataset.zip"}[dataset]
    local = hf_hub_download(repo, zip_name, repo_type="dataset", token=None)
    out = DEST / subdir
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(local) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            fname = Path(name).name
            if fname in needed:
                (out / fname).write_bytes(z.read(name))
                written += 1
    print(f"[{dataset}] HF zip 解压: {written}/{len(needed)} -> {out}")
    return written


def fetch_hf_arrow(dataset: str, repo: str, subdir: str) -> int:
    """HF 官方 arrow: 按 index 导出图片重编号(仅当无 ModelScope 包时)."""
    import pyarrow.ipc as ipc
    from huggingface_hub import HfApi, hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    api = HfApi()
    files = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".arrow"))
    tpl = {
        "xlrs": "xlrs_{:05d}.png",
        "xlrs_caption": "xlrs_caption_{:05d}.jpg",
        "xlrs_grounding": "xlrs_vg_{:05d}.jpg",
    }[dataset]
    out = DEST / subdir
    out.mkdir(parents=True, exist_ok=True)
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
                    (out / tpl.format(int(idx))).write_bytes(data)
                    written += 1
    print(f"[{dataset}] HF arrow 导出: {written} -> {out}")
    return written


def _needed_files(dataset: str) -> set[str]:
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
    ap.add_argument("--source", choices=["modelscope", "hf"], default="modelscope",
                    help="modelscope=清洗裁剪包(默认); hf=官方全量")
    args = ap.parse_args()
    targets = list(SOURCES) if args.all else [args.dataset]
    if not args.dataset and not args.all:
        ap.error("需 --dataset 或 --all")
    for ds in targets:
        subdir, tar_name, hf_repo, hf_kind = SOURCES[ds]
        if args.source == "modelscope":
            fetch_ms(ds, subdir, tar_name)
        else:
            (fetch_hf_zip if hf_kind == "zip" else fetch_hf_arrow)(ds, hf_repo, subdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
