#!/usr/bin/env python3
"""Build assets/<sub>/<sha256-first48>.png from raw image directories.

训练 json 引用的图片形如 assets/vrsbench/<hash>.png，其中 <hash> 为图片内容
sha256 的前 48 个十六进制字符。本脚本扫描原始数据集图片目录，对其按内容哈希
重命名（软链）为 assets 相对结构，从而满足训练脚本的 --image_folder 引用。

用法:
  python scripts/build_assets_from_raw.py \
      --raw VRSBench/images --sub vrsbench \
      --out /path/to/repo/data/assets
  # 支持多组: 可重复 --raw --sub
  python scripts/build_assets_from_raw.py \
      --raw VRSBench/images --sub vrsbench \
      --raw LEVIR-CC/images --sub levir_cc \
      --out data/assets
"""
import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_one(p):
    return hash_file(p), str(p)


def build(args):
    out_root = Path(args.out)
    pairs = list(zip(args.raw, args.sub))
    # 收集所有图片文件
    files = []
    for raw in args.raw:
        base = Path(raw).expanduser()
        if not base.exists():
            print(f"[warn] raw dir missing: {base}")
            continue
        for root, _, names in os.walk(base):
            for n in names:
                if n.lower().endswith(IMG_EXT):
                    files.append(Path(root) / n)
    print(f"total raw images: {len(files)}")

    linked = 0
    dup = 0
    missing_subs = set(args.sub)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for h, p in ex.map(_hash_one, files, chunksize=64):
            sub = None
            for raw, s in pairs:
                if str(p).startswith(str(Path(raw).expanduser())):
                    sub = s
                    break
            if sub is None:
                continue
            missing_subs.discard(sub)
            d = out_root / sub
            d.mkdir(parents=True, exist_ok=True)
            target = d / f"{h[:48]}.png"
            if target.exists():
                dup += 1
                continue
            target.symlink_to(p)
            linked += 1

    print(f"linked={linked} dup(existing)={dup}")
    for s in args.sub:
        cnt = len(list((out_root / s).glob("*.png")))
        print(f"  assets/{s}: {cnt} images")
    print("DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True,
                    help="原始图片目录（可多次）")
    ap.add_argument("--sub", action="append", required=True,
                    help="corresponding assets 子目录名（与 --raw 一一对应）")
    ap.add_argument("--out", required=True, help="assets 根目录（应含 vrsbench/ levir_cc/ 等子目录）")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if len(args.raw) != len(args.sub):
        ap.error("--raw 与 --sub 数量必须一致")
    build(args)


if __name__ == "__main__":
    main()