#!/usr/bin/env python3
"""显式下载评测/推理模型到 models/<别名>/（或 --dir 指定位置）。

设计目的：评委/复现机在跑评测前先把模型下载好、放到约定位置，
避免首次运行才拉取（网络波动、断点难续）。

用法（在任一已建环境内执行，仓库根 scripts/ 同级）：
  python scripts/fetch_models.py --all                       # 12 个评测模型(4 专家 × bf16/w8a8/gptq)
  python scripts/fetch_models.py --all --base                # 再加 base 基座(可选)
  python scripts/fetch_models.py --quant bf16                # 只下载 bf16 一档
  python scripts/fetch_models.py expert_general expert_ground_w8a8   # 指定个别模型
  python scripts/fetch_models.py --all --dir /data/models    # 放到自定义目录
  python scripts/fetch_models.py --all --force               # 已存在也强制重下

默认目标 = <仓库>/models/（可用环境变量 RSMLLM_MODELS_ROOT 覆盖）。
下载完成后的模型目录与 router_eval / get_model 的本地优先约定一致：
评测与路由推理会直接使用 models/<别名>/，不再重复下载。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import MODELS_ROOT, MODEL_REGISTRY  # noqa: E402
from rsmllm.models import get_model  # noqa: E402

# 评测模型矩阵: 4 专家 × bf16 / w8a8 / gptq(canonical, 含一次 delta+LoRA 合并来源)
EXPERT_ALIASES = {
    "bf16": ["expert_general", "expert_ground", "expert_change", "expert_caption"],
    "w8a8": ["expert_general_w8a8", "expert_ground_w8a8",
             "expert_change_w8a8", "expert_caption_w8a8"],
    "gptq": ["expert_general_gptq", "expert_ground_gptq",
             "expert_change_gptq", "expert_caption_gptq"],
}


def _complete(directory: Path) -> bool:
    """判断目录是否为完整可加载模型(单文件或分片 safetensors 均可, 看 config.json)."""
    if not (directory / "config.json").is_file():
        return False
    if (directory / "model.safetensors").is_file():
        return True
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return False
    import json

    try:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return all((directory / name).is_file() for name in set(weight_map.values()))


def fetch(alias: str, dest_root: Path, force: bool) -> int:
    if alias not in MODEL_REGISTRY:
        print(f"  ✗ 未知别名/模型: {alias} (可用: {' '.join(MODEL_REGISTRY)})", file=sys.stderr)
        return 1
    model_id = MODEL_REGISTRY[alias]
    dest = dest_root / alias
    if not force and _complete(dest):
        print(f"  = {alias}: 已就绪 {dest} (跳过; --force 可重下)")
        return 0
    print(f"  → {alias}: 从 ModelScope 下载 {model_id} ...", flush=True)
    try:
        src = Path(get_model(alias))
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {alias} 下载失败: {exc}", file=sys.stderr)
        return 1
    if src.resolve() == dest.resolve():
        print(f"  = {alias}: 已就绪 {dest}")
        return 0
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    ok = _complete(dest)
    print(f"  ✓ {alias} -> {dest}" + ("" if ok else "  (警告: config.json 缺失)"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aliases", nargs="*", help="ModelScope 别名/模型(见 MODEL_REGISTRY)")
    ap.add_argument("--all", action="store_true", help="下载 12 个评测模型")
    ap.add_argument("--quant", choices=["bf16", "w8a8", "gptq"], help="只下载某一量化档")
    ap.add_argument("--base", action="store_true", help="额外下载 base 基座(可选)")
    ap.add_argument("--dir", type=Path, default=None, help="目标目录(默认仓库 models/)")
    ap.add_argument("--force", action="store_true", help="已存在的模型也重新下载覆盖")
    args = ap.parse_args()

    dest_root = (args.dir or MODELS_ROOT).resolve()
    selected: list[str] = []
    if args.all or args.quant:
        quants = [args.quant] if args.quant else list(EXPERT_ALIASES)
        for q in quants:
            selected += EXPERT_ALIASES[q]
    selected += args.aliases
    if args.base:
        selected.append("base")
    if not selected:
        ap.error("请指定模型别名, 或 --all / --quant bf16|w8a8|gptq")
    # 去重并保持顺序
    selected = list(dict.fromkeys(selected))
    print(f"目标目录: {dest_root}")
    rc = 0
    for alias in selected:
        rc |= fetch(alias, dest_root, args.force)
    print("\n完成。模型位置: " + str(dest_root) + "  (评测/路由自动优先使用)")
    print("说明: 只跑评测演示时下载 bf16 一档即可; 量化对比演示再加 --quant w8a8/gptq。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
