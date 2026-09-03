#!/usr/bin/env python3
"""显式下载评测/推理模型到交互控制台使用的 ModelScope 缓存。

设计目的：评委/复现机在跑评测前先把模型下载好、放到约定位置，
避免首次运行才拉取（网络波动、断点难续）。

用法（在任一已建环境内执行，仓库根 scripts/ 同级）：
  python scripts/fetch_models.py --all                       # 12 个评测模型(4 专家 × bf16/w8a8/gptq)
  python scripts/fetch_models.py --all --base                # 再加 base 基座(可选)
  python scripts/fetch_models.py --quant bf16                # 只下载 bf16 一档
  python scripts/fetch_models.py expert_general expert_ground_w8a8   # 指定个别模型
  python scripts/fetch_models.py --all --dir /data/models    # 自定义缓存根目录

默认缓存根目录 = <仓库>/.models/（RSMLLM_MODEL_CACHE 可覆盖），与
``rsmllm.sh`` 首次使用模型时的下载位置完全相同。模型保留 ModelScope 的仓库
目录结构；下载脚本不再复制一份到 ``models/<别名>/``。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsmllm.config import MODELS_CACHE, MODEL_REGISTRY  # noqa: E402
from rsmllm.models import get_model  # noqa: E402

# 评测模型矩阵: 4 专家 × bf16 / w8a8 / gptq(canonical, 含一次 delta+LoRA 合并来源)
EXPERT_ALIASES = {
    "bf16": ["expert_general", "expert_ground", "expert_change", "expert_caption"],
    "w8a8": ["expert_general_w8a8", "expert_ground_w8a8",
             "expert_change_w8a8", "expert_caption_w8a8"],
    "gptq": ["expert_general_gptq", "expert_ground_gptq",
             "expert_change_gptq", "expert_caption_gptq"],
}


def _complete(directory: Path, alias: str) -> bool:
    """判断目录是否完整可加载(与 get_model 命中规则一致).

    完整模型: config.json + 权重(单文件, 或 index.json 全部分片);
    LoRA adapter(_lora 别名): adapter_config.json + adapter_model.{safetensors,bin}.
    """
    marker = "adapter_config.json" if alias.endswith("_lora") else "config.json"
    if not (directory / marker).is_file():
        return False
    if (directory / "adapter_model.safetensors").is_file():
        return True
    if (directory / "adapter_model.bin").is_file():
        return True
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


def fetch(alias: str, cache_root: Path, force: bool) -> int:
    if alias not in MODEL_REGISTRY:
        print(f"  ✗ 未知别名/模型: {alias} (可用: {' '.join(MODEL_REGISTRY)})", file=sys.stderr)
        return 1
    model_id = MODEL_REGISTRY[alias]
    print(f"  → {alias}: 检查/下载 ModelScope 模型 {model_id} ...", flush=True)
    try:
        src = Path(get_model(alias, cache_dir=str(cache_root)))
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ {alias} 下载失败: {exc}", file=sys.stderr)
        return 1
    ok = _complete(src, alias)
    print(f"  ✓ {alias} -> {src}" + ("" if ok else "  (警告: 模型文件不完整)"))
    if force:
        print("    说明: --force 保留兼容；ModelScope 会校验缓存并补齐缺失文件。")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aliases", nargs="*", help="ModelScope 别名/模型(见 MODEL_REGISTRY)")
    ap.add_argument("--all", action="store_true", help="下载 12 个评测模型")
    ap.add_argument("--quant", choices=["bf16", "w8a8", "gptq"], help="只下载某一量化档")
    ap.add_argument("--base", action="store_true", help="额外下载 base 基座(可选)")
    ap.add_argument("--dir", type=Path, default=None, help="缓存根目录(默认仓库 .models/)")
    ap.add_argument("--force", action="store_true", help="兼容参数；校验并补齐现有缓存")
    args = ap.parse_args()

    cache_root = (args.dir or MODELS_CACHE).resolve()
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
    print(f"缓存根目录: {cache_root}")
    rc = 0
    for alias in selected:
        rc |= fetch(alias, cache_root, args.force)
    print("\n完成。缓存根目录: " + str(cache_root) + "  (评测/路由直接复用)")
    print("说明: 只跑评测演示时下载 bf16 一档即可; 量化对比演示再加 --quant w8a8/gptq。")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
