"""模型懒加载: 运行时才从 ModelScope 拉取(首次), 本地缓存命中即复用.

用法(任何实验入口):
    from rsmllm.models import get_model
    model_dir = get_model("w8a8")          # 别名 -> 首次自动下载, 之后直接读缓存
    model_dir = get_model("/path/to/local")  # 本地路径原样返回
    model_dir = get_model("HITSZ-JBGS/xxx")  # 直接给 ModelScope id
"""
from __future__ import annotations

import os
from pathlib import Path

from rsmllm.config import MODELS_CACHE, MODEL_REGISTRY


def get_model(name: str, *, cache_dir: str | None = None) -> str:
    """解析模型引用到本地目录; 未命中缓存时按需调用 ModelScope snapshot_download."""
    p = Path(name).expanduser()
    # 本地目录优先(复现/离线场景): 仅当名字像是路径时才检查，
    # 避免把模型别名(如 "base")误认为是仓库里的同名目录。
    if p.is_absolute() or ("/" in name or "\\" in name):
        if p.exists():
            return str(p)

    model_id = MODEL_REGISTRY.get(name, name)  # 别名 or 直接 id
    cache = cache_dir or os.environ.get("RSMLLM_MODEL_CACHE") or str(MODELS_CACHE)
    if not os.environ.get("MODELSCOPE_OFFLINE"):
        try:
            from modelscope import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "需要 modelscope: uv add modelscope  (或用本地模型路径绕过)" ) from e
        path = snapshot_download(model_id, cache_dir=cache)
        return str(path)
    # 离线守卫: 无缓存时报错而不是误用
    raise FileNotFoundError(f"模型 {name!r} 本地无缓存且 MODELSCOPE_OFFLINE=1")


def register(id_or_path: str, alias: str | None = None) -> str:
    """运行时注册本地/远端模型(交互式/自定义场景)."""
    target = id_or_path if Path(id_or_path).expanduser().exists() else id_or_path
    if alias:
        MODEL_REGISTRY[alias] = target
    return get_model(id_or_path)
