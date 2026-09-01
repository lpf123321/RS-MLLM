"""数据集懒加载: 运行时才从 ModelScope 拉取(首次), 本地缓存命中即复用.

与 rsmllm/models.py 对称。数据集(清洗后训练 json)统一缓存在 DATASETS_CACHE
(默认 <仓库>/datasets), 首次按别名从 ModelScope 下载并还原内部文件名。

用法(任何实验入口):
    from rsmllm.data import get_dataset
    d = get_dataset("training")          # -> datasets/ 下数据集根目录
    from rsmllm.data import restore_training_data
    restore_training_data()               # 还原内部名到 finetune_framework/VRSbench/
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rsmllm.config import DATASETS_CACHE, DATA_REGISTRY

# 训练数据发布名 -> 内部名映射(与 scripts/training_data_manifest.sh 保持一致)
TRAIN_MAP = {
    # 顶层
    "stage1_unified_sft_train.json": "manifest_sft_train.json",
    "stage1_combined_train.json": "combined_train.json",
    "grounding_domain_align_train.json": "a1_domainalign.json",
    "change_anti_forget_train.json": "a2_change_mix.json",
    "general_anti_forget_train.json": "g_a2_mix.json",
    "caption_dual_domain_train.jsonl": "expert_data_caption.jsonl",
    # expert_data/
    "expert_data": {
        "general_understanding_train.json": "general_understanding.json",
        "referring_grounding_train.json": "grounding.json",
        "change_detection_train.json": "change.json",
        "stats.json": "stats.json",
    },
    # expert_data_v2/
    "expert_data_v2": {
        "general_understanding_train.json": "general_understanding.json",
        "referring_grounding_train.json": "grounding.json",
        "change_detection_train.json": "change.json",
        "stats.json": "stats.json",
    },
}

# 训练 json 还原目标目录(相对仓库根)
VRSBENCH_DIR = "finetune_framework/VRSbench"


def get_dataset(name: str, *, cache_dir: str | None = None) -> str:
    """解析数据集引用到本地目录; 未命中缓存时按需调用 ModelScope snapshot_download."""
    p = Path(name).expanduser()
    # 本地目录优先(复现/离线场景): 仅当名字像是路径(含分隔符/绝对路径)时才检查，
    # 避免把数据集别名(如 "training")误认为是仓库里的同名目录。
    if p.is_absolute() or ("/" in name or "\\" in name):
        if p.exists():
            return str(p)

    dataset_id = DATA_REGISTRY.get(name, name)  # 别名 or 直接 id
    cache = cache_dir or os.environ.get("RSMLLM_DATASETS_CACHE") or str(DATASETS_CACHE)
    if not os.environ.get("MODELSCOPE_OFFLINE"):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "需要 modelscope: pip install modelscope  (或用本地数据集路径绕过)" ) from e
        path = snapshot_download(
            model_id=dataset_id, repo_type="dataset", local_dir=cache,
            token=os.environ.get("MODELSCOPE_API_TOKEN") or None,
        )
        return str(path)
    raise FileNotFoundError(f"数据集 {name!r} 本地无缓存且 MODELSCOPE_OFFLINE=1")


def _find_src(root: Path, pub: str, nested_sub: str | None = None):
    """在下载根中定位发布名文件(支持根目录与旧嵌套结构)."""
    cands = [root / pub]
    if nested_sub:
        cands.insert(0, root / nested_sub / pub)
        cands.append(Path(str(root) + "/finetune_framework/VRSbench") / nested_sub / pub)
    for c in cands:
        if c.is_file():
            return c
    return None


def restore_training_data(dataset_root: str | None = None, *, dest_root: str | None = None) -> Path:
    """还原训练 json 发布名 -> 内部名, 放置到 <仓库>/finetune_framework/VRSbench/.

    dataset_root: 已下载的数据集目录(一般来自 get_dataset('training')); 缺省自动拉取.
    返回: 训练数据目标目录.
    """
    from rsmllm.config import REPO_ROOT
    root = Path(dataset_root) if dataset_root else Path(get_dataset("training"))
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    dest_root = Path(dest_root) if dest_root else REPO_ROOT / VRSBENCH_DIR
    dest_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    # 顶层
    top = {k: v for k, v in TRAIN_MAP.items() if isinstance(v, str)}
    for pub, int_name in top.items():
        src = _find_src(root, pub)
        if src:
            import shutil
            shutil.copyfile(src, dest_root / int_name)
            copied += 1
    # 子目录
    for sub, m in TRAIN_MAP.items():
        if not isinstance(m, dict):
            continue
        (dest_root / sub).mkdir(parents=True, exist_ok=True)
        for pub, int_name in m.items():
            src = _find_src(root, pub, nested_sub=sub)
            if src:
                import shutil
                shutil.copyfile(src, dest_root / sub / int_name)
                copied += 1
    return dest_root


# ============================================================
# 评测数据懒加载: 保证数据集图片就绪 + 返回可移植(相对路径)评测清单
# ============================================================

# 评测数据集 -> (ModelScope/HF 下载子目录, 仓库内评测清单)
BENCH_DATASETS = {
    "vrsbench": ("VRSBench", "datasets_data/vrsbench_eval.jsonl"),
    "mme": ("MME-RealWorld-RS", "datasets_data/mme_rs.jsonl"),
    "xlrs": ("XLRS-Bench-lite", "datasets_data/xlrs.jsonl"),
    "xlrs_caption": ("XLRS-Bench_caption_en", "datasets_data/xlrs_caption_en.jsonl"),
    "xlrs_grounding": ("XLRS-Bench_visual_grounding_en",
                       "datasets_data/xlrs_grounding_test.jsonl"),
    "levircc": ("LEVIR-CC", "datasets_data/levircc_test.jsonl"),
}

_OLD_PREFIX = "/users/u2024311136/shared/shared_datasets/"


def ensure_benchmark_data(dataset: str, *, refresh: bool = False) -> Path:
    """确保评测数据集图片就绪, 返回<仓库>/datasets/shared_datasets/<X>目录.

    首次会调用 scripts/fetch_benchmark_data.py 从 ModelScope 或 HF 下载图片。
    """
    from rsmllm.config import REPO_ROOT
    if dataset not in BENCH_DATASETS:
        raise ValueError(f"未知评测数据集: {dataset} (可选 {list(BENCH_DATASETS)})")
    subdir, _ = BENCH_DATASETS[dataset]
    img_dir = REPO_ROOT / "datasets" / "shared_datasets" / subdir
    # 已有图片则视为就绪(至少 1 个 .png/.jpg)
    if not refresh and any(t in {".png", ".jpg", ".jpeg", ".webp"}
                           for t in {p.suffix for p in img_dir.rglob("*") if p.is_file()}):
        return img_dir
    # 触发下载
    import subprocess as sp
    script = REPO_ROOT / "scripts" / "fetch_benchmark_data.py"
    cmd = [sys.executable, str(script), "--dataset", dataset]
    print(f"[data] 下载评测图片: {dataset} -> {img_dir}")
    sp.run(cmd, check=True)
    return img_dir


def get_eval_manifest(dataset: str) -> Path:
    """返回可移植评测清单: 图片绝对路径重映射为相对 <datasets>/shared_datasets/<X>.

    输出写入 <仓库>/.tmp_manifests/<dataset>.jsonl, 路径均相对该清单所在目录，
    评测器(Sample.from_dict manifest_dir 逻辑)可正确解析。
    """
    from rsmllm.config import REPO_ROOT
    if dataset not in BENCH_DATASETS:
        raise ValueError(f"未知评测数据集: {dataset} (可选 {list(BENCH_DATASETS)})")
    subdir, rel_manifest = BENCH_DATASETS[dataset]
    src_manifest = REPO_ROOT / rel_manifest
    if not src_manifest.exists():
        raise FileNotFoundError(f"评测清单缺失: {src_manifest}")

    out_dir = REPO_ROOT / ".tmp_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.jsonl"

    # 写入 lead map: 把以旧共享前缀开头的图路径改相对
    old_lead = f"shared_datasets/{subdir}"
    written = 0
    with src_manifest.open(encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as gh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            imgs = []
            if "messages" in d:
                for m in d["messages"]:
                    for c in m.get("content", []):
                        if c.get("type") == "image" and c.get("image"):
                            imgs.append(c["image"])
            elif "image" in d:
                imgs.append(d["image"])
            for full in imgs:
                if full.startswith(_OLD_PREFIX):
                    new = full[len(_OLD_PREFIX):]
                    # 相对 manifest_dir: .tmp_manifests 与 datasets/shared_datasets 平级
                    if "messages" in d:
                        for m in d["messages"]:
                            for c in m.get("content", []):
                                if c.get("image") == full:
                                    c["image"] = str(Path("../datasets/shared_datasets") / new)
                    elif d.get("image") == full:
                        d["image"] = str(Path("../datasets/shared_datasets") / new)
            gh.write(json.dumps(d, ensure_ascii=False) + "\n")
            written += 1
    print(f"[data] 评测清单(相对路径) -> {out_path} ({written} 行)")
    return out_path