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

from rsmllm.config import DATASETS_CACHE, DATA_REGISTRY, REPO_ROOT

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


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def get_dataset(
    name: str,
    *,
    cache_dir: str | None = None,
    allow_patterns: list[str] | tuple[str, ...] | None = None,
) -> str:
    """解析数据集引用到本地目录; 未命中缓存时按需调用 ModelScope snapshot_download."""
    p = Path(name).expanduser()
    # 本地目录优先(复现/离线场景): 仅当名字像是路径(含分隔符/绝对路径)时才检查，
    # 避免把数据集别名(如 "training")误认为是仓库里的同名目录。
    if p.is_absolute():
        if p.exists():
            return str(p.resolve())
    elif "/" in name or "\\" in name:
        for candidate in (_repo_path(p), p):
            if candidate.exists():
                return str(candidate.resolve())

    dataset_id = DATA_REGISTRY.get(name, name)  # 别名 or 直接 id
    cache = Path(
        cache_dir or os.environ.get("RSMLLM_DATASETS_CACHE") or str(DATASETS_CACHE)
    ).expanduser()
    if not cache.is_absolute():
        cache = _repo_path(cache)
    if not os.environ.get("MODELSCOPE_OFFLINE"):
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "需要 modelscope: pip install modelscope  (或用本地数据集路径绕过)" ) from e
        path = snapshot_download(
            model_id=dataset_id, repo_type="dataset", local_dir=str(cache.resolve()),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
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
    root = _repo_path(dataset_root) if dataset_root else Path(get_dataset("training"))
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")

    dest_root = _repo_path(dest_root) if dest_root else REPO_ROOT / VRSBENCH_DIR
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

# dataset -> build_sample_manifest 输出的清单文件名(DATASETS 映射名)
BENCH_MANIFEST_NAME = {
    "vrsbench": "vrsbench_eval.jsonl",
    "mme": "mme_rs.jsonl",
    "xlrs": "xlrs.jsonl",
    "xlrs_caption": "xlrs_caption_en.jsonl",
    "xlrs_grounding": "xlrs_grounding_test.jsonl",
    "levircc": "levircc_test.jsonl",
}


def _resolve_manifest_image(manifest: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("manifest image path is empty")
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def validate_eval_manifest(manifest: Path) -> int:
    """Validate image metadata before a subtask can reach the evaluator."""
    manifest = Path(manifest)
    if not manifest.is_file():
        raise ValueError(f"evaluation manifest does not exist: {manifest}")

    sample_count = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in evaluation manifest {manifest}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"manifest row {line_number} is not an object: {manifest}"
                )
            images = row.get("images")
            if not isinstance(images, list) or not images:
                raise ValueError(
                    f"manifest row {line_number} has no images: {manifest}"
                )
            for image_number, image in enumerate(images, start=1):
                if not isinstance(image, dict):
                    raise ValueError(
                        f"manifest row {line_number} image {image_number} is not an object"
                    )
                width = image.get("width")
                height = image.get("height")
                if (
                    isinstance(width, bool)
                    or isinstance(height, bool)
                    or not isinstance(width, int)
                    or not isinstance(height, int)
                    or width <= 0
                    or height <= 0
                ):
                    raise ValueError(
                        f"manifest row {line_number} image {image_number} has invalid "
                        f"dimensions {width!r}x{height!r}: {manifest}"
                    )
                image_path = _resolve_manifest_image(manifest, image.get("path"))
                if not image_path.is_file():
                    raise ValueError(
                        f"manifest row {line_number} image {image_number} does not "
                        f"exist: {image_path}"
                    )
            sample_count += 1

    if sample_count == 0:
        raise ValueError(f"evaluation manifest is empty: {manifest}")
    return sample_count


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
    sp.run(cmd, check=True, cwd=str(REPO_ROOT))
    return img_dir


def prepare_eval(dataset: str, *, refresh_images: bool = False,
                 force_build: bool = False) -> Path:
    """首次评测准备: 图片下载到 datasets/shared_datasets + 构建可移植评测清单.

    流程:
      1) ensure_benchmark_data(dataset)  -> 图片就绪(本机已有则复用; 否则 ModelScope/HF 下载)
      2) 若 evaluation/vllm_eval/manifests/<dataset>.jsonl 缺失或校验失败, 调用
         scripts/build_sample_manifest.py 原子重建(图片相对路径
         ../../../datasets/shared_datasets + 真实尺寸), 供评测器/路由评测直接使用。
      3) 基础清单通过图片路径和尺寸校验后, 才允许生成子任务清单或启动评测。

    返回: 评测清单路径(evaluation/vllm_eval/manifests/<dataset>.jsonl).
    """
    from rsmllm.config import REPO_ROOT
    if dataset not in BENCH_DATASETS:
        raise ValueError(f"未知评测数据集: {dataset} (可选 {list(BENCH_DATASETS)})")

    ensure_benchmark_data(dataset, refresh=refresh_images)

    manifest = REPO_ROOT / "evaluation" / "vllm_eval" / "manifests" / BENCH_MANIFEST_NAME[dataset]
    rebuild = force_build or not manifest.exists()
    if not rebuild:
        try:
            validate_eval_manifest(manifest)
        except (OSError, ValueError) as exc:
            print(f"[data] 现有评测清单无效, 重新构建: {exc}", flush=True)
            rebuild = True
    if rebuild:
        import subprocess as sp
        script = REPO_ROOT / "scripts" / "build_sample_manifest.py"
        eval_python = REPO_ROOT / "evaluation" / "vllm_eval" / ".venv" / "bin" / "python"
        builder_python = str(eval_python) if eval_python.is_file() else sys.executable
        cmd = [builder_python, str(script), "--dataset", dataset,
               "--images-root", "../../../datasets/shared_datasets"]
        print(f"[data] 构建评测清单: {dataset}")
        sp.run(cmd, check=True, cwd=str(REPO_ROOT))
    validate_eval_manifest(manifest)
    return manifest
