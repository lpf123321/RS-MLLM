"""RS-MLLM 集中配置: 所有路径/模型注册/报告配置参数. 环境变量可覆盖: RSMLLM_*.

独立运行约束:
  - 不依赖任何共享服务器的绝对路径; 默认数据/模型/产物都在仓库内或 MODEL_CACHE.
  - 数据与模型通过 ModelScope 按需获取(见 rsmllm/models.py), 本地缓存命中即复用.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _settle(key: str, default_repo: Path, cluster_default: str | None = None) -> Path:
    """env 优先 -> 仓库本地可用 -> 集群路径(兼容正在运行的链路)"""
    v = os.environ.get(key)
    if v:
        return Path(v).expanduser().resolve()
    if default_repo.exists():
        return default_repo
    if cluster_default:
        return Path(cluster_default)
    return default_repo


def _env_path(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    return Path(v).expanduser().resolve() if v else default


# 数据根(清洗后清单默认为仓库 datasets/; 图片/其他大数据通过 RSMLLM_*_ROOT 指向)
DATA_ROOT = _settle("RSMLLM_DATA_ROOT", REPO_ROOT / "datasets", "/users/u2024311136/shared/shared_datasets")
IMAGES_ROOT = _env_path("RSMLLM_IMAGES_ROOT", DATA_ROOT / "images")
MODELS_CACHE = _env_path("RSMLLM_MODEL_CACHE", REPO_ROOT / ".models")
MODELS_ROOT = _env_path("RSMLLM_MODELS_ROOT", REPO_ROOT / "models")
# 数据集缓存(清洗后清单/训练 json): 首次从 ModelScope 自动拉取到 datasets/
DATASETS_CACHE = _env_path("RSMLLM_DATASETS_CACHE", REPO_ROOT / "datasets")
RESULTS_DIR = _env_path("RSMLLM_RESULTS_DIR", REPO_ROOT / "results")
LOGS_DIR = _env_path("RSMLLM_LOGS_DIR", REPO_ROOT / "logs")

# Manifest 命名(仓库 datasets_data/ 与共享数据集构建一致)
MANIFESTS = {
    "vrsbench": DATA_ROOT / "vrsbench_eval.jsonl",
    "mme": DATA_ROOT / "mme_rs.jsonl",
    "xlrs": DATA_ROOT / "xlrs.jsonl",
    "levircc": DATA_ROOT / "levircc_test.jsonl",
}

# 评测子集(报告附录: 770 量化对比 / 590 运行时配对 / 400 离散精度 / 950 量化部署)
SUBSETS = {
    "950": "e2_manifest",
    "770": "quant_compare",
    "590": "runtime_pair",
    "400": "discrete_direction",
    "full": "alltask_full",
}

# 报告配置参数(与《技术报告》附录"评测校验条件"保持一致)
REPORT_CONF = {
    "vllm_version": "0.26.0",
    "dtype": "bfloat16",
    "sampling": "greedy",
    "max_model_len": 16384,
    "batch_size": 64,
    "max_num_seqs": 64,
    "min_pixels": 200704,
    "max_pixels": 2097152,
    "gpu_memory_utilization": 0.85,
}

# ModelScope 模型注册表(download → 按需拉取)
MODEL_REGISTRY = {
    "base": "Fun10165/qwen-3.5-rs",
    "mmerestore_bf16": "Fun10165/rs-mllm-mmerestore-bf16",
    "w8a8": "Fun10165/rs-mllm-mmerestore-w8a8-int8",
    "gptq": "Fun10165/rs-mllm-mmerestore-w4a16-gptq",
    "expert_general": "Fun10165/rs-mllm-expert-general",
    "expert_ground": "Fun10165/rs-mllm-expert-ground",
    "expert_general_lora": "Fun10165/rs-mllm-expert-general-lora",
    "expert_ground_lora": "Fun10165/rs-mllm-expert-ground-lora",
    "expert_change": "Fun10165/rs-mllm-expert-change",
    "expert_caption": "Fun10165/rs-mllm-expert-caption",
    "expert_general_w8a8": "Fun10165/rs-mllm-expert-general-w8a8-int8",
    "expert_general_gptq": "Fun10165/rs-mllm-expert-general-w4a16-gptq",
    "expert_ground_w8a8": "Fun10165/rs-mllm-expert-ground-w8a8-int8",
    "expert_ground_gptq": "Fun10165/rs-mllm-expert-ground-w4a16-gptq",
    "expert_change_w8a8": "Fun10165/rs-mllm-expert-change-w8a8-int8",
    "expert_change_gptq": "Fun10165/rs-mllm-expert-change-w4a16-gptq",
    "expert_caption_w8a8": "Fun10165/rs-mllm-expert-caption-w8a8-int8",
    "expert_caption_gptq": "Fun10165/rs-mllm-expert-caption-w4a16-gptq",
}

# ModelScope 数据集注册表(清洗后训练清单 json): 首次自动拉取到 DATASETS_CACHE
DATA_REGISTRY = {
    # 训练数据: 14 个清洗 json(发布名), fetch 后还原内部名到 finetune_framework/VRSbench/
    "training": "yasumi/rs-mllm-datasets",
}

# 原始图片来源(HuggingFace 官方数据集): 供 build_assets_from_raw 按内容哈希还原 assets
HF_RAW = {
    "vrsbench": "lmms-lab/VRSBench",       # images/ 下按原始名存放
    "levir_cc": "chuangao/LEVIR-CC-CN",    # 备用: images train/val/test 双时相
}
