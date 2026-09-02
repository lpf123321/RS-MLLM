"""RS-MLLM 交互式控制台: 菜单驱动, 一问一答, 运行时才拉取模型.

启动:  python -m rsmllm.console   (或仓库根 `rsmllm` 脚本)
     rsmllm> 1      # 评测
     ...           # 问答式收集参数 -> 自动解析模型(首次下载) -> 执行
"""
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from rsmllm.config import REPORT_CONF
from rsmllm.models import get_model, MODEL_REGISTRY
from rsmllm.router_eval import QUANT_EXPERTS, build_subtask_manifest, resolve_model

REPO_ROOT = Path(__file__).resolve().parent.parent
# 训练类分支(deepspeed/trl)依赖根环境; 评测/推理(vllm)依赖评测环境
TRAIN_PY = str(REPO_ROOT / ".venv" / "bin" / "python")
if not Path(TRAIN_PY).exists():
    TRAIN_PY = sys.executable  # uv run 或根 venv 启动时兜底
# 评测/推理类分支(vllm)依赖评测环境
EVAL_PY = str(REPO_ROOT / "evaluation" / "vllm_eval" / ".venv" / "bin" / "python")
if not Path(EVAL_PY).exists():
    EVAL_PY = sys.executable

MAIN_MENU = {
    "1": ("评测实验", "route 一键路由(4专家×任务) / single 指定专家+量化+数据集"),
    "2": ("推理服务", "按报告配置启动 vLLM 推理"),
    "3": ("训练", "多专家 SFT / 蒸馏流程"),
    "4": ("量化", "W8A8 / GPTQ 转换与部署"),
    "5": ("数据预处理", "四个数据集清洗与清单构建"),
    "6": ("Token 剪枝", "L2/Scope/路由剪枝与评测"),
    "7": ("容错探针", "位翻转注入与完整性验证"),
    "8": ("工具", "TTFT 测量 / batch 扫描 / 模型查看"),
    "q": ("退出", ""),
}


def _repo_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing if existing else ""
    )
    return env


def _run_repo(
    command: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    child_env = _repo_env()
    if env:
        child_env.update(env)
    return subprocess.run(
        command,
        check=False,
        cwd=str(REPO_ROOT),
        env=child_env,
    )


def _ask(prompt: str, default: str = "") -> str:
    r = input(f"rsmllm> {prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return r or default


def _ask_model() -> str:
    aliases = ", ".join(f"{k}" for k in MODEL_REGISTRY)
    raw = _ask(f"模型别名({aliases})或 ModelScope id 或本地路径", "mmerestore_bf16")
    print(f"  ↳ 解析模型: {raw}")
    return get_model(raw)  # 首次自动下载(缓存命中则秒回)


def _cmd_eval() -> None:
    mode = _ask(
        "评测模式 (route=自动选择已合并 delta+LoRA 并用 vLLM 跑 / single=单模型评测)",
        "route",
    )
    if mode == "route":
        quant = _ask("量化方式 (bf16 / w8a8 / gptq)", "bf16")
        limit = _ask("每任务样本上限(留空=全量, 验证用填小值)", "")
        print(f"  → 一键路由专家评测: 量化={quant} (专家/任务自动分配)")
        cmd = [EVAL_PY, "-m", "rsmllm.router_eval", "--quant", quant]
        if limit:
            cmd += ["--limit", limit]
        result = _run_repo(cmd)
        if result.returncode:
            print(f"  ✗ 一键路由评测失败 (exit={result.returncode})")
        return
    expert = _ask("专家(general=VQA/多选 / grounding=指代定位 / change=变化描述 / caption=图像描述)",
                  "general")
    if expert not in QUANT_EXPERTS["bf16"]:
        print(f"  ✗ 未知专家 {expert!r}, 使用默认 general")
        expert = "general"
    quant = _ask("量化方式 (bf16 / w8a8 / gptq)", "bf16")
    if quant not in QUANT_EXPERTS:
        print(f"  ✗ 未知量化方式 {quant!r}, 使用默认 bf16")
        quant = "bf16"
    alias, _profile = QUANT_EXPERTS[quant][expert]
    print(f"  → 模型: {alias} ({expert} × {quant}); 本地 models/ 优先, 缺失时 ModelScope 拉取")
    # 与 route 同一解析: canonical 专家(含 merge_manifest 校验), 量化版由 canonical 导出
    model_dir = resolve_model(expert, quant)
    datasets = _ask(
        "数据集(空格/逗号分隔: vrsbench mme xlrs xlrs_caption xlrs_grounding levircc; all=全部)",
        "all",
    )
    if datasets.strip().lower() == "all":
        ds_list = ["vrsbench", "mme", "xlrs", "xlrs_caption", "xlrs_grounding", "levircc"]
    else:
        ds_list = [d for d in datasets.replace("，", " ").replace(",", " ").split() if d]
    if not ds_list:
        print("  ✗ 未识别到数据集, 使用默认 vrsbench")
        ds_list = ["vrsbench"]
    subtask = _ask("子任务(留空=全量 / vqa / caption / referring / mcq / change; 仅对含该任务类型的数据集生效)", "")
    # vLLM 评测器(与 route 同引擎同环境): evaluation/vllm_eval/vision_opd_vllm_eval.py
    # 数据集 -> 评测清单(由 prepare_eval 自动构建), 子任务按 task_type 切分
    manifest_src = {
        "vrsbench": "vrsbench_eval.jsonl",
        "mme": "mme_rs.jsonl",
        "xlrs": "xlrs.jsonl",
        "xlrs_caption": "xlrs_caption_en.jsonl",
        "xlrs_grounding": "xlrs_grounding_test.jsonl",
        "levircc": "levircc_test.jsonl",
    }
    subtask_task = {
        "vqa": "open_vqa", "caption": "caption", "referring": "bbox",
        "mcq": "single_choice", "change": "change_caption",
    }
    alias, profile = QUANT_EXPERTS[quant][expert]
    from rsmllm.data import prepare_eval
    eval_dir = Path(__file__).resolve().parent.parent / "evaluation" / "vllm_eval"
    for ds in ds_list:
        prepare_eval(ds)   # 图片就绪复用; 首次: 下载评测图片 + 自动构建清单
        manifest = eval_dir / "manifests" / manifest_src[ds]
        if subtask:
            task_type = subtask_task.get(subtask)
            if not task_type:
                print(f"  ✗ 未知子任务 {subtask!r} (可选 vqa/caption/referring/mcq/change), 跳过 {ds}")
                continue
            label = f"{ds}_{subtask}"
            manifest = build_subtask_manifest(manifest.name, task_type, label)
            if manifest.stat().st_size == 0:
                print(f"  [warn] {ds} 无 {subtask} 子任务样本, 跳过")
                continue
        cmd = [EVAL_PY, str(eval_dir / "vision_opd_vllm_eval.py"),
               "--manifest", str(manifest),
               "--model", model_dir,
               "--model-profile", profile,
               "--quantization", quant,
               "--min-pixels", str(REPORT_CONF["min_pixels"]),
               "--max-pixels", str(REPORT_CONF["max_pixels"]),
               "--batch-size", str(REPORT_CONF["batch_size"]),
               "--max-model-len", str(REPORT_CONF["max_model_len"]),
               "--max-num-seqs", str(REPORT_CONF["max_num_seqs"]),
               "--gpu-memory-utilization", str(REPORT_CONF["gpu_memory_utilization"]),
               "--enforce-eager"]
        print(f"  → 评测 {ds} (model={alias}, profile={profile}, manifest={manifest.name}, subtask={subtask or '全量'})")
        _run_repo(cmd)


def _cmd_serve() -> None:
    model = _ask_model()
    mode = _ask("模式 (webui=网页界面 / cli=命令行对话)", "webui")
    model_dir = get_model(model)  # 标准 ModelScope 用法: 缓存复用, 返回真实路径
    # vLLM 在评测环境(evaluation/vllm_eval/.venv), 用它的 python 启动
    eval_py = Path(__file__).resolve().parent.parent / "evaluation" / "vllm_eval" / ".venv" / "bin" / "python"
    py = eval_py if eval_py.exists() else sys.executable
    if mode == "webui":
        print(f"  → WebUI 推理 model={model_dir} (浏览器打开 http://127.0.0.1:7860)")
        cmd = [str(py), "-m", "rsmllm.webui", "--model", model_dir]
    else:
        port = _ask("端口", "8001")
        print(f"  → CLI 推理 model={model_dir} port={port}")
        cmd = [str(py), "-m", "rsmllm.serve",
               "--model", model_dir, "--port", port,
               "--gpu-mem", str(REPORT_CONF["gpu_memory_utilization"])]
    _run_repo(cmd)


def _cmd_quantize() -> None:
    method = _ask("方法 (w8a8-int8 / w4a16-gptq)", "w8a8-int8")
    model = _ask_model()
    out = _ask("输出目录", "quantized_models/out")
    cmd = [TRAIN_PY, str(REPO_ROOT / "rsmllm" / "quantize.py"), "--method", method,
           "--model", model, "--output", out]
    _run_repo(cmd)


def _cmd_data() -> None:
    action = _ask("操作 (download=下载评测图片 / preprocess=数据清洗)", "download")
    if action == "download":
        ds = _ask("数据集(all / vrsbench / mme / xlrs / xlrs_caption / xlrs_grounding / levircc)", "all")
        print("  → 下载评测图片(ModelScope 清洗包, ~8.8GB; --source hf 走官方源)")
        cmd = [EVAL_PY, str(REPO_ROOT / "scripts" / "fetch_benchmark_data.py")]
        if ds != "all":
            cmd += ["--dataset", ds]
        _run_repo(cmd)
    else:
        ds = _ask("数据集 (vrsbench/mme/xlrs/levircc)", "vrsbench")
        cmd = [TRAIN_PY, str(REPO_ROOT / "scripts" / f"preprocess_{ds}.py")]
        _run_repo(cmd)


def _cmd_simple(name: str, script: str) -> None:
    print(f"  → {name}: {script}")
    path = REPO_ROOT / script
    if script.endswith(".sh"):
        _run_repo(["bash", str(path)])
    else:
        _run_repo([TRAIN_PY, str(path)])


def _cmd_simple_eval(name: str, script: str) -> None:
    """评测环境解释器执行(vllm 依赖)."""
    print(f"  → {name}: {script}")
    _run_repo([EVAL_PY, str(REPO_ROOT / script)])


def main() -> int:
    print("=" * 60)
    print("RS-MLLM 交互控制台  (报告功能 ←→ 可执行入口)")
    print("模型首次使用会从 ModelScope 拉取; 所有参数均可直接回车取默认值")
    print("=" * 60)
    while True:
        print()
        for k, (title, desc) in MAIN_MENU.items():
            print(f"  [{k}] {title}  — {desc}" if desc else f"  [{k}] {title}")
        choice = _ask("选择功能").lower()
        if choice in ("q", "quit", "exit"):
            print("bye")
            return 0
        try:
            if choice == "1":
                _cmd_eval()
            elif choice == "2":
                _cmd_serve()
            elif choice == "3":
                stage = _ask("训练阶段(dry 预览 / all / stage1_clean / ga2_general / a1_grounding / a2b_change / caption)", "dry")
                print("  → 训练流程: train.sh", stage)
                args = ["bash", str(REPO_ROOT / "scripts/train.sh")]
                if stage == "dry":
                    args += ["stage1_clean", "--dry-run"]  # dry 预览默认 stage1_clean
                else:
                    args.append(stage)
                _run_repo(args)
            elif choice == "4":
                _cmd_quantize()
            elif choice == "5":
                _cmd_data()
            elif choice == "6":
                _cmd_simple("Token 剪枝", "evaluation/run_prune_sweep.py")
            elif choice == "7":
                _cmd_simple_eval("容错探针", "scripts/quant_tol_probe.py")
            elif choice == "8":
                sub = _ask("工具子项: ttft=batch=模型查看").lower()
                if sub == "ttft":
                    _cmd_simple("TTFT 测量", "scripts/ttft_serve_probe.py")
                elif sub == "batch":
                    _cmd_simple("batch 扫描", "scripts/run_batch_scan.py")
                else:
                    print("\n".join(f"  {k} → {v}" for k, v in MODEL_REGISTRY.items()))
            else:
                print("未知功能, 重试")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
