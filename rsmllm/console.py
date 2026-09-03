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
from rsmllm.eval_reporting import print_combined_key_metrics
from rsmllm.models import get_model, MODEL_REGISTRY
from rsmllm.router_eval import (
    QUANT_EXPERTS,
    ROUTE_PLAN,
    SRC_TO_DATASET,
    build_subtask_manifest,
    new_evaluation_output_dir,
    resolve_model,
    run_eval,
)

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
    "1": ("评测实验", "route 一键路由(4专家×任务) / single 指定专家+量化+负责任务"),
    "2": ("推理服务", "按报告配置启动 vLLM 推理"),
    "3": ("训练", "多专家 SFT / 蒸馏流程"),
    "4": ("量化", "W8A8 / GPTQ 转换与部署"),
    "5": ("数据预处理", "四个数据集清洗与清单构建"),
    "6": ("Token 剪枝", "L2/Scope/路由剪枝与评测"),
    "7": ("容错探针", "位翻转注入与完整性验证"),
    "8": ("工具", "TTFT 测量 / batch 扫描 / 模型查看"),
    "q": ("退出", ""),
}

# 实现和数字入口继续保留，只从交互菜单中暂时隐藏。
HIDDEN_MENU_KEYS = frozenset({"4", "5", "6", "7", "8"})


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


def _single_task_plan(
    expert: str, selection: str
) -> list[tuple[str, str, str | None]]:
    """Resolve a single-mode selection strictly within one expert's route."""
    available = ROUTE_PLAN[expert]
    if selection.strip().lower() == "all":
        return list(available)

    aliases: dict[str, tuple[str, str, str | None]] = {}
    for task in available:
        task_name, src_name, _task_type = task
        aliases[task_name] = task
        aliases[SRC_TO_DATASET[src_name]] = task

    requested = [
        value
        for value in selection.replace("，", " ").replace(",", " ").split()
        if value
    ]
    unknown = [value for value in requested if value not in aliases]
    if not requested or unknown:
        choices = ", ".join(task[0] for task in available)
        detail = ", ".join(unknown) if unknown else "空输入"
        raise ValueError(f"不属于 {expert} 专家的任务: {detail}; 可选: {choices}, all")

    selected = []
    for value in requested:
        task = aliases[value]
        if task not in selected:
            selected.append(task)
    return selected


def _cmd_eval() -> None:
    mode = _ask(
        "评测模式 (route=按 prompt 自动选择专家 / single=指定专家评测)",
        "route",
    )
    if mode == "route":
        quant = _ask("量化方式 (bf16 / w8a8 / gptq)", "bf16")
        limit = _ask("每任务样本上限(留空=全量, 验证用填小值)", "")
        print(f"  → Prompt Router 全量评测: 量化={quant} (逐任务核验路由后选择专家)")
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
    available_names = ", ".join(task[0] for task in ROUTE_PLAN[expert])
    while True:
        selection = _ask(
            f"{expert} 专家任务(空格/逗号分隔: {available_names}; all=该专家全部)",
            "all",
        )
        try:
            tasks = _single_task_plan(expert, selection)
            break
        except ValueError as exc:
            print(f"  ✗ {exc}")

    # 直接使用 ROUTE_PLAN 的固定清单与 task_type，禁止专家/任务错配。
    alias, profile = QUANT_EXPERTS[quant][expert]
    from rsmllm.data import prepare_eval
    completed_outputs: list[Path] = []
    for task_name, src_name, task_type in tasks:
        ds = SRC_TO_DATASET[src_name]
        prepare_eval(ds)   # 图片就绪复用; 首次: 下载评测图片 + 自动构建清单
        manifest = build_subtask_manifest(
            src_name, task_type, task_name.replace("-", "_")
        )
        print(
            f"  → 评测 {task_name} (model={alias}, profile={profile}, "
            f"manifest={manifest.name})"
        )
        output_dir = new_evaluation_output_dir(manifest, profile)
        rc = run_eval(
            model_dir,
            profile,
            manifest,
            None,
            quant,
            output_dir=output_dir,
        )
        if rc == 0:
            completed_outputs.append(output_dir)
        if rc:
            print(f"  ✗ {task_name} 评测失败 (exit={rc})")
    print_combined_key_metrics(completed_outputs)


def _cmd_serve() -> None:
    quant = _ask("量化方式 (bf16 / w8a8 / gptq)", "bf16").lower()
    if quant not in QUANT_EXPERTS:
        raise ValueError(f"不支持的量化方式: {quant}")
    mode = _ask("模式 (webui=网页界面 / cli=命令行对话)", "webui")
    # vLLM 在评测环境(evaluation/vllm_eval/.venv), 用它的 python 启动
    eval_py = Path(__file__).resolve().parent.parent / "evaluation" / "vllm_eval" / ".venv" / "bin" / "python"
    py = eval_py if eval_py.exists() else sys.executable
    cmd = [str(py), "-m", "rsmllm.router", "--serve", "--quant", quant]
    if mode == "webui":
        print("  → 启动四专家 Router + WebUI (浏览器打开 http://127.0.0.1:7860)")
        cmd.append("--webui")
    else:
        print("  → 启动四专家 Router + CLI；每条 prompt 自动选择专家")
        cmd.append("--chat")
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


def _cmd_simple(
    name: str, script: str, extra_args: list[str] | None = None
) -> None:
    print(f"  → {name}: {script}")
    path = REPO_ROOT / script
    command = [str(path), *(extra_args or [])]
    if script.endswith(".sh"):
        _run_repo(["bash", *command])
    else:
        _run_repo([TRAIN_PY, *command])


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
            if k in HIDDEN_MENU_KEYS:
                continue
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
                    model = _ask_model()
                    _cmd_simple(
                        "batch 扫描",
                        "scripts/run_batch_scan.py",
                        ["--model", model],
                    )
                else:
                    print("\n".join(f"  {k} → {v}" for k, v in MODEL_REGISTRY.items()))
            else:
                print("未知功能, 重试")
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
