"""量化容错敏感性比较探针：对 bf16 / W8A8-INT8 / W4A16-GPTQ 三个模型表示，
注入固定种子的权重位翻转，测量 (A) 完整性哈希检测率 (B) 绕过校验加载后的输出漂移。

独立运行:
    python scripts/quant_tol_probe.py \
        --model bf16 --model-dir /path/to/model_mmerestore \
        --model w8a8 --model-dir /path/to/model_mmerestore_w8a8_int8 \
        --model gptq --model-dir /path/to/model_mmerestore_w4a16_gptq \
        --out-dir results/quant_tol_probe
"""
import argparse
import typing  # noqa: F401
import torch  # noqa: F401
import os
import json
import random
import shutil
import hashlib
import numpy as np
from pathlib import Path

from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()

PROMPTS = [
    "请用一句话描述这幅遥感图像。",
    "图中主要地物是什么？",
    "描述图像左上角的区域。",
    "这是一幅什么类型的遥感影像？",
    "图中是否有水体？",
    "估算图像中的建筑物数量。",
    "图像拍摄于什么季节？",
    "描述图中的植被分布。",
    "图中主干道走向如何？",
    "指出图中颜色最深的区域。",
] * 2


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flip_weights(model_dir: str, out_dir: str, n: int, seed: int, dtype: str) -> int:
    """复制权重目录，对主 safetensors 的权重张量随机翻转 n 个位（按表示）。"""
    os.makedirs(out_dir, exist_ok=True)
    keep = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt",
            "preprocessor_config.json", "chat_template.jinja", "*.provenance")
    for f in os.listdir(model_dir):
        src = os.path.join(model_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, f))
    import safetensors
    from safetensors.torch import load_file, save_file
    weight_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors") and "index" not in f]
    if not weight_files:
        return 0
    rng = random.Random(seed)
    # 遍历全部分片(不能只改 shard1, 否则副本缺权重)
    for wf in weight_files:
        t = load_file(os.path.join(model_dir, wf))
        # 选一个大的权重张量作为注入目标
        key = max(t, key=lambda k: t[k].numel())
        arr = t[key].cpu()
        flat_n = arr.numel()
        if dtype == "bf16" or arr.dtype == torch.bfloat16:
            a16 = arr.view(torch.uint16)
            fl = a16.flatten()
            for _ in range(n):
                idx = rng.randrange(flat_n)
                bit = rng.randrange(16)
                fl[idx] = fl[idx] ^ (1 << bit)
            t[key] = a16.view(torch.bfloat16)
        else:
            a8 = arr.contiguous().view(torch.uint8).flatten()
            for _ in range(n):
                idx = rng.randrange(a8.numel())
                bit = rng.randrange(8)
                a8[idx] = a8[idx] ^ (1 << bit)
            t[key] = a8.view(arr.dtype)
        save_file(t, os.path.join(out_dir, wf))
    return n


def run(model_dir: str, prompts: list[str], sp: SamplingParams) -> dict:
    # 参数与评测框架(vision_opd_vllm_eval)一致: trust_remote_code 必需(远程 code),
    # dtype 显式 bfloat16, max_model_len 16384
    llm = LLM(model=model_dir, dtype="bfloat16", trust_remote_code=True,
              limit_mm_per_prompt={"image": 2},
              mm_processor_kwargs={"min_pixels": 200704, "max_pixels": 2097152},
              gpu_memory_utilization=0.30,
              max_num_seqs=8, max_model_len=16384, enable_lora=False)
    outs = llm.generate(prompts, sp)
    try:
        llm.shutdown()  # vLLM 0.26: 显式关闭引擎, 释放显存(仅删引用不够, EngineCore 子进程不释放)
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return [o.outputs[0].text.strip() for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # 每个模型: --model <名字> --model-dir <路径>（可多组）
    ap.add_argument("--model", action="append", default=[], help="模型名(bf16/w8a8/gptq/任意)")
    ap.add_argument("--model-dir", action="append", default=[], help="模型目录，与 --model 一一对应")
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "quant_tol_probe"),
        help="输出目录(相对路径按仓库根解析)",
    )
    ap.add_argument("--n-flips", type=int, default=20, help="每个权重张量注入的翻转位个数")
    ap.add_argument("--seed", type=int, default=20260826, help="随机种子(可复现)")
    ap.add_argument("--max-tokens", type=int, default=40, help="生成最大 token 数(短答)")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)

    if len(args.model) != len(args.model_dir):
        ap.error("--model 与 --model-dir 数量必须一致(如 --model bf16 --model-dir /path/to/model)")
    if not args.model:
        ap.error("至少指定一组 --model NAME --model-dir DIR")
    MODELS = {name: _repo_path(path) for name, path in zip(args.model, args.model_dir)}
    SEED = args.seed
    N_FLIPS = args.n_flips
    OUT = _repo_path(args.out_dir)

    OUT.mkdir(parents=True, exist_ok=True)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    result = {}
    for name, mdir in MODELS.items():
        print(f"=== {name} ===", flush=True)
        # 干净基线
        base = run(str(mdir), PROMPTS, sp)
        h0 = sha256_file(os.path.join(mdir, [f for f in os.listdir(mdir)
                                             if f.endswith(".safetensors") and "index" not in f][0]))
        # 注入 + 翻转副本
        dst = OUT / f"{name}_flipped"
        n = flip_weights(str(mdir), str(dst), N_FLIPS, SEED, name)
        h1 = sha256_file(os.path.join(dst, [f for f in os.listdir(dst)
                                            if f.endswith(".safetensors") and "index" not in f][0]))
        detected = 0 if h0 == h1 else 1  # 哈希不一致 = 完整性检查应捕获
        # 绕过校验（探针）：加载翻转副本（量化模型若加载失败仅记录, 不中断实验)
        try:
            flipped = run(str(dst), PROMPTS, sp)
            changed = sum(1 for a, b in zip(base, flipped) if a != b)
            sample = flipped[0][:40]
            flip_load = "ok"
        except Exception as e:
            changed, sample, flip_load = -1, "", f"load_failed: {type(e).__name__}"
        result[name] = {
            "injected": n,
            "hash_detected": detected,
            "output_changed": changed,
            "flip_load": flip_load,
            "n_prompts": len(PROMPTS),
            "sample": sample,
        }
        print(json.dumps(result[name], ensure_ascii=False), flush=True)
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", OUT / "summary.json")


if __name__ == "__main__":
    main()
