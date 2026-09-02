#!/usr/bin/env python3
"""CPU-only preflight for every README 3.5 training entry point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

CANONICAL_MODELS = {
    "Qwen3.5-4B": "config.json", "expert_general": "config.json",
    "expert_ground": "config.json", "expert_change": "config.json",
    "expert_caption": "config.json", "expert_general_lora": "adapter_config.json",
    "expert_ground_lora": "adapter_config.json", "expert_general_full": "config.json",
    "expert_ground_full": "config.json",
}
FIVE_STAGE = {
    "manifest_sft_train.json": 165395, "g_a2_mix.json": 28830,
    "a1_domainalign.json": 31871, "a2_change_mix.json": 37180,
    "expert_data_caption.jsonl": 10064,
}


def load(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list): raise ValueError(f"expected JSON array: {path}")
    return value


def check_images(rows: list[dict], root: Path) -> tuple[int, int]:
    paths = set()
    for row in rows:
        images = row.get("image", [])
        if isinstance(images, str): images = [images]
        for value in images:
            path = Path(value)
            candidate = path if path.is_absolute() else root / path
            paths.add(candidate)
    missing = sum(not path.is_file() for path in paths)
    return len(paths), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--expert-data", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve(); failures=[]; report={"models":{},"configs":{},"five_stage":{},"expert_data":{}}
    for name, marker in CANONICAL_MODELS.items():
        path=repo/"models"/name; passed=(path/marker).is_file()
        report["models"][name]={"path":str(path.resolve()) if path.exists() else str(path),"passed":passed}
        if not passed: failures.append(f"missing model {name}/{marker}")
    package=repo/"training/distillation/expert_lora"
    sys.path.insert(0,str(package/"src"))
    from expert_lora.runner import load_config
    for path in sorted((package/"configs").glob("*.json")):
        try: cfg=load_config(path); report["configs"][path.stem]={"records":cfg["expected_records"],"passed":True}
        except Exception as exc: failures.append(f"{path}: {exc}"); report["configs"][path.stem]={"passed":False}
    five_root=repo/"finetune_framework/VRSbench"
    for name, expected in FIVE_STAGE.items():
        path=five_root/name
        try:
            rows=load(path); unique,missing=check_images(rows,repo/"data")
            passed=len(rows)==expected and missing==0
            report["five_stage"][name]={"records":len(rows),"unique_images":unique,"missing_images":missing,"passed":passed}
            if not passed: failures.append(f"{name}: records={len(rows)} missing_images={missing}")
        except Exception as exc: failures.append(f"{name}: {exc}")
    if args.expert_data:
        from expert_lora.data import load_records, validate_records
        for config_name in ("general_exp7","grounding_bootstrap_942","grounding_exp5"):
            cfg=load_config(package/"configs"/f"{config_name}.json"); path=args.expert_data/cfg["data"]
            rows=load_records(path); result=validate_records(rows,schema=cfg["data_schema"],image_root=args.expert_data,check_images=True,allow_absolute=False)
            result["expected_records"]=cfg["expected_records"]
            result["passed"] = result["passed"] and len(rows)==cfg["expected_records"]
            report["expert_data"][config_name]=result
            if not result["passed"]: failures.append(f"{config_name}: {result['errors'][:3]}")
    report["passed"]=not failures; report["failures"]=failures
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
