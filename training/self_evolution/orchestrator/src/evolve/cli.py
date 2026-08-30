from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .baseline import snapshot_repositories
from .config import load_config, workspace
from .data import prepare
from .datasets import build_opd_parquet, build_sam_dataset
from .io import read_jsonl, write_json
from .metrics import paired_trajectory_comparison, sam_trace_metrics, trajectory_metrics
from .negatives import assert_no_negative_manifest_leak, ensure_verified_negatives
from .promotion import evaluate_candidate
from .report import generate_report
from .system import evaluate_sam_selectivity, train_mllm, train_sam
from .trajectory import generate_direct_answers, generate_trajectories, generate_trajectories_parallel


def _state_path(config: dict[str, Any], stage: str) -> Path:
    return workspace(config) / "runs" / stage / "state.json"


def load_or_initialize_state(config: dict[str, Any], stage: str) -> dict[str, Any]:
    path = _state_path(config, stage)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "stage": stage, "next_round": 0, "stopped": False,
        "current_sam": config["models"]["sam_t0"],
        "current_mllm": config["models"]["mllm_t0"],
        "accepted_rounds": [],
    }
    write_json(path, state)
    return state


def _sam_summary(metrics: dict[str, Any]) -> dict[str, float]:
    positive = metrics["positive"]
    return {
        "pseudo_ap50": float(positive.get("bbox_ap50", positive.get("AP50", 0.0))),
        "pseudo_recall_at_0_5": float(positive.get(
            "recall_at_0_5", positive.get("top1_recall_at_0_5", positive.get("recall_at_1_iou_0.50", 0.0))
        )),
        "pseudo_top1_iou_mean": float(positive.get("top1_iou_mean", positive.get("top1_mean_iou", 0.0))),
        "negative_fpr_at_0_5": float(metrics["negative"]["negative_fpr_at_0_5"]),
        "formal_promotion_metric": False,
    }


def run_round(config: dict[str, Any], stage: str, round_id: int) -> dict[str, Any]:
    root = workspace(config)
    state = load_or_initialize_state(config, stage)
    if state["stopped"]:
        raise RuntimeError(f"The {stage} loop is already stopped")
    if round_id != int(state["next_round"]):
        raise ValueError(f"Expected round {state['next_round']}, got {round_id}")
    round_root = root / "runs" / stage / f"round_{round_id:02d}"
    round_root.mkdir(parents=True, exist_ok=True)
    current_sam, current_mllm = state["current_sam"], state["current_mllm"]

    workers = int(config.get("trajectory_workers", 8))
    baseline_train = generate_trajectories_parallel(config, "train", stage, round_id, current_sam, current_mllm, workers)
    baseline_valid = generate_trajectories_parallel(config, "valid", stage, round_id, current_sam, current_mllm, workers)
    train_sam_data = build_sam_dataset(config, baseline_train, stage, round_id)
    valid_sam_data = build_sam_dataset(config, baseline_valid, stage, round_id)
    if not train_sam_data["samples"] or not valid_sam_data["samples"]:
        raise ValueError(
            f"No usable branch-3 pseudo targets: train={train_sam_data['samples']} valid={valid_sam_data['samples']}"
        )
    train_sam_data = ensure_verified_negatives(config, train_sam_data, "train", device=0)
    valid_sam_data = ensure_verified_negatives(config, valid_sam_data, "valid", device=0)
    assert_no_negative_manifest_leak(
        train_sam_data["negative_manifest"], valid_sam_data["negative_manifest"]
    )
    opd_data = build_opd_parquet(config, baseline_train, stage, round_id)

    sam_gpu_count = 1 if stage == "smoke" else int(config["sam_training"].get("gpus", 4))
    devices = list(range(sam_gpu_count))
    sam_candidate = train_sam(
        config, stage, round_id, current_sam, train_sam_data["data_root"],
        train_sam_data["negative_manifest"], valid_sam_data["negative_manifest"], devices,
    )
    mllm_candidate = train_mllm(config, stage, round_id, current_mllm, opd_data["output"])

    paths = {"baseline": baseline_valid}
    combinations = {
        "sam_only": (sam_candidate["merged_checkpoint"], current_mllm),
        "mllm_only": (current_sam, mllm_candidate["model"]),
        "both": (sam_candidate["merged_checkpoint"], mllm_candidate["model"]),
    }
    for tag, (sam_path, mllm_path) in combinations.items():
        paths[tag] = generate_trajectories_parallel(
            config, "valid", stage, round_id, sam_path, mllm_path, workers, tag=tag,
        )

    direct_base = generate_direct_answers(config, "valid", stage, round_id, current_mllm, "baseline")
    direct_candidate = generate_direct_answers(config, "valid", stage, round_id, mllm_candidate["model"], "candidate")
    direct_metrics = {
        "baseline": trajectory_metrics(direct_base, int(config["seed"])),
        "candidate": trajectory_metrics(direct_candidate, int(config["seed"])),
    }

    sam_eval_base = evaluate_sam_selectivity(
        config, stage, round_id, current_sam, train_sam_data["data_root"],
        valid_sam_data["negative_manifest"], "baseline", device=0,
    )
    sam_eval_candidate = evaluate_sam_selectivity(
        config, stage, round_id, sam_candidate["merged_checkpoint"], train_sam_data["data_root"],
        valid_sam_data["negative_manifest"], "candidate", device=0,
    )
    sam_summaries = {"baseline": _sam_summary(sam_eval_base), "candidate": _sam_summary(sam_eval_candidate)}

    metrics = {}
    for tag, path in paths.items():
        metrics[tag] = {
            "cvsearch": trajectory_metrics(path, int(config["seed"])),
            "sam_trace": sam_trace_metrics(path),
        }
        if tag != "baseline":
            metrics[tag]["paired_vs_baseline"] = paired_trajectory_comparison(
                paths["baseline"], path, int(config["seed"]),
            )
    metrics["direct"] = direct_metrics
    metrics["sam_selectivity"] = sam_summaries
    write_json(round_root / "metrics.json", metrics)

    baseline_metric = metrics["baseline"]["cvsearch"]
    decisions = {}
    for tag in ("sam_only", "mllm_only", "both"):
        uses_new_sam = tag in {"sam_only", "both"}
        uses_new_mllm = tag in {"mllm_only", "both"}
        sam_metric = sam_summaries["candidate" if uses_new_sam else "baseline"]
        direct_drop = (
            direct_metrics["baseline"]["accuracy"] - direct_metrics["candidate"]["accuracy"]
            if uses_new_mllm else 0.0
        )
        decision = evaluate_candidate(
            baseline_metric, metrics[tag]["cvsearch"], sam_summaries["baseline"], sam_metric,
            config["promotion"], regression_drop=direct_drop,
        )
        decisions[tag] = decision

    if stage == "smoke":
        selected = max(
            combinations,
            key=lambda tag: (
                metrics[tag]["cvsearch"]["accuracy"], -metrics[tag]["cvsearch"]["mean_pop"],
                -sam_summaries["candidate" if tag in {"sam_only", "both"} else "baseline"]["negative_fpr_at_0_5"],
            ),
        )
        decisions["smoke_override"] = "Quality gates are recorded but not enforced during the 80-sample plumbing smoke test."
        accepted = True
    else:
        eligible = [tag for tag in combinations if decisions[tag]["accepted"]]
        selected = max(
            eligible,
            key=lambda tag: (
                metrics[tag]["cvsearch"]["accuracy"], -metrics[tag]["cvsearch"]["mean_pop"],
                -sam_summaries["candidate" if tag in {"sam_only", "both"} else "baseline"]["negative_fpr_at_0_5"],
            ),
        ) if eligible else None
        accepted = selected is not None
    promotion = {"round": round_id, "stage": stage, "selected": selected, "accepted": accepted, "decisions": decisions}
    write_json(round_root / "promotion.json", promotion)

    if accepted and selected is not None:
        selected_sam, selected_mllm = combinations[selected]
        state.update({
            "current_sam": selected_sam, "current_mllm": selected_mllm,
            "next_round": round_id + 1,
            "accepted_rounds": state["accepted_rounds"] + [{
                "round": round_id, "combination": selected,
                "sam": selected_sam, "mllm": selected_mllm,
            }],
        })
    else:
        state["stopped"] = True
    write_json(_state_path(config, stage), state)
    write_json(round_root / ".done", {"completed": True, "round": round_id, "stage": stage})
    return promotion


def evaluate_checkpoint(
    config: dict[str, Any], split: str, checkpoint: str, stage: str,
) -> dict[str, Any]:
    state = load_or_initialize_state(config, stage)
    if split == "test":
        max_rounds = int(config["promotion"]["max_rounds"])
        finalized = bool(state["stopped"]) or int(state["next_round"]) >= max_rounds
        if not finalized:
            raise RuntimeError(
                f"test is sealed until model selection finishes; next_round={state['next_round']} max_rounds={max_rounds}"
            )
    if checkpoint == "t0":
        sam_path, mllm_path = config["models"]["sam_t0"], config["models"]["mllm_t0"]
        round_id = -1
    else:
        sam_path, mllm_path = state["current_sam"], state["current_mllm"]
        round_id = max(0, int(state["next_round"]) - 1)
    trajectory = generate_trajectories_parallel(
        config, split, "full", round_id, sam_path, mllm_path, workers=8,
        tag=f"evaluation_{checkpoint}",
    )
    direct = generate_direct_answers(config, split, "full", round_id, mllm_path, f"evaluation_{checkpoint}")
    result = {
        "split": split, "checkpoint": checkpoint, "round": round_id,
        "sam": sam_path, "mllm": mllm_path,
        "cvsearch": trajectory_metrics(trajectory, int(config["seed"])),
        "sam_trace": sam_trace_metrics(trajectory),
        "direct": trajectory_metrics(direct, int(config["seed"])),
    }
    output = workspace(config) / "runs" / "final" / f"{split}_{checkpoint}.json"
    write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CVSearch/SAM3/Vision-OPD self-evolution orchestrator")
    parser.add_argument("--config", default="configs/vopd_qwen3vl_2b.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("prepare")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--round", type=int, default=0)
    round_parser = sub.add_parser("run-round")
    round_parser.add_argument("--round", type=int, required=True)
    round_parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    loop = sub.add_parser("run-loop")
    loop.add_argument("--max-rounds", type=int, default=3)
    loop.add_argument("--stage", choices=("smoke", "full"), default="full")
    generate = sub.add_parser("generate")
    generate.add_argument("--split", choices=("train", "valid", "test"), required=True)
    generate.add_argument("--stage", choices=("smoke", "full"), default="full")
    generate.add_argument("--round", type=int, required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--num-shards", type=int, default=1)
    generate.add_argument("--sam-path")
    generate.add_argument("--mllm-path")
    generate.add_argument("--tag")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--split", choices=("valid", "test"), default="test")
    evaluate.add_argument("--checkpoint", choices=("best", "t0"), default="best")
    evaluate.add_argument("--stage", choices=("full",), default="full")
    sub.add_parser("report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "doctor":
        print(json.dumps({
            "ok": True,
            "config": config["_config_path"],
            "workspace": str(workspace(config)),
            "repositories": config["repositories"],
            "validated_sam_method": config["validated_sam_method"],
            "mllm": config["models"]["mllm_t0"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "prepare":
        snapshot_repositories(config, "before")
        print(json.dumps(prepare(config), indent=2, ensure_ascii=False))
    elif args.command == "smoke":
        print(json.dumps(run_round(config, "smoke", args.round), indent=2, ensure_ascii=False))
    elif args.command == "run-round":
        print(json.dumps(run_round(config, args.stage, args.round), indent=2, ensure_ascii=False))
    elif args.command == "run-loop":
        state = load_or_initialize_state(config, args.stage)
        while not state["stopped"] and int(state["next_round"]) < args.max_rounds:
            run_round(config, args.stage, int(state["next_round"]))
            state = load_or_initialize_state(config, args.stage)
        print(json.dumps(state, indent=2, ensure_ascii=False))
    elif args.command == "generate":
        state = load_or_initialize_state(config, args.stage)
        print(generate_trajectories(
            config, args.split, args.stage, args.round,
            args.sam_path or state["current_sam"], args.mllm_path or state["current_mllm"],
            limit=args.limit, shard_index=args.shard_index, num_shards=args.num_shards,
            tag=args.tag,
        ))
    elif args.command == "evaluate":
        print(json.dumps(
            evaluate_checkpoint(config, args.split, args.checkpoint, args.stage),
            indent=2, ensure_ascii=False,
        ))
    elif args.command == "report":
        snapshot_repositories(config, "after")
        print(generate_report(config))
