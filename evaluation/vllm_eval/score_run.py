#!/usr/bin/env python3
"""Score a vLLM evaluation output using the repo's official aggregation.

用法: python score_run.py <outdir>
读取 <outdir>/prediction_attempts.jsonl，每行含 sample 与 prediction；
先经 score_prediction 得到 score，再走 summarize_predictions 的分组聚合
（离散题 accuracy 只在 clean_eligible 上算；caption 用 caption_metrics、
grounding 用 bbox_metrics —— 与评测器同一份官方口径）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Sample  # noqa: E402
from scoring import score_prediction, summarize_predictions  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法: {Path(sys.argv[0]).name} <outdir>", file=sys.stderr)
        return 2
    outdir = sys.argv[1]
    attempts = Path(outdir) / "prediction_attempts.jsonl"

    rows = []
    for line in attempts.open(encoding="utf-8"):
        r = json.loads(line)
        sample_dict = r["sample"]
        sample = Sample.from_dict(sample_dict)
        score = score_prediction(sample, r.get("prediction", ""))
        # summarize_predictions 需要 sample 作为 dict（按 dataset/subtask 分组）
        rows.append({"sample": sample_dict, "prediction": r.get("prediction", ""), "score": score})

    official, clean = summarize_predictions(rows)

    print(f"=== total {len(rows)} rows ===")
    print(f"protocol: {clean['protocol']}")
    diag = clean["global_discrete_micro_diagnostic"]
    if diag["scoreable_samples"]:
        print(
            f"global clean_acc: {diag['accuracy']:.4f} "
            f"({diag['correct']}/{diag['scoreable_samples']})"
        )
    else:
        print("无离散题(纯 caption/bbox 集)，global clean_acc=N/A")

    print("\n--- 按数据集 (clean micro) ---")
    for ds, s in sorted(clean["datasets"].items()):
        acc = s.get("micro_accuracy")
        acc_s = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"{ds}: {acc_s} ({s.get('correct')}/{s.get('scoreable_samples')})")

    print("\n--- 按数据集/子任务 (clean) ---")
    for gname, g in sorted(clean["groups"].items()):
        parts = []
        acc = g.get("accuracy")
        parts.append(f"acc={acc:.4f}" if acc is not None else "acc=N/A")
        parts.append(f"eligible={g.get('eligible_scoreable_samples')}")
        if g.get("bbox_metrics"):
            b = g["bbox_metrics"]
            parts.append(f"bbox_acc05={b.get('accuracy_at_0_5')}")
        if g.get("caption_smoke_metrics"):
            parts.append(f"caption={g['caption_smoke_metrics']}")
        print(f"{gname}: " + " | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
