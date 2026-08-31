#!/usr/bin/env python3
"""Score a vLLM evaluation output (clean_correct official protocol).

用法: python score_run.py <outdir>
读取 <outdir>/prediction_attempts.jsonl，输出 clean_correct 准确率。
同一目录下的 schema/scoring 模块即评测器所使用的官方计分。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import Sample  # noqa: E402
from scoring import score_prediction  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(f"用法: {Path(sys.argv[0]).name} <outdir>", file=sys.stderr)
        return 2
    outdir = sys.argv[1]
    attempts = Path(outdir) / "prediction_attempts.jsonl"
    corr = tot = 0
    per_task: dict[str, list[int]] = {}
    for line in attempts.open(encoding="utf-8"):
        r = json.loads(line)
        s = Sample.from_dict(r["sample"])
        sc = score_prediction(s, r.get("prediction", ""))
        ok = bool(sc.get("clean_correct")) if isinstance(sc, dict) else bool(sc)
        corr += ok
        tot += 1
        task = getattr(s, "subtask", "unknown") or "unknown"
        per_task.setdefault(task, [0, 0])
        per_task[task][0] += ok
        per_task[task][1] += 1
    print(f"accuracy: {corr}/{tot} = {corr / tot:.4f}")
    for task, (ok, n) in sorted(per_task.items()):
        print(f"  {task}: {ok}/{n} = {ok / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
