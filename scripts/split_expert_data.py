#!/usr/bin/env python3
"""按任务切分清洗后 manifest 为 3 个专家训练集 + 合成 MCQ。

遵循 TaskExpertLoRA.md 方案（2026-08-05 定稿）：
- General Understanding = VQA + Caption + 合成 MCQ（比例 ~70/20/10）
- Grounding = Referring
- Change = LEVIR-CC

输入: finetune_framework/VRSbench/manifest_sft_train.json
输出: finetune_framework/VRSbench/expert_data/
  general_understanding.json   (vqa + caption + mcq)
  grounding.json               (refer)
  change.json                  (levir)
  stats.json                   (切分统计，用于复现)
"""
import argparse
import json
import os
import random
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "finetune_framework/VRSbench/manifest_sft_train.json")
OUT_DIR = os.path.join(REPO, "finetune_framework/VRSbench/expert_data")

DEFAULT_SEED = 42

MCQ_SYSTEM_PROMPT = (
    'Answer EXACTLY in format "X. (X) FullOptionText" with the letter '
    "repeated in parentheses. Example: \"D. (D) White\". "
    "You MUST include the parenthesized letter - never omit it. Output ONLY that line."
)


def classify(sample):
    """按 id 判断任务类型：vqa / caption / refer / levir"""
    i = sample["id"]
    if i.startswith("levir_cc"):
        return "levir"
    low = i.lower()
    if "/vqa/" in low:
        return "vqa"
    if "/caption/" in low:
        return "caption"
    if "/refer/" in low:
        return "refer"
    raise ValueError(f"unknown id: {i}")


def build_mcq(vqa_samples, target, rand, rng):
    """从 vqa 样本构造四选一 MCQ，答案格式 X. (X) FullOptionText。

    - 正确答案 = 原 VQA 答案
    - 3 干扰项 = 从答案池随机抽取，排除近似重复（normalize 后去重）
    """
    norm = lambda a: re.sub(r"\s+", " ", a.strip().lower())
    distinct = sorted(set(s["conversations"][1]["value"].strip() for s in vqa_samples))
    pool = distinct
    pool_norm = [norm(a) for a in pool]
    norm_to_idx = {}
    for i, a in enumerate(pool):
        norm_to_idx.setdefault(pool_norm[i], i)

    mcq = []
    indices = rng.sample(range(len(vqa_samples)), target)
    letter_dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for idx in indices:
        s = vqa_samples[idx]
        q = re.sub(r"^<image>\s*\n?", "", s["conversations"][0]["value"])
        correct = s["conversations"][1]["value"].strip()
        ci_pool = norm_to_idx[norm(correct)]
        # 拒绝抽样：避免与正确项下标相同或规范化后重复
        picked = set()
        while len(picked) < 3:
            j = rng.randrange(len(pool))
            if j == ci_pool or pool_norm[j] == pool_norm[ci_pool] or j in picked:
                continue
            picked.add(j)
        distractors = [pool[j] for j in picked]
        opts = [correct] + distractors
        rng.shuffle(opts)
        ci = opts.index(correct)
        letter = "ABCD"[ci]
        letter_dist[letter] += 1

        lines = [f"[MCQ] {q}"]
        lines += [f"({chr(65+i)}) {o}" for i, o in enumerate(opts)]
        lines.append(MCQ_SYSTEM_PROMPT)

        mcq.append({
            "id": f"mcq_synth_{idx}",
            "image": s["image"],
            "conversations": [
                {"from": "human", "value": "<image>\n" + "\n".join(lines)},
                {"from": "gpt", "value": f"{letter}. ({letter}) {correct}"},
            ],
        })
    return mcq, letter_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--vqa-ratio", type=float, default=0.70,
                    help="VQA 占 General Understanding 的目标比例")
    ap.add_argument("--caption-ratio", type=float, default=0.20,
                    help="Caption 占 General Understanding 的目标比例")
    ap.add_argument("--mcq-ratio", type=float, default=0.10,
                    help="MCQ 占 General Understanding 的目标比例")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    ratios = [args.vqa_ratio, args.caption_ratio, args.mcq_ratio]
    if abs(sum(ratios) - 1.0) > 1e-6:
        ap.error(f"ratios must sum to 1.0, got {sum(ratios)}")

    with open(args.manifest) as f:
        data = json.load(f)

    buckets = {"vqa": [], "caption": [], "refer": [], "levir": []}
    for s in data:
        buckets[classify(s)].append(s)

    rand = random.Random(args.seed)
    rng = random.Random(args.seed)

    # 以 VQA 全量为锚点，反推总数与 caption/mcq 数量
    n_vqa = len(buckets["vqa"])
    total_gu = int(round(n_vqa / args.vqa_ratio))
    n_caption = int(round(total_gu * args.caption_ratio))
    n_mcq = int(round(total_gu * args.mcq_ratio))
    n_caption = min(n_caption, len(buckets["caption"]))
    n_mcq = max(n_mcq, 0)

    captions = rng.sample(buckets["caption"], n_caption) if n_caption else []
    mcq, letter_dist = build_mcq(buckets["vqa"], n_mcq, rand, rng)

    general = buckets["vqa"] + captions + mcq
    grounding = buckets["refer"]
    change = buckets["levir"]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, lst in [("general_understanding", general),
                      ("grounding", grounding),
                      ("change", change)]:
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(lst, f, indent=2, ensure_ascii=False)

    stats = {
        "manifest": args.manifest,
        "seed": args.seed,
        "vqa_ratio": args.vqa_ratio,
        "caption_ratio": args.caption_ratio,
        "mcq_ratio": args.mcq_ratio,
        "counts": {k: len(v) for k, v in buckets.items()},
        "mcq_synthetic": len(mcq),
        "mcq_letter_dist": letter_dist,
        "general_understanding": {
            "vqa": len(buckets["vqa"]),
            "caption": len(captions),
            "mcq": len(mcq),
            "total": len(general),
            "vqa_pct": round(100 * len(buckets["vqa"]) / len(general), 1),
            "caption_pct": round(100 * len(captions) / len(general), 1),
            "mcq_pct": round(100 * len(mcq) / len(general), 1),
        },
        "grounding": len(grounding),
        "change": len(change),
        "total": len(general) + len(grounding) + len(change),
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
