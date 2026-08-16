#!/usr/bin/env python3
"""
独立指标计算脚本 — 修复了 MCQAccuracy 的答案提取正则，支持 (X) 格式。
直接读取 prediction JSON 文件，无需重跑推理。

用法:
  python3 prune/eval_metrics.py --predictions outputs/inference_results/predictions_mme.json
  python3 prune/eval_metrics.py --all-scope-runs  # 批量计算所有 SCOPE 结果
"""
import argparse, json, os, re, sys
from collections import defaultdict


# ── Fixed answer extraction ──

def extract_mcq_letter(text: str) -> str:
    """从 MCQ 模型输出中提取选项字母 A-E，支持多种格式。"""
    if not text:
        return ""
    text = text.strip()

    # 1) (X) 括号格式 — 模型最常见输出: "(C)", "(A)"
    m = re.search(r'\(([A-Ea-e])\)', text)
    if m:
        return m.group(1).upper()

    # 2) X. 或 X) 前导格式: "D.", "D)", "B. (B) White"
    m = re.search(r'(?<!\w)([A-Ea-e])\s*[.)]', text)
    if m:
        return m.group(1).upper()

    # 3) 独立行单个字母
    m = re.search(r'(?:^|\n)\s*([A-Ea-e])\s*(?:\n|$)', text)
    if m:
        return m.group(1).upper()

    # 4) "Answer: X" / "option X" 等前缀
    m = re.search(r'(?:answer|option|答案|选择)\s*(?:is|:|：)?\s*\(?([A-Ea-e])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 5) 最后一个 (X) 格式（兜底）
    matches = re.findall(r'\(([A-Ea-e])\)', text)
    if matches:
        return matches[-1].upper()

    return ""


def extract_ref_letter(ref: str) -> str:
    """从参考答案中提取选项字母。"""
    if not ref:
        return ""
    # "D. (D) White" → D
    m = re.search(r'^([A-Ea-e])[.)]', ref.strip())
    if m:
        return m.group(1).upper()
    m = re.search(r'\(([A-Ea-e])\)', ref)
    if m:
        return m.group(1).upper()
    return ""


# ── Metrics ──

def compute_mcq_accuracy(predictions: list) -> dict:
    """计算 MCQ Accuracy，使用修复后的答案提取。"""
    correct = 0
    total = 0
    no_pred_letter = 0
    per_sample = []

    for r in predictions:
        ref = r.get('references', [''])[0]
        pred = r.get('prediction', '')
        ref_letter = extract_ref_letter(ref)
        if not ref_letter:
            continue

        pred_letter = extract_mcq_letter(pred)
        total += 1

        if not pred_letter:
            no_pred_letter += 1
            per_sample.append({'status': 'no_letter', 'ref': ref[:60], 'pred': pred[:80]})
        elif pred_letter == ref_letter:
            correct += 1
            per_sample.append({'status': 'correct', 'ref': ref[:60], 'pred': pred[:80]})
        else:
            per_sample.append({'status': 'wrong', 'ref': ref[:60], 'pred': pred[:80], 
                               'ref_letter': ref_letter, 'pred_letter': pred_letter})

    acc = correct / total if total > 0 else 0
    return {
        'MCQ_Accuracy': acc,
        'total': total,
        'correct': correct,
        'no_letter': no_pred_letter,
        'wrong_letter': total - correct - no_pred_letter,
        'per_sample': per_sample,
    }


def compute_levircc_metrics(predictions: list) -> dict:
    """计算 LEVIR-CC 的 BLEU/ROUGE/CIDEr 指标。"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("  [WARN] nltk not installed, BLEU will be 0")
        return {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0, 'ROUGE-L': 0, 'CIDEr': 0}

    import numpy as np
    smoothie = SmoothingFunction().method1

    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    rouge_scores = []
    cider_refs = []  # for CIDEr computation
    cider_preds = []

    for r in predictions:
        pred = (r.get('prediction') or '').strip()
        raw_refs = r.get('references', [])
        refs = []
        for ref in raw_refs:
            if isinstance(ref, str):
                refs.append(ref.strip())
            elif isinstance(ref, dict):
                txt = ref.get('raw', ' '.join(ref.get('tokens', [])) if isinstance(ref.get('tokens', []), list) else '')
                refs.append(txt.strip())
            else:
                refs.append(str(ref).strip())
        refs = [r for r in refs if r]
        if not pred or not refs:
            continue

        cider_refs.append(refs)
        cider_preds.append(pred)

        ref_tokens = [ref.split() for ref in refs]
        pred_tokens = pred.split()

        if not pred_tokens:
            continue

        for n in [1, 2, 3, 4]:
            try:
                b = sentence_bleu(ref_tokens, pred_tokens, weights=tuple([1.0/n]*n),
                                  smoothing_function=smoothie)
            except Exception:
                b = 0.0
            bleu_scores[n].append(b)

        # ROUGE-L
        pred_set = set(pred_tokens)
        rl = []
        for rt in ref_tokens:
            ref_set = set(rt)
            if not ref_set:
                rl.append(0)
                continue
            overlap = len(pred_set & ref_set)
            rl.append(overlap / len(ref_set) if ref_set else 0)
        rouge_scores.append(max(rl) if rl else 0)

    result = {}
    for n, scores in bleu_scores.items():
        result[f'BLEU-{n}'] = float(np.mean(scores)) if scores else 0.0
    result['ROUGE-L'] = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    result['CIDEr'] = _compute_cider(cider_refs, cider_preds)
    return result


def _compute_cider(refs_list, preds_list):
    """CIDEr metric for image caption evaluation."""
    from math import log, sqrt
    from collections import Counter
    import numpy as np

    all_refs = []
    all_preds = []
    for refs, pred in zip(refs_list, preds_list):
        if not pred or not refs:
            continue
        all_preds.append(pred.split())
        all_refs.append([r.split() for r in refs])

    if not all_preds:
        return 0.0

    N = len(all_preds)
    cider_scores = []

    for n in range(1, 5):
        df = Counter()
        for ref_group in all_refs:
            seen = set()
            for tokens in ref_group:
                for i in range(len(tokens) - n + 1):
                    g = tuple(tokens[i:i + n])
                    if g not in seen:
                        df[g] += 1
                        seen.add(g)

        n_scores = []
        for pred_tokens, ref_group in zip(all_preds, all_refs):
            if not pred_tokens:
                continue
            hyp_ng = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
            hyp_max = max(hyp_ng.values()) if hyp_ng else 1
            hyp_vec = {g: c / hyp_max * (log((N + 1) / (df.get(g, 0) + 1)) + 1)
                       for g, c in hyp_ng.items()}

            ref_scores = []
            for ref_tokens in ref_group:
                if not ref_tokens:
                    continue
                ref_ng = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
                ref_max = max(ref_ng.values()) if ref_ng else 1
                ref_vec = {g: c / ref_max * (log((N + 1) / (df.get(g, 0) + 1)) + 1)
                           for g, c in ref_ng.items()}

                dot = sum(hyp_vec.get(k, 0) * ref_vec.get(k, 0) for k in set(hyp_vec) | set(ref_vec))
                n1 = sqrt(sum(v**2 for v in hyp_vec.values()))
                n2 = sqrt(sum(v**2 for v in ref_vec.values()))
                ref_scores.append(dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0)

            n_scores.append(np.mean(ref_scores) if ref_scores else 0.0)

        cider_scores.append(float(np.mean(n_scores)) if n_scores else 0.0)

    weights = [0.25, 0.25, 0.25, 0.25]
    cider = sum(w * s for w, s in zip(weights, cider_scores))
    return float(max(cider * 10.0, 0.0))


# ── Main ──

def _get_ref_text(refs):
    """从可能的 references 格式中提取第一个参考文本。"""
    if not refs:
        return ""
    r0 = refs[0]
    if isinstance(r0, str):
        return r0
    if isinstance(r0, dict):
        return r0.get('raw', r0.get('tokens', [''])[0] if isinstance(r0.get('tokens', []), list) else '')
    return str(r0)


def evaluate_file(path: str):
    """评测单个 prediction JSON 文件。"""
    with open(path) as f:
        preds = json.load(f)

    if not preds:
        return {}

    # 检测数据集类型: 有 references 且选项格式 → MCQ
    sample = preds[0]
    ref0 = _get_ref_text(sample.get('references', []))
    is_mcq = bool(ref0 and re.search(r'\b[A-E][.)]', ref0))

    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(path)}")
    print(f"Samples: {len(preds)}  |  Type: {'MCQ' if is_mcq else 'caption/other'}")
    print(f"{'='*60}")

    if is_mcq:
        result = compute_mcq_accuracy(preds)
        acc = result['MCQ_Accuracy']
        print(f"  MCQ Accuracy:     {acc*100:.2f}%  ({result['correct']}/{result['total']})")
        print(f"  No letter in pred: {result['no_letter']}/{result['total']} ({100*result['no_letter']/max(result['total'],1):.1f}%)")
        print(f"  Wrong letter:      {result['wrong_letter']}/{result['total']} ({100*result['wrong_letter']/max(result['total'],1):.1f}%)")

        # Show some error examples
        if result['no_letter'] > 0:
            print(f"\n  === Samples with NO letter extracted (first 5) ===")
            for s in result['per_sample']:
                if s['status'] == 'no_letter':
                    print(f"    pred: {s['pred'][:100]}")
                    print(f"    ref:  {s['ref'][:100]}")
                    print()
                if sum(1 for x in result['per_sample'] if x['status'] == 'no_letter' and 
                       x == s) >= 5:
                    break
        
        return {'MCQ_Accuracy': acc, 'samples': len(preds), **{k: result[k] for k in ['correct', 'no_letter', 'wrong_letter']}}
    else:
        result = compute_levircc_metrics(preds)
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")
        return {**result, 'samples': len(preds)}


def main():
    parser = argparse.ArgumentParser(description='Compute metrics for SCOPE pruning results')
    parser.add_argument('--predictions', type=str, nargs='+', help='Prediction JSON file(s)')
    parser.add_argument('--all-scope-runs', action='store_true', help='Evaluate all scope results')
    parser.add_argument('--output', type=str, default=None, help='Save summary JSON')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    files = []
    if args.predictions:
        files.extend(args.predictions)
    if args.all_scope_runs:
        # Collect all scope prediction files
        out_dirs = [
            os.path.join(repo_root, 'outputs', 'inference_results'),
            os.path.join(repo_root, 'outputs', 'inference_results', 'scope_r01'),
            os.path.join(repo_root, 'prune', 'output'),
        ]
        for d in out_dirs:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith('predictions_') and f.endswith('.json'):
                        files.append(os.path.join(d, f))
        # Also check for VRSBench chunk results
        vrs_dir = os.path.join(repo_root, 'prune', 'output')
        if os.path.isdir(vrs_dir):
            for f in sorted(os.listdir(vrs_dir)):
                if 'vrs' in f and f.endswith('.json') and 'scope' in f:
                    files.append(os.path.join(vrs_dir, f))

    if not files:
        print("No files specified. Use --predictions or --all-scope-runs")
        return

    summary = {}
    for path in files:
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        result = evaluate_file(path)
        if result:
            label = os.path.basename(path).replace('.json', '')
            summary[label] = result

    # Print summary table
    if len(summary) > 1:
        print(f"\n{'='*60}")
        print(f"{'Dataset':30s} {'Samples':>8s} {'MCQ Acc':>10s} {'No Letter':>10s}")
        print(f"{'─'*30} {'─'*8} {'─'*10} {'─'*10}")
        for label, r in sorted(summary.items()):
            if 'MCQ_Accuracy' in r:
                print(f"{label:30s} {r.get('samples',0):8d} {r['MCQ_Accuracy']*100:>9.2f}% {r.get('no_letter',0):10d}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
