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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.metrics import BLEU, CIDEr, ROUGEL


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
    """计算 LEVIR-CC 指标，复用 evaluation/metrics 的标准实现。"""
    references = []
    predictions_text = []
    for record in predictions:
        refs = []
        for ref in record.get("references", []):
            if isinstance(ref, str):
                text = ref.strip()
            elif isinstance(ref, dict):
                tokens = ref.get("tokens", [])
                text = ref.get("raw", " ".join(tokens) if isinstance(tokens, list) else "")
                text = str(text).strip()
            else:
                text = str(ref).strip()
            if text:
                refs.append(text)
        if not refs:
            refs = [""]
        references.append(refs)
        predictions_text.append((record.get("prediction") or "").strip())

    result = {}
    for metric in (BLEU(max_n=4), ROUGEL(), CIDEr()):
        result.update(metric.compute(references, predictions_text))
    return result


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
