#!/usr/bin/env python3
"""构建 Caption 专家双域训练数据集。

两个长度域，指令均置于 user 文本（与评测端路线 X 对齐，不依赖 system）：
  - VRS 域  : human=<image>\n[CAP] Describe the image in detail，gpt=~53词短描述
  - XLRS 域 : human=<image>\n<九宫格长指令全文>，gpt=~598词长描述

输出 jsonl，结构与 g_a2_mix 一致（image 相对路径 + conversations human/gpt）。
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()

VRS_INSTRUCTION = "[CAP] Describe the image in detail"

def load_g_a2_mix_caption(path):
    d = json.load(open(path))
    return [s for s in d if "Describe the image" in s["conversations"][0]["value"]]


def build_vrs(short_samples):
    """VRS 域：短描述，指令=评测一致 [CAP] Describe...。"""
    out = []
    for s in short_samples:
        out.append({
            "id": s.get("id", "vrs_cap"),
            "image": s["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{VRS_INSTRUCTION}"},
                {"from": "gpt", "value": s["conversations"][1]["value"]},
            ],
        })
    return out


def load_long_desc(path):
    with open(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def build_xlrs(long_samples, xlrs_instruction):
    """XLRS 域：长描述，指令=九宫格元指令全文（拼入 user 文本）。"""
    out = []
    for s in long_samples:
        out.append({
            "id": s.get("id", "xlrs_cap"),
            "image": s["image"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{xlrs_instruction}"},
                {"from": "gpt", "value": s["conversations"][1]["value"]},
            ],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g_a2_mix", default=str(REPO_ROOT / "datasets" / "training35" / "ga2_general.json"))
    ap.add_argument("--long_desc", default=str(REPO_ROOT / ".models" / "five-stage-data" / "caption_dual_domain_train.jsonl"),
                    help="历史长描述 JSON/JSONL；默认读取 ModelScope staging 数据")
    ap.add_argument("--xlrs_prompt", default=str(REPO_ROOT / "evaluation" / "prompts" / "xlrs_caption_en.txt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "datasets" / "training35" / "caption.generated.json"),
                    help="重建结果；不覆盖训练入口当前指向的已校验数据")
    ap.add_argument("--xlrs_ratio", type=float, default=1.0,
                    help="XLRS长样本数 = 该比例 * VRS短样本数（默认1:1）")
    args = ap.parse_args()

    xlrs_inst = resolve_path(args.xlrs_prompt).read_text(encoding="utf-8").strip()

    short = load_g_a2_mix_caption(str(resolve_path(args.g_a2_mix)))
    long = load_long_desc(str(resolve_path(args.long_desc)))
    print(f"VRS 短样本: {len(short)}，XLRS 长样本: {len(long)}")

    # 对齐：以 image 为 key，保证长短用同一批图（要求完全覆盖）
    long_by_img = {s["image"]: s for s in long}
    vrs_samples = []
    for s in short:
        if s["image"] in long_by_img:
            vrs_samples.append(s)
    print(f"对齐（长短均有图的）VRS 样本: {len(vrs_samples)}")

    vrs_ds = build_vrs(vrs_samples)
    n_xlrs = max(1, int(len(vrs_samples) * args.xlrs_ratio))
    long_ds = build_xlrs([long_by_img[s["image"]] for s in vrs_samples[:n_xlrs]], xlrs_inst)

    all_data = vrs_ds + long_ds
    output = resolve_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=1)
    print(f"写入 {output}: 共 {len(all_data)} 条 (VRS {len(vrs_ds)} + XLRS {len(long_ds)})")


if __name__ == "__main__":
    main()