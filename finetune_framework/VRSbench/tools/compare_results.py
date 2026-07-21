#!/usr/bin/env python3
"""
对比基座模型与微调模型的评估结果，生成格式化表格。

用法:
    python compare_results.py <eval_results.json>              # 输出到同名 .md 文件
    python compare_results.py <eval_results.json> -o out.md    # 指定输出路径
    python compare_results.py <eval_results.json> -t           # 打印终端表格
    python compare_results.py <eval_results.json> -t --md      # 打印 Markdown 到终端
"""

import json
import sys
import os

# ── 终端表格字符 ──────────────────────────────────────────────
BOX = {
    "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
    "h": "─", "v": "│",
    "t": "┬", "b": "┴", "l": "├", "r": "┤", "c": "┼",
}

EPSILON = 1e-8  # 小于此值视为 0

TASK_NAMES = {
    "caption": "CAPTION",
    "vqa": "VQA",
    "referring": "REFERRING",
}

METRIC_ORDER = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr-D"]
REFERRING_METRICS = ["mean_iou", "Acc@0.25", "Acc@0.5", "Acc@0.7"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_change(diff: float) -> str:
    """格式化变化量，正数带 + 号"""
    if diff == 0:
        return "+0.0000"
    return f"{diff:+.4f}"


def is_zero(v: float) -> bool:
    return abs(v) < EPSILON


def fmt_improvement(base_val: float, ft_val: float) -> str:
    """格式化提升百分比"""
    if is_zero(base_val) and is_zero(ft_val):
        return "        —   "
    if is_zero(base_val):
        return "      NEW  ▲"
    pct = (ft_val - base_val) / base_val * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct: 8.1f}%  {'▲' if pct >= 0 else '▼'}"


def fmt_val(v: float) -> str:
    if is_zero(v):
        return "        0.0000"
    return f"{v: 12.4f}"


def get_metric_list(task_key: str) -> list:
    """根据任务类型返回对应的指标列表。"""
    if task_key == "referring":
        return REFERRING_METRICS
    return METRIC_ORDER


def render_terminal(data: dict) -> str:
    """渲染 box-drawing 终端表格"""
    lines = []
    tasks = data["config"]["tasks"]
    base = data["base_model"]
    ft = data["finetuned_model"]

    # 列配置：指标名 | 基座模型 | 微调模型 | 变化 | 提升
    COL_W = [16, 16, 16, 16, 16]
    HEADERS = ["指标", "基座模型", "微调模型", "变化", "提升"]

    for task_key in tasks:
        task_name = TASK_NAMES.get(task_key, task_key.upper())
        bm = base[task_key]["metrics"]
        fm = ft[task_key]["metrics"]
        metric_list = get_metric_list(task_key)

        w_total = sum(COL_W) + len(COL_W) - 1
        # 顶边框 + 任务标题
        lines.append(f"{BOX['tl']}{BOX['h'] * w_total}{BOX['tr']}")
        title = f" 任务: {task_name}"
        lines.append(f"{BOX['v']}{title:<{w_total}}{BOX['v']}")
        # 分隔线
        lines.append(f"{BOX['l']}{BOX['h'] * w_total}{BOX['r']}")

        # 表头
        hdr = BOX["v"]
        for i, h in enumerate(HEADERS):
            hdr += f"{h:^{COL_W[i]}}{BOX['v']}"
        lines.append(hdr)

        # 分隔线 ───┼───
        sep = BOX["l"]
        for i, w in enumerate(COL_W):
            sep += BOX["h"] * w
            sep += BOX["c"] if i < len(COL_W) - 1 else BOX["r"]
        lines.append(sep)

        for metric in metric_list:
            bv = bm.get(metric, 0)
            fv = fm.get(metric, 0)
            diff = fv - bv
            row = BOX["v"]
            row += f" {metric:<{COL_W[0] - 1}}{BOX['v']}"
            row += f"{fmt_val(bv)}{BOX['v']}"
            row += f"{fmt_val(fv)}{BOX['v']}"
            row += f" {fmt_change(diff):>{COL_W[3] - 1}}{BOX['v']}"
            row += f" {fmt_improvement(bv, fv):>{COL_W[4] - 1}}{BOX['v']}"
            lines.append(row)

        # 样本信息分隔行
        lines.append(sep)

        bv_samples = bm.get("valid_samples", 0)
        fv_samples = fm.get("valid_samples", 0)
        row = BOX["v"]
        row += f" 有效样本"
        row += f"{'':>{COL_W[0] - 4 - 1}}{BOX['v']}"
        row += f"{bv_samples: 14d}  {BOX['v']}"
        row += f"{fv_samples: 14d}  {BOX['v']}"
        row += f"{'':>{COL_W[3]}}{BOX['v']}"
        row += f"{'':>{COL_W[4]}}{BOX['v']}"
        lines.append(row)

        # 底边框
        lines.append(f"{BOX['bl']}{BOX['h'] * w_total}{BOX['br']}")
        lines.append("")  # 空行分隔任务

    return "\n".join(lines)


def render_markdown(data: dict) -> str:
    """渲染完整的 Markdown 报告（含汇总表 + 各任务明细）"""
    tasks = data["config"]["tasks"]
    base = data["base_model"]
    ft = data["finetuned_model"]

    out_lines = []

    # ── 页头信息 ──
    cfg = data["config"]
    out_lines.append("# 微调效果对比报告")
    out_lines.append("")
    out_lines.append(f"- **基座模型**: `{cfg.get('base_model', 'N/A')}`")
    out_lines.append(f"- **LoRA 权重**: `{cfg.get('lora_path', 'N/A')}`")
    out_lines.append(f"- **任务**: {', '.join(TASK_NAMES.get(t, t) for t in tasks)}")
    out_lines.append(f"- **样本数**: {cfg.get('max_samples', 'N/A')}")
    out_lines.append("")

    # ── 汇总总表：一个表展示所有任务 × 所有指标 ──
    out_lines.append("## 总览")
    out_lines.append("")
    # 表头
    header = "| 任务 | 指标 | 基座模型 | 微调模型 | 变化 | 提升 |"
    sep =    "|------|------|---------|---------|------|------|"
    out_lines.append(header)
    out_lines.append(sep)

    for task_key in tasks:
        task_name = TASK_NAMES.get(task_key, task_key.upper())
        bm = base[task_key]["metrics"]
        fm = ft[task_key]["metrics"]
        metric_list = get_metric_list(task_key)
        first = True
        for metric in metric_list:
            bv = bm.get(metric, 0)
            fv = fm.get(metric, 0)
            diff = fv - bv
            if is_zero(bv) and is_zero(fv):
                impr = "—"
            elif is_zero(bv):
                impr = "NEW ▲"
            else:
                pct = (fv - bv) / bv * 100
                impr = f"{pct:+.1f}% {'▲' if pct >= 0 else '▼'}"
            label = f"**{task_name}**" if first else ""
            first = False
            out_lines.append(
                f"| {label} | {metric} | {bv:.4f} | {fv:.4f} | {diff:+.4f} | {impr} |"
            )
        # 有效样本行
        bv_s = bm.get("valid_samples", 0)
        fv_s = fm.get("valid_samples", 0)
        out_lines.append(
            f"| | 有效样本 | {bv_s} | {fv_s} | | |"
        )
    out_lines.append("")

    # ── 各任务明细表 ──
    out_lines.append("## 各任务明细")
    out_lines.append("")

    for task_key in tasks:
        task_name = TASK_NAMES.get(task_key, task_key.upper())
        bm = base[task_key]["metrics"]
        fm = ft[task_key]["metrics"]
        metric_list = get_metric_list(task_key)

        out_lines.append(f"### {task_name}")
        out_lines.append("")
        out_lines.append("| 指标 | 基座模型 | 微调模型 | 变化 | 提升 |")
        out_lines.append("|------|---------|---------|------|------|")

        for metric in metric_list:
            bv = bm.get(metric, 0)
            fv = fm.get(metric, 0)
            diff = fv - bv
            if is_zero(bv) and is_zero(fv):
                impr = "—"
            elif is_zero(bv):
                impr = "NEW ▲"
            else:
                pct = (fv - bv) / bv * 100
                impr = f"{pct:+.1f}% {'▲' if pct >= 0 else '▼'}"
            out_lines.append(
                f"| {metric} | {bv:.4f} | {fv:.4f} | {diff:+.4f} | {impr} |"
            )

        bv_samples = bm.get("valid_samples", 0)
        fv_samples = fm.get("valid_samples", 0)
        out_lines.append(
            f"| 有效样本 | {bv_samples} | {fv_samples} | | |"
        )
        out_lines.append("")

    return "\n".join(out_lines)


def render_terminal_md(data: dict) -> str:
    """渲染纯 Markdown 文本（供 --md 打印到终端）"""
    # 复用 render_markdown 即可
    return render_markdown(data)


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {os.path.basename(__file__)} <eval_results.json> [-o out.md] [-t] [--md]")
        print(f"")
        print(f"  默认：输出到与 JSON 同名的 .md 文件")
        print(f"  -o <path>  指定输出文件路径")
        print(f"  -t         打印终端表格（box-drawing）")
        print(f"  --md       配合 -t，打印 Markdown 到终端")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_json(json_path)

    # 解析参数
    to_terminal = "-t" in sys.argv
    to_terminal_md = "--md" in sys.argv
    output_path = None
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    if to_terminal:
        # 打印到终端
        if to_terminal_md:
            print(render_markdown(data))
        else:
            print(render_terminal(data))
    else:
        # 写入文件
        if output_path is None:
            base_name = os.path.splitext(json_path)[0]
            output_path = f"{base_name}_compare.md"
        md_content = render_markdown(data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"📁 对比报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
