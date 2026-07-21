#!/usr/bin/env python3
"""
从 VRSBench_train.json 中提取 referring 任务数据，用于 GRPO 微调。

用法:
    python prepare_referring_grpo_data.py                          # 生成全量数据
    python prepare_referring_grpo_data.py --max_samples 10000      # 固定种子采样
    python prepare_referring_grpo_data.py --val_split 0.05         # 额外划分验证集
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="从 VRSBench 训练数据中提取 referring 任务，准备 GRPO 微调数据"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="输入 JSON 路径 (默认: VRSBench/data/VRSBench_train.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 JSON 路径 (默认: VRSBench/data/VRSBench_referring_grpo_full.json)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="验证集比例 (默认 0，不划分)",
    )
    parser.add_argument(
        "--val_output",
        default=None,
        help="验证集输出路径",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最大样本数 (0=全部，默认0)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")

    input_path = args.input or os.path.join(data_dir, "VRSBench_train.json")
    output_path = args.output or os.path.join(
        data_dir, "VRSBench_referring_grpo_full.json"
    )
    val_output = args.val_output or os.path.join(
        data_dir, "VRSBench_referring_grpo_full_val.json"
    )

    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"原始数据总量: {len(all_data)} 条")

    # 筛选 referring 任务: 包含 [refer] 标记的对话
    referring_data = []
    for item in all_data:
        conversations = item.get("conversations", [])
        if not conversations:
            continue
        # 检查 human 消息中是否包含 [refer]
        human_msg = conversations[0] if conversations else {}
        if isinstance(human_msg, dict) and "[refer]" in human_msg.get("value", ""):
            referring_data.append(item)

    print(f"Referring 任务数据: {len(referring_data)} 条")

    # 采样
    if args.max_samples > 0 and len(referring_data) > args.max_samples:
        import random
        random.seed(42)
        referring_data = random.sample(referring_data, args.max_samples)
        print(f"采样后数据: {len(referring_data)} 条 (max_samples={args.max_samples})")

    # 统计唯一的图片数量
    unique_images = len(set(item["image"] for item in referring_data))
    print(f"涉及图片数: {unique_images} 张")

    # 输出示例
    if referring_data:
        print("\n样例数据:")
        sample = referring_data[0]
        print(f"  image: {sample['image']}")
        for conv in sample["conversations"]:
            print(f"  [{conv['from']}]: {conv['value'][:120]}...")

    # 保存
    if args.val_split > 0:
        import random
        random.seed(42)
        random.shuffle(referring_data)
        split_idx = int(len(referring_data) * (1 - args.val_split))
        train_data = referring_data[:split_idx]
        val_data = referring_data[split_idx:]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(val_output, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 训练集: {len(train_data)} 条 → {output_path}")
        print(f"✅ 验证集: {len(val_data)} 条 → {val_output}")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(referring_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 已保存: {output_path}")


if __name__ == "__main__":
    main()
