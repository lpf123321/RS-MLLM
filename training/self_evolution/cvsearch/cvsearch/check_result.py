#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import argparse
import os

def is_error_vstar(obj):
    """vstar：output非0即为错误"""
    output = obj.get("output")
    try:
        return int(output) != 0
    except (TypeError, ValueError):
        return True

def is_error_hrbench(obj):
    """HR-Bench：answer与output逐元素比较"""
    answer = obj.get("answer")
    output = obj.get("output")
    if not isinstance(answer, list) or not isinstance(output, list):
        return True
    if len(answer) != len(output):
        return True
    for a, o in zip(answer, output):
        if a != o:
            return True
    return False

def detect_dataset_type(obj):
    """自动检测数据集类型（备选方案）"""
    return "hrbench" if "answer" in obj else "vstar"

def detect_errors(filepath, dataset_type=None):
    errors = []
    used_type = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[警告] 第{line_num}行JSON解析失败: {e}", file=sys.stderr)
                continue

            if used_type is None:
                if dataset_type is not None:
                    used_type = dataset_type
                else:
                    used_type = detect_dataset_type(obj)
                    print(f"[信息] 自动检测数据集类型: {used_type}", file=sys.stderr)

            if used_type == "vstar":
                if is_error_vstar(obj):
                    errors.append((line_num, obj))
            elif used_type == "hrbench":
                if is_error_hrbench(obj):
                    errors.append((line_num, obj))
            else:
                raise ValueError(f"未知数据集类型: {used_type}")
    return errors, used_type

def main():
    parser = argparse.ArgumentParser(
        description="检测模型预测结果（JSONL）中的错误样本，支持vstar和HR-Bench数据集"
    )
    parser.add_argument("file", help="输入的JSONL文件路径")
    parser.add_argument("-d", "--dataset", choices=["vstar", "hrbench"],
                        help="显式指定数据集类型（若不指定则自动根据字段检测）")
    parser.add_argument("-o", "--output",
                        help="指定输出文件路径（若未指定，自动生成在输入文件同级目录；若为 '-' 则输出到标准输出）")
    args = parser.parse_args()

    errors, used_type = detect_errors(args.file, args.dataset)
    total = len(errors)
    print(f"[信息] 数据集类型: {used_type}，共发现 {total} 个错误样本", file=sys.stderr)

    # 确定输出目标
    if args.output is None:
        # 自动生成：输入文件同级目录，文件名加 _errors
        dirname = os.path.dirname(args.file) or '.'
        basename = os.path.basename(args.file)
        name, ext = os.path.splitext(basename)
        if ext == '':
            ext = '.jsonl'
        output_path = os.path.join(dirname, f"{name}_errors{ext}")
        print(f"[信息] 自动生成输出文件: {output_path}", file=sys.stderr)
        write_to_file = True
    elif args.output == '-':
        output_path = None
        write_to_file = False
    else:
        output_path = args.output
        write_to_file = True

    # 写入或打印
    if write_to_file:
        with open(output_path, 'w', encoding='utf-8') as f:
            for _, obj in errors:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        print(f"[信息] 错误样本已写入 {output_path}", file=sys.stderr)
    else:
        # 输出纯净JSONL到标准输出（每行一个JSON对象）
        for _, obj in errors:
            print(json.dumps(obj, ensure_ascii=False))

if __name__ == "__main__":
    main()
