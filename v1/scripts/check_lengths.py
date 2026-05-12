#!/usr/bin/env python3
"""检查训练数据的字符长度和token长度分布"""
import json, os

os.chdir(os.path.join(os.path.dirname(__file__), ".."))

for fname in ["data/processed/train.jsonl", "data/processed/val.jsonl"]:
    if not os.path.exists(fname):
        print(f"文件不存在: {fname}")
        continue
    
    lines = [json.loads(l) for l in open(fname, "r", encoding="utf-8")]
    
    # 计算assistant回复（HTML代码）的字符长度
    html_lengths = [len(m["messages"][2]["content"]) for m in lines]
    # 计算整条数据的总字符长度（system+user+assistant）
    total_lengths = [
        len(m["messages"][0]["content"]) + len(m["messages"][1]["content"]) + len(m["messages"][2]["content"])
        for m in lines
    ]
    
    print(f"\n===== {fname} ({len(lines)} 条) =====")
    print(f"HTML代码(assistant)字符长度:")
    print(f"  最小: {min(html_lengths)}, 最大: {max(html_lengths)}, 平均: {sum(html_lengths)//len(html_lengths)}")
    print(f"  >2000字符: {sum(1 for l in html_lengths if l>2000)} 条")
    print(f"  >3000字符: {sum(1 for l in html_lengths if l>3000)} 条")
    print(f"  >5000字符: {sum(1 for l in html_lengths if l>5000)} 条")
    
    print(f"总长度(system+user+assistant)字符:")
    print(f"  最小: {min(total_lengths)}, 最大: {max(total_lengths)}, 平均: {sum(total_lengths)//len(total_lengths)}")
    print(f"  >3000字符: {sum(1 for l in total_lengths if l>3000)} 条")
    print(f"  >4000字符: {sum(1 for l in total_lengths if l>4000)} 条")
    print(f"  >5000字符: {sum(1 for l in total_lengths if l>5000)} 条")

# 粗略估算token数（中文约1.5字符/token，代码约3-4字符/token，取平均约2.5）
print("\n===== 粗略token估算 =====")
for fname in ["data/processed/train.jsonl"]:
    lines = [json.loads(l) for l in open(fname, "r", encoding="utf-8")]
    total_lengths = [
        len(m["messages"][0]["content"]) + len(m["messages"][1]["content"]) + len(m["messages"][2]["content"])
        for m in lines
    ]
    # 加上chat template的overhead（约200-300字符）
    estimated_tokens = [(l + 250) / 2.5 for l in total_lengths]
    print(f"训练集预估token长度:")
    print(f"  最小: {int(min(estimated_tokens))}, 最大: {int(max(estimated_tokens))}, 平均: {int(sum(estimated_tokens)//len(estimated_tokens))}")
    print(f"  >1024 tokens: {sum(1 for t in estimated_tokens if t>1024)} 条 ({sum(1 for t in estimated_tokens if t>1024)*100//len(estimated_tokens)}%)")
    print(f"  >2048 tokens: {sum(1 for t in estimated_tokens if t>2048)} 条")
    print(f"\n⚠️ 训练时 max_seq_length=1024，超过1024 tokens的数据在训练时会被截断！")
