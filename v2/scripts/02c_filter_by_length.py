#!/usr/bin/env python3
"""
=================================================================
PocketVibe V2（重做）— 训练集按 token 长度预过滤
=================================================================
用途：
  用真实 Qwen2.5-Coder tokenizer 精确测量 data/processed/train.jsonl 每条
  样本的 token 数，剔除 > MAX_SEQ_LEN 的样本（这些样本在训练时一定会被
  截断），输出 train_filtered.jsonl。训练脚本 03_train_qlora_v2.py 会
  优先使用这个文件，从而保证训练阶段 0 截断。

背景：
  V2 首轮训练最严重的 bug —— max_seq_length=1024，但 train.jsonl 中 100% 样本
  > 1024 tokens，导致整个训练集都被截尾，模型从未见过完整的 </html> 闭合。

V2 重做在这里做了两件事：
  1. 训练脚本将 max_seq_length 提升到 4096（本脚本的阈值）
  2. 本脚本再预过滤 token 数 > 4096 的个别超长样本（通常占 < 5%）

运行：
  python scripts/02c_filter_by_length.py
  # 或自定义阈值
  python scripts/02c_filter_by_length.py --max-len 4096

输出：
  data/processed/train_filtered.jsonl   （供训练脚本优先加载）
  data/processed/val_filtered.jsonl     （顺带把验证集也过滤了，防止评估时截断）
  控制台打印 token 长度分布 + 剔除样本列表
=================================================================
"""
import os
import json
import hashlib
import argparse
from collections import Counter, defaultdict

from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

HERE       = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.path.dirname(HERE)
DATA_DIR   = os.path.join(PROJECT, "data", "processed")


def measure_token_len(tokenizer, messages) -> int:
    """用 Qwen 官方 chat template 渲染后测量真实 token 数。
    与训练脚本完全一致，保证测量值 == 训练时实际长度。
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def histogram(token_lens, buckets=(512, 1024, 1536, 2048, 2560, 3072, 3584,
                                   4096, 5120, 6144, 8192)):
    """按桶打印 token 长度分布"""
    cnt = Counter()
    for n in token_lens:
        for b in buckets:
            if n <= b:
                cnt[b] += 1
                break
        else:
            cnt[buckets[-1] + 1] += 1

    total = len(token_lens)
    prev = 0
    for b in buckets:
        c = cnt[b]
        if c == 0:
            prev = b
            continue
        pct = c / total * 100
        bar = "█" * int(pct / 2)
        print(f"   ({prev:>5} - {b:>5}] tokens  | {c:>4} 条 ({pct:5.1f}%) {bar}")
        prev = b
    overflow = cnt[buckets[-1] + 1]
    if overflow:
        print(f"   (    > {buckets[-1]:>5}] tokens  | {overflow:>4} 条 "
              f"({overflow/total*100:5.1f}%) ← 这些会被截断")


def analyze_instruction_collapse(items):
    """统计"指令坍缩"：同一份 HTML 对应多少条不同 instruction。

    背景（对应用户担忧 #5）：
      V1 阶段 A 的伪增强做法 "50 种子 × 10 指令变体 = 500 条"，导致同一份
      assistant HTML 被多条不同 user instruction 对应 → 教模型"无论怎么问都
      输出这份 HTML"，是"记忆屏障"的根源。
      V2 改用 Evol-Instruct + 多实现扩充后应已解决，本函数用于验证。

    健康信号：95%+ 的 HTML 只对应 1 条 instruction
    警告信号：若 >5% 的 HTML 对应 ≥3 条不同 instruction → 仍存在指令坍缩
    """
    html2insts = defaultdict(set)
    for item in items:
        messages = item["messages"]
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        asst = next((m["content"] for m in messages if m.get("role") == "assistant"), "")
        # 用 HTML 的 sha1 前 16 字符作为 key，节省内存
        html_key = hashlib.sha1(asst.encode("utf-8")).hexdigest()[:16]
        html2insts[html_key].add(user)

    total_html = len(html2insts)
    if total_html == 0:
        return

    # 按 "一份 HTML 对应多少条不同 instruction" 分桶
    buckets = Counter()
    for insts in html2insts.values():
        buckets[len(insts)] += 1

    max_n = max(buckets.keys())
    multi = sum(c for n, c in buckets.items() if n >= 2)
    heavy = sum(c for n, c in buckets.items() if n >= 3)

    print(f"\n🔎 指令多样性检查（检测 V1 式伪增强/指令坍缩）")
    print(f"   唯一 HTML 数：{total_html}")
    print(f"   1 对 1 配对：{buckets[1]:>4} 份 ({buckets[1]/total_html*100:5.1f}%)  ✅ 健康")
    if multi:
        print(f"   1 对 ≥2  ：{multi:>4} 份 ({multi/total_html*100:5.1f}%)   ⚠ 多指令共享同一 HTML")
    if heavy:
        print(f"   1 对 ≥3  ：{heavy:>4} 份 ({heavy/total_html*100:5.1f}%)   ❌ 强烈指令坍缩信号")

    # 列出"最严重"的 top-5
    worst = sorted(html2insts.items(), key=lambda kv: -len(kv[1]))[:5]
    if worst[0][1] and len(worst[0][1]) >= 2:
        print(f"\n   Top-5 重复最多的 HTML：")
        for html_key, insts in worst:
            if len(insts) < 2:
                break
            preview = list(insts)[0][:45]
            print(f"     [{html_key}] {len(insts):>2} 条不同指令 | 例: {preview}...")

    # 总体健康评级
    if heavy / total_html > 0.05:
        print(f"\n   ⚠⚠ 健康评级：不通过 — ≥3 条指令共享同一 HTML 的比例 > 5%")
        print(f"       建议：检查 01a/01b 数据生成脚本，或考虑按 HTML 哈希去重")
    elif multi / total_html > 0.10:
        print(f"\n   ⚠ 健康评级：轻度异常 — 多指令共享 HTML 的比例 > 10%")
    else:
        print(f"\n   ✅ 健康评级：通过 — 指令与 HTML 基本 1:1 配对")


def filter_file(tokenizer, src_path: str, dst_path: str, max_len: int, label: str):
    """过滤单个 jsonl 文件"""
    if not os.path.exists(src_path):
        print(f"⚠ 文件不存在，跳过: {src_path}")
        return

    print(f"\n{'='*70}")
    print(f"处理 [{label}]: {src_path}")
    print(f"{'='*70}")

    keep = []
    drop = []
    token_lens = []
    all_items = []   # 用于指令坍缩分析

    with open(src_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if not messages:
                continue

            all_items.append(item)

            n = measure_token_len(tokenizer, messages)
            token_lens.append(n)

            if n <= max_len:
                keep.append((line, n, item))
            else:
                user_inst = next((m["content"] for m in messages
                                  if m.get("role") == "user"), "")
                drop.append((line_no, n, user_inst[:50]))

    total = len(token_lens)
    if total == 0:
        print(f"   ✗ 文件为空")
        return

    token_lens.sort()
    avg = sum(token_lens) / total
    p50 = token_lens[total // 2]
    p90 = token_lens[int(total * 0.9)]
    p95 = token_lens[int(total * 0.95)]
    p99 = token_lens[int(total * 0.99)]
    mx  = token_lens[-1]
    mn  = token_lens[0]

    print(f"\n📊 Token 长度统计（{total} 条样本）")
    print(f"   min={mn}  avg={avg:.0f}  median={p50}  p90={p90}  "
          f"p95={p95}  p99={p99}  max={mx}")
    print(f"\n📈 长度分布：")
    histogram(token_lens)

    print(f"\n🔍 过滤阈值: max_len = {max_len}")
    print(f"   保留: {len(keep):>4} 条 ({len(keep)/total*100:5.1f}%)")
    print(f"   剔除: {len(drop):>4} 条 ({len(drop)/total*100:5.1f}%)")

    if drop:
        print(f"\n⚠ 被剔除的超长样本（共 {len(drop)} 条）：")
        for line_no, n, preview in drop[:20]:
            print(f"   行 {line_no:>4}: {n:>5} tokens | {preview}...")
        if len(drop) > 20:
            print(f"   ... 另有 {len(drop) - 20} 条未列出")

    # ★ V2 重做新增：指令坍缩检测（只对训练集做，验证集没必要）
    if "train" in label.lower():
        analyze_instruction_collapse(all_items)

    # 写出过滤后的文件
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as fout:
        for line, _, _ in keep:
            fout.write(line + "\n")

    print(f"\n✅ 输出: {dst_path} ({len(keep)} 条)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=4096,
                        help="最大 token 长度阈值，超过的样本会被剔除（默认 4096）")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"数据目录（默认 {DATA_DIR}）")
    args = parser.parse_args()

    print(f"🔧 加载 tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n🎯 过滤阈值 max_len = {args.max_len}")
    print(f"   说明：> {args.max_len} tokens 的样本会被剔除，")
    print(f"         保证训练脚本（max_seq_length={args.max_len}）下 0 截断。")

    # 训练集
    filter_file(
        tokenizer,
        src_path=os.path.join(args.data_dir, "train.jsonl"),
        dst_path=os.path.join(args.data_dir, "train_filtered.jsonl"),
        max_len=args.max_len,
        label="train.jsonl",
    )

    # 验证集（顺带过滤，防止 eval 阶段截断导致 eval_loss 虚高）
    filter_file(
        tokenizer,
        src_path=os.path.join(args.data_dir, "val.jsonl"),
        dst_path=os.path.join(args.data_dir, "val_filtered.jsonl"),
        max_len=args.max_len,
        label="val.jsonl",
    )

    print(f"\n{'='*70}")
    print(f"✅ 全部完成")
    print(f"{'='*70}")
    print(f"下一步:")
    print(f"  1. 检查上方剔除列表，若被剔除的样本过多（> 10%）可考虑提高 --max-len")
    print(f"  2. 上传到 HPC: scp data/processed/*.jsonl <user>@hpc:~/PocketVibe/data/processed/")
    print(f"  3. 提交训练: sbatch slurm/train.slurm")
    print(f"     （训练脚本会自动优先使用 train_filtered.jsonl）")


if __name__ == "__main__":
    main()
