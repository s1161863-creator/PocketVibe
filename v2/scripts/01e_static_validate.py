#!/usr/bin/env python3
"""
PocketVibe V2 — 01e: HTML 静态质量校验
=================================================================
对合并后的数据做严格的静态校验，过滤劣质样本
校验项：
  1. HTML 完整性（DOCTYPE + </html>）
  2. viewport meta 标签
  3. 代码长度（200~8000 字符）
  4. 无外部 CDN 引用
  5. 有 <style> 内联 CSS
  6. 有 <script> 内联 JS
  7. 无明显截断（不以 </html> 之外的标签结尾）

输出：data/processed/train_validated.jsonl + 过滤报告
=================================================================
运行：python scripts/01e_static_validate.py
"""
import json, os, re
from collections import Counter

INPUT  = "data/processed/merged_deduped.jsonl"
OUTPUT = "data/processed/train_validated.jsonl"
REJECT = "data/processed/rejected_samples.jsonl"


def validate(code: str) -> tuple[bool, str]:
    """
    返回 (是否通过, 失败原因)
    通过则返回 (True, "OK")
    """
    code_lower = code.lower().strip()

    # 1. 必须以 <!DOCTYPE html> 开头（允许空白）
    if not re.match(r'^\s*<!doctype\s+html', code_lower):
        return False, "不以<!DOCTYPE html>开头"

    # 2. 必须以 </html> 结尾（允许尾部空白）
    if not re.search(r'</html>\s*$', code_lower):
        return False, "缺少</html>闭合标签"

    # 3. viewport meta 标签
    if "viewport" not in code_lower:
        return False, "缺少viewport meta标签"

    # 4. 代码长度
    if len(code) < 200:
        return False, f"代码过短({len(code)}字符，最低200)"
    if len(code) > 10000:
        return False, f"代码过长({len(code)}字符，最高10000)"

    # 5. 外部 CDN 检查
    cdn_patterns = [
        r'cdn\.', r'unpkg\.com', r'jsdelivr', r'googleapis',
        r'cloudflare', r'bootstrapcdn', r'cdnjs\.',
    ]
    for pat in cdn_patterns:
        if re.search(pat, code_lower):
            return False, f"引用了外部CDN: {pat}"

    # 6. 必须有内联 CSS
    if '<style' not in code_lower:
        return False, "缺少<style>内联CSS标签"

    # 7. 必须有内联 JS
    if '<script' not in code_lower:
        return False, "缺少<script>内联JS标签"

    # 8. 基本结构检查：有 <head> 和 <body>
    if '<head' not in code_lower or '<body' not in code_lower:
        return False, "缺少<head>或<body>标签"

    return True, "OK"


def main():
    if not os.path.exists(INPUT):
        print(f"❌ 输入文件不存在：{INPUT}")
        print("   请先运行：python scripts/01d_merge_and_dedupe.py")
        return

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    total = 0
    passed = 0
    failed = 0
    fail_reasons = Counter()
    cat_stats = Counter()      # 每个类别通过多少条
    cat_fail_stats = Counter() # 每个类别失败多少条

    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout, \
         open(REJECT, "w", encoding="utf-8") as frej:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            code = item["messages"][-1]["content"]  # assistant 输出
            category = item.get("_category", "unknown")

            ok, reason = validate(code)

            if ok:
                # 写出时保留 _category 等元数据（训练时 DataCollator 会忽略未知字段）
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                passed += 1
                cat_stats[category] += 1
            else:
                failed += 1
                fail_reasons[reason] += 1
                cat_fail_stats[category] += 1
                item["_reject_reason"] = reason
                frej.write(json.dumps(item, ensure_ascii=False) + "\n")

    pass_rate = passed / total * 100 if total else 0

    print(f"\n{'='*50}")
    print(f"📊 静态校验结果")
    print(f"{'='*50}")
    print(f"  总条数：{total}")
    print(f"  通过：  {passed}  ({pass_rate:.1f}%)")
    print(f"  淘汰：  {failed}")

    if fail_reasons:
        print(f"\n❌ 淘汰原因分布：")
        for reason, cnt in fail_reasons.most_common():
            print(f"  {reason:40s}: {cnt}")

    print(f"\n✅ 通过类别分布：")
    for cat, cnt in sorted(cat_stats.items(), key=lambda x: -x[1]):
        fail_cnt = cat_fail_stats.get(cat, 0)
        total_cat = cnt + fail_cnt
        rate = cnt / total_cat * 100 if total_cat else 0
        print(f"  {cat:25s}: {cnt:4d} 通过 / {total_cat:4d} 总计 ({rate:.0f}%)")

    print(f"\n📄 输出文件：{OUTPUT}")
    print(f"📄 淘汰文件：{REJECT}")
    print(f"⏭️  下一步：python scripts/02_category_split.py")


if __name__ == "__main__":
    main()
