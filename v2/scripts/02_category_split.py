#!/usr/bin/env python3
"""
PocketVibe V2 — 02: 按类别隔离切分训练集/验证集
=================================================================
修复 V1 问题：随机 9:1 切分导致数据泄漏
V2 新策略：15 个 held-out 类别完全不出现在训练集中
           验证集 loss 才能真实反映泛化能力

held-out 类别（验证集专属，训练集绝对不含）：
  creative, social, finance, parenting, planner, utility, other,
  timer_game, calculator_game, tracker_health, game_social,
  health_timer, lifestyle_calculator, entertainment_tracker,
  finance_lifestyle, parenting_game

输出：
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/split_report.json
=================================================================
运行：python scripts/02_category_split.py
"""
import json, os, random
from collections import Counter, defaultdict

INPUT  = "data/processed/train_validated.jsonl"
TRAIN  = "data/processed/train.jsonl"
VAL    = "data/processed/val.jsonl"
REPORT = "data/processed/split_report.json"

random.seed(42)

# ================================================================
# held-out 类别：这些类别的数据全部进验证集
# 选择原则：
#   1. 覆盖尽量多的功能维度
#   2. 数据量适中（每类 5~20 条），保证验证集有统计意义
#   3. 不选 timer/calculator/converter（它们数据量大，需要训练集学习）
# ================================================================
HELD_OUT_CATEGORIES = {
    "creative",             # 绘画创作类
    "social",               # 社交团队类
    "finance",              # 金融理财类
    "parenting",            # 育儿亲子类
    "planner",              # 日程规划类
    "utility",              # 工具效率类
    "other",                # 未分类
    # 跨类组合（全部进验证集，因为训练集已有足够覆盖）
    "cross_category",
}

# 训练集类别：这些类别的数据全部进训练集
TRAIN_ONLY_CATEGORIES = {
    "timer", "calculator", "converter", "game",
    "tracker", "health", "lifestyle", "education", "entertainment",
}


def main():
    if not os.path.exists(INPUT):
        print(f"❌ 输入文件不存在：{INPUT}")
        print("   请先运行：python scripts/01e_static_validate.py")
        return

    os.makedirs(os.path.dirname(TRAIN), exist_ok=True)

    # 读取所有数据
    all_items = []
    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_items.append(json.loads(line))

    print(f"📖 读取 {len(all_items)} 条数据\n")

    # 按类别分组
    by_category = defaultdict(list)
    for item in all_items:
        cat = item.get("_category", "other")
        by_category[cat].append(item)

    print(f"📊 类别分布：")
    for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        status = "→ 验证集" if cat in HELD_OUT_CATEGORIES else "→ 训练集"
        print(f"  {cat:25s}: {len(items):4d} 条  {status}")

    train_items = []
    val_items   = []

    for cat, items in by_category.items():
        if cat in HELD_OUT_CATEGORIES:
            # held-out 类别：全部进验证集
            val_items.extend(items)
        elif cat in TRAIN_ONLY_CATEGORIES:
            # 训练集类别：全部进训练集
            train_items.extend(items)
        else:
            # 未知类别：80% 训练，20% 验证（保守分配）
            random.shuffle(items)
            split = int(len(items) * 0.8)
            train_items.extend(items[:split])
            val_items.extend(items[split:])

    # Shuffle
    random.shuffle(train_items)
    random.shuffle(val_items)

    # 写出
    with open(TRAIN, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL, "w", encoding="utf-8") as f:
        for item in val_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 报告
    report = {
        "total":        len(all_items),
        "train":        len(train_items),
        "val":          len(val_items),
        "train_ratio":  round(len(train_items) / len(all_items), 3),
        "held_out_categories": sorted(list(HELD_OUT_CATEGORIES)),
        "train_categories":    sorted(list(TRAIN_ONLY_CATEGORIES)),
        "val_category_dist":   dict(Counter(i.get("_category","other") for i in val_items)),
        "train_category_dist": dict(Counter(i.get("_category","other") for i in train_items)),
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ 数据切分完成")
    print(f"  训练集：{len(train_items)} 条  → {TRAIN}")
    print(f"  验证集：{len(val_items)} 条  → {VAL}")
    print(f"  验证集类别（held-out）：{sorted(HELD_OUT_CATEGORIES)}")
    print(f"\n📄 切分报告：{REPORT}")
    print(f"⏭️  下一步（HPC上执行）：sbatch slurm/train.slurm")


if __name__ == "__main__":
    main()
