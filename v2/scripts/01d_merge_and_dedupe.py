#!/usr/bin/env python3
"""
PocketVibe V2 — 01d: 合并去重 + 打类别标签
=================================================================
合并三个来源的数据，去除重复指令，为每条数据打上规范化类别标签
输出：data/processed/merged_deduped.jsonl
=================================================================
运行：python scripts/01d_merge_and_dedupe.py
"""
import json, os, re

FILES = [
    ("data/seed/seed_examples.jsonl",          "seed"),
    ("data/processed/multi_impl.jsonl",         "multi_impl"),
    ("data/processed/evol_tools.jsonl",         "evol_tools"),
    ("data/processed/cross_cat.jsonl",          "cross_cat"),
]
OUTPUT = "data/processed/merged_deduped.jsonl"

# cross_category 子类型 → 归入主类别（取第一个）
CROSS_TO_CATEGORY = {
    "timer_game":              "timer",
    "calculator_game":         "calculator",
    "timer_tracker":           "timer",
    "converter_lifestyle":     "converter",
    "tracker_health":          "tracker",
    "game_social":             "game",
    "education_game":          "education",
    "health_timer":            "health",
    "lifestyle_calculator":    "lifestyle",
    "entertainment_tracker":   "entertainment",
    "finance_lifestyle":       "finance",
    "parenting_game":          "parenting",
}

def normalize_instruction(text: str) -> str:
    """规范化指令文本用于去重"""
    text = text.strip().lower()
    # 去掉标点、空格、语气词
    text = re.sub(r'[，。！？、,.!? \t\n帮我做个帮我做一个帮我弄个做一个做个弄个来个搞个]', '', text)
    return text


def dedup_key(item: dict) -> str:
    """去重 key：(规范化指令, 风格/实现变体)
       multi_impl 的 4 种风格应视为不同样本，不能只按指令去重
    """
    instr = item["messages"][1]["content"]
    norm = normalize_instruction(instr)
    # multi_impl 用 _style 区分（如 simple-light / bold-dark / retro / glassmorphism）
    style = item.get("_style", "")
    source_key = item.get("_source_key", "")
    # cross_cat 用 _cross_type 区分
    cross = item.get("_cross_type", "")
    return f"{norm}|||{style}|||{source_key}|||{cross}"


def get_category(item: dict) -> str:
    """从 item 的 metadata 字段推断类别"""
    # evol_tools 和 multi_impl 直接有 _category
    cat = item.get("_category", "")
    if cat and cat not in ("cross_category", "seed_multi_impl", ""):
        return cat
    # cross_category 看 _cross_type
    cross = item.get("_cross_type", "")
    if cross:
        return CROSS_TO_CATEGORY.get(cross, "cross_category")
    # seed：通过指令关键词简单分类
    instr = item["messages"][1]["content"].lower() if len(item["messages"]) > 1 else ""
    if any(k in instr for k in ["计时", "秒表", "倒计时", "番茄", "计数器"]):
        return "timer"
    if any(k in instr for k in ["计算器", "bmi", "小费", "折扣", "分摊", "面积"]):
        return "calculator"
    if any(k in instr for k in ["换算", "转换", "互转", "进制"]):
        return "converter"
    if any(k in instr for k in ["游戏", "猜", "石头", "剪刀", "打地鼠", "棋"]):
        return "game"
    if any(k in instr for k in ["记录", "待办", "打卡", "清单", "日历"]):
        return "tracker"
    if any(k in instr for k in ["健康", "卡路里", "体脂", "血压", "睡眠", "呼吸"]):
        return "health"
    if any(k in instr for k in ["生活", "吃什么", "倒数日", "备忘", "天气"]):
        return "lifestyle"
    if any(k in instr for k in ["学习", "单词", "成绩", "gpa", "口算", "古诗", "乘法"]):
        return "education"
    if any(k in instr for k in ["随机", "密码", "颜色", "抛硬币", "星座", "塔罗"]):
        return "entertainment"
    if any(k in instr for k in ["画板", "日记", "节拍", "评分"]):
        return "creative"
    if any(k in instr for k in ["投票", "记分板", "分组", "抽奖", "红包"]):
        return "social"
    if any(k in instr for k in ["房贷", "复利", "存钱", "理财", "税"]):
        return "finance"
    if any(k in instr for k in ["宝宝", "儿童", "孩子", "月龄", "育儿"]):
        return "parenting"
    if any(k in instr for k in ["日程", "课程表", "规划", "周计划"]):
        return "planner"
    if any(k in instr for k in ["文本", "markdown", "编码", "正则"]):
        return "utility"
    return "other"


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    seen_norm = set()   # 去重用（规范化指令）
    all_items = []
    source_stats = {}

    for filepath, source_tag in FILES:
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在，跳过：{filepath}")
            continue
        count_in = 0
        count_kept = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                count_in += 1
                key = dedup_key(item)
                if key in seen_norm:
                    continue
                seen_norm.add(key)
                # 补充/覆盖元数据
                item["_source"] = source_tag
                if "_category" not in item or not item["_category"]:
                    item["_category"] = get_category(item)
                else:
                    # 重新规范化 cross_category
                    if item["_category"] == "cross_category":
                        item["_category"] = get_category(item)
                    elif item["_category"] == "seed_multi_impl":
                        item["_category"] = get_category(item)

                all_items.append(item)
                count_kept += 1

        source_stats[source_tag] = {"in": count_in, "kept": count_kept}
        print(f"  {source_tag}: 读入 {count_in} → 保留 {count_kept} 条（去重后）")

    # 写出
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for item in all_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 类别统计
    from collections import Counter
    cat_counts = Counter(item["_category"] for item in all_items)
    print(f"\n📊 类别分布（共 {len(all_items)} 条）：")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {cnt:4d} 条")

    print(f"\n✅ 01d 完成！总计 {len(all_items)} 条唯一数据")
    print(f"📄 输出：{OUTPUT}")
    print(f"⏭️  下一步：python scripts/01e_static_validate.py")


if __name__ == "__main__":
    main()
