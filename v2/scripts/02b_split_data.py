#!/usr/bin/env python3
"""Split validated data into train/val/test while keeping identical HTML together."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "train_clean.jsonl"
TRAIN = ROOT / "data" / "processed" / "train.jsonl"
VAL = ROOT / "data" / "processed" / "val.jsonl"
TEST = ROOT / "data" / "processed" / "test.jsonl"
REPORT = ROOT / "data" / "processed" / "split_report.json"

SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05
random.seed(SEED)

CATEGORY_KEYWORDS = {
    "计时器/秒表": ["倒计时", "秒表", "计时", "timer"],
    "随机/骰子": ["骰子", "硬币", "随机", "抽签", "抽奖", "转盘"],
    "计算器": ["计算器", "计算", "进制", "百分比"],
    "换算工具": ["换算", "互转", "单位", "温度", "长度", "重量", "汇率"],
    "记分/投票": ["记分", "投票", "评分"],
    "猜数/游戏": ["猜数", "剪刀布", "游戏", "翻牌", "井字棋", "扫雷", "打地鼠", "贪吃蛇", "弹球", "华容道"],
    "番茄钟": ["番茄", "专注"],
    "待办/清单": ["待办", "清单", "打卡", "习惯", "打包"],
    "记账/理财": ["记账", "账", "存钱", "预算", "收入", "支出", "房贷", "利息", "股票", "工资"],
    "日历/日期": ["日历", "纪念日", "倒数日", "日期", "考试倒计时", "月龄", "孕周"],
    "画板/创意": ["画板", "涂色", "颜色", "画画", "Markdown"],
    "记录/日记": ["备忘", "日记", "记录", "心情", "喝水", "喂奶", "血压", "体重"],
    "音频/节拍": ["节拍", "白噪音", "音乐", "声音"],
    "打字/测试": ["打字", "反应", "视力", "速度测试", "专注力"],
    "教育/学习": ["乘法", "单词", "GPA", "课程表", "古诗", "口算", "元素周期表"],
    "生活工具": ["密码", "字数", "年龄", "小费", "AA", "面积", "时间间隔", "电费", "快递", "垃圾分类", "穿衣", "食谱"],
    "社交/娱乐": ["真心话", "卧底", "表白", "星座", "生肖", "塔罗", "情话"],
    "育儿": ["宝宝", "儿童", "小朋友", "故事计时"],
    "团建/分组": ["分组", "座位", "红包分配", "团建"],
}


def classify_instruction(instruction: str) -> tuple[str, str]:
    lowered = instruction.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            if category in {"计时器/秒表", "随机/骰子", "计算器", "换算工具", "记分/投票"}:
                return category, "L1_简单"
            if category in {"猜数/游戏", "音频/节拍", "打字/测试"}:
                return category, "L3_挑战"
            return category, "L2_中等"
    return "其他", "L1_简单"


def code_hash(code: str) -> str:
    normalized = re.sub(r"\s+", "", code.lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def build_groups(data: list[dict]) -> list[dict]:
    buckets: defaultdict[str, list[dict]] = defaultdict(list)
    for item in data:
        buckets[code_hash(item["messages"][-1]["content"])].append(item)

    groups: list[dict] = []
    for hash_value, items in buckets.items():
        category, level = classify_instruction(items[0]["messages"][1]["content"])
        groups.append({"code_hash": hash_value, "items": items, "category": category, "level": level})
    return groups


def stratified_group_split(groups: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped_by_category: defaultdict[str, list[dict]] = defaultdict(list)
    for group in groups:
        grouped_by_category[group["category"]].append(group)

    train_groups: list[dict] = []
    val_groups: list[dict] = []
    test_groups: list[dict] = []

    for category, category_groups in grouped_by_category.items():
        random.shuffle(category_groups)
        n = len(category_groups)
        n_test = max(1, round(n * TEST_RATIO)) if n >= 3 else 0
        n_val = max(1, round(n * VAL_RATIO)) if n >= 2 else 0
        if n_test + n_val >= n:
            n_test = 1 if n >= 3 else 0
            n_val = 1 if n >= 2 else 0
        test_groups.extend(category_groups[:n_test])
        val_groups.extend(category_groups[n_test:n_test + n_val])
        train_groups.extend(category_groups[n_test + n_val :])
        if not train_groups and category_groups:
            train_groups.append(category_groups[-1])
        _ = category
    return train_groups, val_groups, test_groups


def flatten(groups: list[dict]) -> list[dict]:
    items: list[dict] = []
    for group in groups:
        items.extend(group["items"])
    random.shuffle(items)
    return items


def count_categories(subset: list[dict]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in subset:
        category, _level = classify_instruction(item["messages"][1]["content"])
        counter[category] += 1
    return dict(counter.most_common())


def leakage_check(train: list[dict], other: list[dict]) -> int:
    train_hashes = {code_hash(item["messages"][-1]["content"]) for item in train}
    return sum(1 for item in other if code_hash(item["messages"][-1]["content"]) in train_hashes)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    with INPUT.open("r", encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh]

    groups = build_groups(data)
    train_groups, val_groups, test_groups = stratified_group_split(groups)
    train = flatten(train_groups)
    val = flatten(val_groups)
    test = flatten(test_groups)

    write_jsonl(TRAIN, train)
    write_jsonl(VAL, val)
    write_jsonl(TEST, test)

    levels = [classify_instruction(item["messages"][1]["content"])[1] for item in data]
    report = {
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "total": len(data),
        },
        "split_ratios": {
            "train": f"{(len(train) / len(data) * 100) if data else 0:.1f}%",
            "val": f"{(len(val) / len(data) * 100) if data else 0:.1f}%",
            "test": f"{(len(test) / len(data) * 100) if data else 0:.1f}%",
        },
        "group_counts": {
            "total_unique_html_groups": len(groups),
            "train_html_groups": len(train_groups),
            "val_html_groups": len(val_groups),
            "test_html_groups": len(test_groups),
        },
        "category_distribution": {
            "train": count_categories(train),
            "val": count_categories(val),
            "test": count_categories(test),
        },
        "complexity_distribution": {
            "L1_简单": sum(1 for level in levels if level == "L1_简单"),
            "L2_中等": sum(1 for level in levels if level == "L2_中等"),
            "L3_挑战": sum(1 for level in levels if level == "L3_挑战"),
        },
        "leakage_check": {
            "val_code_overlap_with_train": leakage_check(train, val),
            "test_code_overlap_with_train": leakage_check(train, test),
            "note": "按相同HTML代码分组后再切分，理论上应为 0。",
        },
        "random_seed": SEED,
    }
    with REPORT.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("=" * 56)
    print("PocketVibe 数据集切分报告")
    print("=" * 56)
    print(f"训练集: {len(train)} 条")
    print(f"验证集: {len(val)} 条")
    print(f"测试集: {len(test)} 条")
    print(f"唯一 HTML 分组: {len(groups)}")
    print(f"验证集代码与训练集重叠: {report['leakage_check']['val_code_overlap_with_train']}")
    print(f"测试集代码与训练集重叠: {report['leakage_check']['test_code_overlap_with_train']}")
    print(f"输出文件: {TRAIN}, {VAL}, {TEST}")
    print(f"划分报告: {REPORT}")


if __name__ == "__main__":
    main()
