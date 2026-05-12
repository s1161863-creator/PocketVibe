#!/usr/bin/env python3
"""Validate generated HTML training data and produce a quality report."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "train_generated.jsonl"
OUTPUT = ROOT / "data" / "processed" / "train_clean.jsonl"
REJECT = ROOT / "data" / "processed" / "rejected_samples.jsonl"
REPORT = ROOT / "data" / "processed" / "quality_report.json"
MIN_QUALITY_SCORE = 60


def check_structure(code: str) -> tuple[bool, list[str]]:
    issues: list[str] = []
    text = code.strip()
    lowered = text.lower()
    if not lowered.startswith("<!doctype html"):
        issues.append("缺少<!DOCTYPE html>声明")
    for tag in ("</html>", "</head>", "</body>"):
        if tag not in lowered:
            issues.append(f"缺少{tag}闭合标签")
    if "<style" not in lowered:
        issues.append("缺少<style>标签")
    if "<script" not in lowered:
        issues.append("缺少<script>标签")
    banned_patterns = [
        r"cdn\.",
        r"unpkg\.com",
        r"jsdelivr",
        r"googleapis\.com",
        r"cdnjs",
        r"<link\s+[^>]*href\s*=\s*[\"']https?://",
        r"<script\s+[^>]*src\s*=\s*[\"']https?://",
    ]
    for pattern in banned_patterns:
        if re.search(pattern, lowered):
            issues.append("引用了外部资源")
            break
    if lowered.count("<!doctype") > 1:
        issues.append("输出了多个HTML文件")
    return not issues, issues


def check_mobile(code: str) -> tuple[int, list[str]]:
    issues: list[str] = []
    lowered = code.lower()
    score = 0
    if "viewport" in lowered and "width=device-width" in lowered:
        score += 10
    else:
        issues.append("缺少viewport meta标签")

    responsive_hits = sum(
        1 for token in ("max-width", "100%", "100vw", "flex", "grid", "media", "vw", "vh") if token in lowered
    )
    if responsive_hits >= 2:
        score += 10
    elif responsive_hits == 1:
        score += 5
        issues.append("响应式信号不足")
    else:
        issues.append("缺少响应式布局")

    padding_matches = [int(item) for item in re.findall(r"padding\s*:\s*(\d+)", lowered)]
    if padding_matches:
        max_padding = max(padding_matches)
        if max_padding >= 10:
            score += 10
        elif max_padding >= 6:
            score += 5
            issues.append(f"按钮padding偏小({max_padding}px)")
        else:
            issues.append(f"按钮padding过小({max_padding}px)")
    else:
        score += 5
    return score, issues


def check_visual(code: str) -> tuple[int, list[str]]:
    issues: list[str] = []
    lowered = code.lower()
    score = 0
    if "linear-gradient" in lowered or "radial-gradient" in lowered:
        score += 8
    else:
        issues.append("缺少渐变背景")

    radii = [int(item) for item in re.findall(r"border-radius\s*:\s*(\d+)", lowered)]
    if radii:
        max_radius = max(radii)
        if max_radius >= 10:
            score += 7
        elif max_radius >= 4:
            score += 4
        else:
            issues.append("圆角过小")
    else:
        issues.append("缺少圆角样式")

    if "box-shadow" in lowered:
        score += 5
    else:
        issues.append("缺少阴影效果")

    if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", code):
        score += 5
    else:
        issues.append("标题缺少emoji")

    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", code))
    if chinese_count >= 3:
        score += 5
    elif chinese_count >= 1:
        score += 3
    else:
        issues.append("界面缺少中文")

    return score, issues


def check_logic(code: str) -> tuple[int, list[str]]:
    issues: list[str] = []
    score = 0
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", code, re.IGNORECASE | re.DOTALL)
    js = "\n".join(scripts)
    if not js.strip():
        return 0, ["缺少JavaScript逻辑"]

    if len(js.strip()) > 50:
        score += 5
    else:
        issues.append("JS代码过短")

    if re.search(r"function\s+\w+|=>\s*{|const\s+\w+\s*=\s*\(", js):
        score += 5
    else:
        issues.append("缺少函数定义")

    dom_hits = sum(
        1
        for token in ("getElementById", "querySelector", "innerHTML", "textContent", "addEventListener", "onclick", "style.")
        if token in js
    )
    if dom_hits >= 2:
        score += 5
    elif dom_hits == 1:
        score += 3
        issues.append("DOM交互偏少")
    else:
        issues.append("缺少DOM交互")

    event_hits = sum(
        1
        for token in ("onclick", "oninput", "onchange", "addEventListener", "setInterval", "setTimeout")
        if token in js or token in code
    )
    if event_hits >= 1:
        score += 5
    else:
        issues.append("缺少事件处理")

    return score, issues


def check_size(code: str) -> tuple[int, list[str]]:
    length = len(code)
    if length < 300:
        return 2, [f"代码过短({length}字符)"]
    if length < 500:
        return 10, [f"代码偏短({length}字符)"]
    if length > 8000:
        return 5, [f"代码过长({length}字符)"]
    if length > 5000:
        return 12, [f"代码偏长({length}字符)"]
    return 20, []


def compute_hash(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def validate_one(item: dict) -> dict:
    instruction = item["messages"][1]["content"]
    code = item["messages"][-1]["content"]
    structure_ok, structure_issues = check_structure(code)
    result = {
        "instruction": instruction,
        "instruction_hash": hashlib.md5(instruction.encode("utf-8")).hexdigest(),
        "code_hash": compute_hash(code),
        "code_length": len(code),
        "layers": {},
        "all_issues": list(structure_issues),
        "total_score": 0,
        "passed": False,
    }
    result["layers"]["structure"] = {"passed": structure_ok, "issues": structure_issues}
    if not structure_ok:
        return result

    mobile_score, mobile_issues = check_mobile(code)
    visual_score, visual_issues = check_visual(code)
    logic_score, logic_issues = check_logic(code)
    size_score, size_issues = check_size(code)

    result["layers"]["mobile"] = {"score": mobile_score, "max": 30, "issues": mobile_issues}
    result["layers"]["visual"] = {"score": visual_score, "max": 30, "issues": visual_issues}
    result["layers"]["logic"] = {"score": logic_score, "max": 20, "issues": logic_issues}
    result["layers"]["size"] = {"score": size_score, "max": 20, "issues": size_issues}
    result["all_issues"].extend(mobile_issues + visual_issues + logic_issues + size_issues)
    result["total_score"] = mobile_score + visual_score + logic_score + size_score
    result["passed"] = result["total_score"] >= MIN_QUALITY_SCORE
    return result


def main() -> None:
    with INPUT.open("r", encoding="utf-8") as fh:
        data = [json.loads(line) for line in fh]

    raw_results = [(item, validate_one(item)) for item in data]
    seen_instruction_hashes: set[str] = set()
    deduped_results: list[tuple[dict, dict]] = []
    duplicate_count = 0

    for item, result in raw_results:
        if result["instruction_hash"] in seen_instruction_hashes:
            result["passed"] = False
            result["all_issues"].append("指令重复")
            duplicate_count += 1
        else:
            seen_instruction_hashes.add(result["instruction_hash"])
        deduped_results.append((item, result))

    passed_items: list[dict] = []
    rejected_items: list[dict] = []
    score_dist: Counter[str] = Counter()
    issue_freq: Counter[str] = Counter()
    layer_scores: defaultdict[str, list[int]] = defaultdict(list)

    for item, result in deduped_results:
        bucket_floor = (result["total_score"] // 10) * 10
        score_dist[f"{bucket_floor}-{bucket_floor + 9}"] += 1
        for issue in result["all_issues"]:
            issue_freq[issue] += 1
        for layer_name, layer_data in result["layers"].items():
            if "score" in layer_data:
                layer_scores[layer_name].append(layer_data["score"])
        if result["passed"]:
            passed_items.append(item)
        else:
            rejected_items.append(
                {
                    "instruction": result["instruction"],
                    "score": result["total_score"],
                    "issues": result["all_issues"],
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for item in passed_items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    with REJECT.open("w", encoding="utf-8") as fh:
        for item in rejected_items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    max_scores = {"mobile": 30, "visual": 30, "logic": 20, "size": 20}
    report = {
        "summary": {
            "total_input": len(data),
            "duplicates_removed": duplicate_count,
            "passed": len(passed_items),
            "rejected": len(rejected_items),
            "pass_rate": f"{(len(passed_items) / len(data) * 100) if data else 0:.1f}%",
            "min_score_threshold": MIN_QUALITY_SCORE,
        },
        "score_distribution": dict(sorted(score_dist.items())),
        "top_issues": dict(issue_freq.most_common(15)),
        "layer_avg_scores": {
            layer: f"{(sum(scores) / len(scores)):.1f}/{max_scores[layer]}"
            for layer, scores in layer_scores.items()
            if scores
        },
        "data_composition": {
            "short_instructions": sum(1 for item, _ in deduped_results if len(item["messages"][1]["content"]) < 20),
            "detailed_instructions": sum(1 for item, _ in deduped_results if len(item["messages"][1]["content"]) >= 20),
        },
    }
    with REPORT.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print("=" * 56)
    print("PocketVibe 数据质量验证报告")
    print("=" * 56)
    print(f"输入数据: {len(data)} 条")
    print(f"去重淘汰: {duplicate_count} 条")
    print(f"通过数量: {len(passed_items)} 条")
    print(f"淘汰数量: {len(rejected_items)} 条")
    print(f"通过率: {(len(passed_items) / len(data) * 100) if data else 0:.1f}%")
    print(f"清洗数据: {OUTPUT}")
    print(f"淘汰样本: {REJECT}")
    print(f"质量报告: {REPORT}")


if __name__ == "__main__":
    main()
