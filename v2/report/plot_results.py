#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PocketVibe 报告作图脚本
=================================================================
基于 data/eval/compare_v1v2p_results.json + showcase_results.json
生成 3 张报告用图：
  1. report/fig_radar_v1_vs_v2.png    - 5 维度雷达图（V1 vs V2 均值）
  2. report/fig_per_case_bars.png     - 逐题 100 分制柱状图（C1-C5）
  3. report/fig_rounds_summary.png    - 两轮均分对比柱状图（Compare / Showcase）

运行: cd "Enoch - Version2" && python report/plot_results.py
依赖: pip install matplotlib numpy
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -------- 中文字体（Windows 优先 Microsoft YaHei / SimHei） --------
for fam in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]:
    try:
        mpl.font_manager.findfont(fam, fallback_to_default=False)
        mpl.rcParams["font.family"] = fam
        break
    except Exception:
        continue
mpl.rcParams["axes.unicode_minus"] = False

# -------- 路径 --------
ROOT = Path(__file__).resolve().parent.parent     # Enoch - Version2/
EVAL = ROOT / "data" / "eval"
OUT  = ROOT / "report"
OUT.mkdir(parents=True, exist_ok=True)

COMPARE = json.loads((EVAL / "compare_v1v2p_results.json").read_text(encoding="utf-8"))
SHOWCASE = json.loads((EVAL / "showcase_results.json").read_text(encoding="utf-8"))

# =================================================================
# 整理 compare 数据：合并同 tag 的 v1/v2 分项
# =================================================================
merged = {}   # tag -> {"case":..., "v1":{...}, "v2":{...}}
for rec in COMPARE["details"]:
    tag = rec["tag"]
    merged.setdefault(tag, {"case": rec["case"]})
    if "v1" in rec:
        merged[tag]["v1"] = rec["v1"]["score"]
    if "v2" in rec:
        merged[tag]["v2"] = rec["v2"]["score"]

# 保持 C1-C5 顺序
ORDER = [
    "C1_depth_stopwatch_lap",
    "C2_breadth_swim_timer",
    "C3_reasoning_calc_paren",
    "C4_combination_todo_pomodoro",
    "C5_cross_rps_scoreboard",
]
SHORT = {
    "C1_depth_stopwatch_lap":      "C1 分段秒表\n(DEPTH)",
    "C2_breadth_swim_timer":       "C2 游泳秒表\n(BREADTH)",
    "C3_reasoning_calc_paren":     "C3 括号计算器\n(REASONING)",
    "C4_combination_todo_pomodoro":"C4 待办+番茄钟\n(COMBINATION)",
    "C5_cross_rps_scoreboard":     "C5 石头剪刀布\n(CROSS)",
}

DIMS = ["J_js_depth", "I_instruct", "F_functional", "C_css", "S_structure"]
DIM_LABELS = ["J\nJS 深度\n(满分30)", "I\n指令遵循\n(满分25)", "F\n功能可运行\n(满分20)", "C\nCSS 质量\n(满分15)", "S\n结构合规\n(满分10)"]
DIM_MAX    = [30, 25, 20, 15, 10]

# =================================================================
# 图 1: 雷达图 (V1 vs V2 均值, 5 维度)
# =================================================================
def plot_radar():
    v1_avg = np.zeros(5); v2_avg = np.zeros(5); n = 0
    for tag in ORDER:
        if "v1" in merged[tag] and "v2" in merged[tag]:
            v1_avg += np.array([merged[tag]["v1"][d] for d in DIMS])
            v2_avg += np.array([merged[tag]["v2"][d] for d in DIMS])
            n += 1
    v1_avg /= n; v2_avg /= n

    # 归一化到 0-100，便于对比（按满分占比）
    v1_pct = v1_avg / np.array(DIM_MAX) * 100
    v2_pct = v2_avg / np.array(DIM_MAX) * 100

    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    v1_p = v1_pct.tolist() + [v1_pct[0]]
    v2_p = v2_pct.tolist() + [v2_pct[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, v1_p, "o-", linewidth=2.5, label=f"V1 (avg={COMPARE['v1_avg']:.1f})", color="#667eea")
    ax.fill(angles, v1_p, alpha=0.22, color="#667eea")
    ax.plot(angles, v2_p, "s-", linewidth=2.5, label=f"V2 (avg={COMPARE['v2_avg']:.1f})", color="#f5576c")
    ax.fill(angles, v2_p, alpha=0.22, color="#f5576c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9, color="#666")
    ax.set_title("图1  V1 vs V2  5 维度能力雷达图（5 题均值，归一化到满分百分比）",
                 fontsize=13, pad=22, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=11)
    ax.grid(True, alpha=0.4)

    path = OUT / "fig_radar_v1_vs_v2.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] 雷达图 -> {path}")

# =================================================================
# 图 2: 逐题柱状图 (C1-C5, V1 vs V2)
# =================================================================
def plot_per_case():
    labels = [SHORT[t] for t in ORDER]
    v1_scores = [merged[t]["v1"]["total"] for t in ORDER]
    v2_scores = [merged[t]["v2"]["total"] for t in ORDER]
    delta     = [v2 - v1 for v1, v2 in zip(v1_scores, v2_scores)]

    x = np.arange(len(ORDER))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.2))
    b1 = ax.bar(x - w/2, v1_scores, w, label="V1", color="#667eea", edgecolor="white")
    b2 = ax.bar(x + w/2, v2_scores, w, label="V2", color="#f5576c", edgecolor="white")

    # 数值标签
    for b, v in zip(b1, v1_scores):
        ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center", fontsize=10, color="#333")
    for b, v in zip(b2, v2_scores):
        ax.text(b.get_x()+b.get_width()/2, v+1, str(v), ha="center", fontsize=10, color="#333")

    # Δ 标注
    for xi, d in zip(x, delta):
        color = "#43a047" if d > 0 else "#d32f2f"
        mark  = "+" if d > 0 else ""
        ax.text(xi, 5, f"Δ{mark}{d}", ha="center", fontsize=11, weight="bold",
                color=color, bbox=dict(facecolor="white", edgecolor=color, boxstyle="round,pad=0.3"))

    ax.set_ylabel("总分 (100 分制)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f"图2  V1 vs V2  逐题对比（Compare-Hard 5 题，V1 胜 {COMPARE['v1_wins']} / 平 {COMPARE['ties']} / V2 胜 {COMPARE['v2_wins']}）",
                 fontsize=13, pad=14, weight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(COMPARE["v1_avg"], color="#667eea", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.axhline(COMPARE["v2_avg"], color="#f5576c", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.text(4.6, COMPARE["v1_avg"]+1, f"V1 均值 {COMPARE['v1_avg']:.1f}", color="#667eea", fontsize=9)
    ax.text(4.6, COMPARE["v2_avg"]-3, f"V2 均值 {COMPARE['v2_avg']:.1f}", color="#f5576c", fontsize=9)

    path = OUT / "fig_per_case_bars.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] 逐题柱状图 -> {path}")

# =================================================================
# 图 3: 两轮均分对比 + 人工评判标记
# =================================================================
def plot_rounds_summary():
    # Round A: Compare-Hard 5 题
    compare_v1 = COMPARE["v1_avg"]
    compare_v2 = COMPARE["v2_avg"]

    # Round B: Showcase V2 主场 2 题（AI 评分）
    show_v1 = np.mean([r["total_score"] for r in SHOWCASE if r["model"] == "V1"])
    show_v2 = np.mean([r["total_score"] for r in SHOWCASE if r["model"] == "V2"])

    rounds = ["第二轮 Compare-Hard\n(C1-C5 通用型 5 题)", "第三轮 Showcase\n(S1-S2 V2 主场 2 题)"]
    v1_vals = [compare_v1, show_v1]
    v2_vals = [compare_v2, show_v2]

    x = np.arange(len(rounds))
    w = 0.32

    fig, ax = plt.subplots(figsize=(10, 6.2))
    b1 = ax.bar(x - w/2, v1_vals, w, label="V1", color="#667eea", edgecolor="white")
    b2 = ax.bar(x + w/2, v2_vals, w, label="V2", color="#f5576c", edgecolor="white")

    for b, v in zip(b1, v1_vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}", ha="center", fontsize=11, color="#333", weight="bold")
    for b, v in zip(b2, v2_vals):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}", ha="center", fontsize=11, color="#333", weight="bold")

    # 人工评判胜负标记
    verdicts = [
        ("V1 胜 4 / 平 0 / V2 胜 1\nAI 评分 + 人工评判一致", "#667eea"),
        ("AI 评分 V2 胜 2/2\n但人工评判 V1 完胜 2/2 (!)", "#d32f2f"),
    ]
    for xi, (txt, col) in zip(x, verdicts):
        ax.text(xi, 30, txt, ha="center", fontsize=10, color=col,
                bbox=dict(facecolor="white", edgecolor=col, boxstyle="round,pad=0.5", linewidth=1.5))

    ax.set_ylabel("平均总分 (100 分制)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(rounds, fontsize=11)
    ax.set_title("图3  两轮评测均分汇总（含人工评判推翻 AI 评分的决定性证据）",
                 fontsize=13, pad=14, weight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    path = OUT / "fig_rounds_summary.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[OK] 两轮均分图 -> {path}")

# =================================================================
if __name__ == "__main__":
    print(f"读取: {EVAL}")
    print(f"输出: {OUT}\n")
    plot_radar()
    plot_per_case()
    plot_rounds_summary()
    print("\n全部完成。把 3 张 PNG 直接插入报告对应位置即可。")
