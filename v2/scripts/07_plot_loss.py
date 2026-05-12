#!/usr/bin/env python3
"""
PocketVibe V2 — 07: Loss 曲线绘图（本地运行）
=================================================================
从 HPC 下载 logs/train_log_v2.json 后，在本地运行此脚本
生成三张图供报告使用：
  1. Loss 曲线（train + eval）
  2. V1 vs V2 eval_loss 对比折线图
  3. 各类别评估得分柱状图（来自 05_eval_compare.py 的输出）

依赖：pip install matplotlib pandas

运行：python scripts/07_plot_loss.py
输出：report/loss_curve_v2.png
      report/v1_vs_v2_loss.png
      report/category_score.png
=================================================================
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")          # 无显示器环境（HPC）也能跑
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 路径配置（本地路径，按实际情况修改）──
PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_V2       = os.path.join(PROJECT_DIR, "logs", "train_log_v2.json")
LOG_V1       = os.path.join(PROJECT_DIR, "logs", "train_log.json")     # V1 旧日志（可选）
EVAL_CAT_JSON= os.path.join(PROJECT_DIR, "data", "eval", "compare_v2_by_cat.json")
REPORT_DIR   = os.path.join(PROJECT_DIR, "report")

os.makedirs(REPORT_DIR, exist_ok=True)

# ── 绘图全局样式 ──
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

COLORS = {
    "train": "#667eea",
    "eval":  "#f5576c",
    "v1":    "#aaa",
    "v2":    "#667eea",
    "base":  "#adb5bd",
    "ft":    "#667eea",
}


# ================================================================
# 图 1：V2 训练/验证 Loss 曲线
# ================================================================
def plot_loss_curve(log_path: str, out_path: str, title: str = "QLoRA V2 — Loss Curve"):
    if not os.path.exists(log_path):
        print(f"⚠️  找不到日志文件：{log_path}，跳过图1")
        return

    with open(log_path, encoding="utf-8") as f:
        logs = json.load(f)

    train_steps  = [x["step"] for x in logs if "loss" in x]
    train_losses = [x["loss"] for x in logs if "loss" in x]
    eval_steps   = [x["step"] for x in logs if "eval_loss" in x]
    eval_losses  = [x["eval_loss"] for x in logs if "eval_loss" in x]

    fig, ax = plt.subplots(figsize=(8, 5))

    if train_losses:
        ax.plot(train_steps, train_losses,
                color=COLORS["train"], alpha=0.7, linewidth=1.2, label="Train Loss")

    if eval_losses:
        ax.plot(eval_steps, eval_losses,
                color=COLORS["eval"], marker="o", markersize=5,
                linewidth=2, label="Eval Loss (held-out)")
        # 标注最低点
        best_idx  = eval_losses.index(min(eval_losses))
        best_step = eval_steps[best_idx]
        best_loss = eval_losses[best_idx]
        ax.annotate(
            f"best={best_loss:.3f}\n(step {best_step})",
            xy=(best_step, best_loss),
            xytext=(best_step + max(eval_steps) * 0.05, best_loss + 0.02),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#888"),
            color="#f5576c",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ 图1 已保存：{out_path}")


# ================================================================
# 图 2：V1 vs V2 eval_loss 对比
# ================================================================
def plot_v1_v2_compare(log_v1: str, log_v2: str, out_path: str):
    has_v1 = os.path.exists(log_v1)
    has_v2 = os.path.exists(log_v2)

    if not has_v2:
        print(f"⚠️  找不到 V2 日志，跳过图2")
        return

    def load_eval(path):
        with open(path, encoding="utf-8") as f:
            logs = json.load(f)
        steps  = [x["step"]      for x in logs if "eval_loss" in x]
        losses = [x["eval_loss"] for x in logs if "eval_loss" in x]
        return steps, losses

    fig, ax = plt.subplots(figsize=(8, 5))

    if has_v1:
        s1, l1 = load_eval(log_v1)
        # 归一化到 [0,1] 方便对比
        max_s1 = max(s1) if s1 else 1
        ax.plot([s / max_s1 for s in s1], l1,
                color=COLORS["v1"], linestyle="--", linewidth=1.5,
                label="V1 Eval Loss (random split)")

    s2, l2 = load_eval(log_v2)
    max_s2 = max(s2) if s2 else 1
    ax.plot([s / max_s2 for s in s2], l2,
            color=COLORS["v2"], linewidth=2, marker="o", markersize=4,
            label="V2 Eval Loss (held-out split)")

    ax.set_xlabel("Training Progress (normalized)")
    ax.set_ylabel("Eval Loss")
    ax.set_title("V1 vs V2 — Eval Loss Comparison", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.8)

    # 注释文字
    ax.text(0.98, 0.95,
            "V2 held-out split\nprevents data leakage",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ 图2 已保存：{out_path}")


# ================================================================
# 图 3：各类别 base vs fine-tuned 得分柱状图
# ================================================================
def plot_category_scores(cat_json: str, out_path: str):
    if not os.path.exists(cat_json):
        print(f"⚠️  找不到类别评估文件：{cat_json}，跳过图3")
        return

    with open(cat_json, encoding="utf-8") as f:
        data = json.load(f)

    # 按 ft_avg 降序
    cats   = sorted(data.keys(), key=lambda c: data[c]["ft_avg"], reverse=True)
    base_s = [data[c]["base_avg"] for c in cats]
    ft_s   = [data[c]["ft_avg"]   for c in cats]

    x      = range(len(cats))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.9), 5))
    bars_b = ax.bar([i - width/2 for i in x], base_s, width,
                    label="Base Model", color=COLORS["base"], alpha=0.85)
    bars_f = ax.bar([i + width/2 for i in x], ft_s, width,
                    label="Fine-tuned (V2)", color=COLORS["ft"], alpha=0.9)

    # 数值标注
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, color="#555")
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7, color=COLORS["ft"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Quality Score (0–1)")
    ax.set_title("Base vs Fine-tuned — Per-Category Quality Score", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.8)

    # 标注 held-out 类别
    held_out = {"creative", "social", "finance", "parenting", "planner", "utility", "cross", "other"}
    for i, cat in enumerate(cats):
        if cat in held_out:
            ax.get_xticklabels()[i].set_color("#f5576c")
    ax.text(0.01, 0.97, "红色 = held-out 类别（训练集中不存在）",
            transform=ax.transAxes, fontsize=7, color="#f5576c", va="top")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ 图3 已保存：{out_path}")


# ================================================================
# main
# ================================================================
def main():
    print(f"📂 项目目录：{PROJECT_DIR}")

    # 图 1：V2 Loss 曲线
    plot_loss_curve(
        LOG_V2,
        os.path.join(REPORT_DIR, "loss_curve_v2.png"),
        title="PocketVibe V2 — QLoRA Training Loss (Held-out Eval)"
    )

    # 图 2：V1 vs V2 eval_loss 对比
    plot_v1_v2_compare(
        LOG_V1,
        LOG_V2,
        os.path.join(REPORT_DIR, "v1_vs_v2_loss.png")
    )

    # 图 3：类别得分柱状图
    plot_category_scores(
        EVAL_CAT_JSON,
        os.path.join(REPORT_DIR, "category_score.png")
    )

    print(f"\n📁 所有图片已保存到：{REPORT_DIR}/")
    print("   报告中引用路径：")
    print("     report/loss_curve_v2.png")
    print("     report/v1_vs_v2_loss.png")
    print("     report/category_score.png")


if __name__ == "__main__":
    main()
