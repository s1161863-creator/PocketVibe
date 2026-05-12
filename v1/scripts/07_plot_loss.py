#!/usr/bin/env python3
"""
PocketVibe — Loss 曲线绘图（本地运行）
用途: 读取 HPC 训练日志，生成 Loss 曲线图，用于报告
运行: python scripts/07_plot_loss.py
依赖: pip install matplotlib
输出: report/loss_curve.png
"""
import json, os
from pathlib import Path

# ── 路径配置 ──
LOG_PATH    = Path("logs/train_log.json")
REPORT_DIR  = Path("report")
OUTPUT_PNG  = REPORT_DIR / "loss_curve.png"

REPORT_DIR.mkdir(exist_ok=True)

if not LOG_PATH.exists():
    print(f"❌ 找不到日志文件: {LOG_PATH}")
    print("   请先从 HPC 下载:")
    print("   scp student07@aaillm.eduhk.hk:~/PocketVibe/logs/train_log.json logs/")
    raise SystemExit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("❌ 缺少 matplotlib，请先安装: pip install matplotlib")
    raise SystemExit(1)

# ── 读取日志 ──
with open(LOG_PATH, "r", encoding="utf-8") as f:
    logs = json.load(f)

train_steps  = [x["step"] for x in logs if "loss" in x]
train_losses = [x["loss"] for x in logs if "loss" in x]
eval_steps   = [x["step"] for x in logs if "eval_loss" in x]
eval_losses  = [x["eval_loss"] for x in logs if "eval_loss" in x]

print(f"✅ 加载日志: {len(train_steps)} 个训练点, {len(eval_steps)} 个验证点")

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(9, 5))

# 训练 loss（折线）
ax.plot(train_steps, train_losses,
        color="#667eea", linewidth=1.5, alpha=0.85,
        label="Training Loss")

# 验证 loss（红点连线）
if eval_losses:
    ax.plot(eval_steps, eval_losses,
            "o-", color="#f5576c", linewidth=2, markersize=7,
            label="Validation Loss")

    # 标注最低验证 loss
    best_idx  = eval_losses.index(min(eval_losses))
    best_step = eval_steps[best_idx]
    best_loss = eval_losses[best_idx]
    ax.annotate(
        f"Best: {best_loss:.4f}\n(epoch {best_idx + 1})",
        xy=(best_step, best_loss),
        xytext=(best_step + max(train_steps) * 0.05, best_loss + 0.02),
        arrowprops=dict(arrowstyle="->", color="#f5576c"),
        fontsize=9, color="#f5576c",
    )

# 样式
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("QLoRA Fine-tuning Loss Curve\n"
             "Qwen2.5-Coder-1.5B-Instruct | LoRA r=32 α=64 | 4 epochs",
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle="--")
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Loss 曲线已保存: {OUTPUT_PNG}")

# 打印训练总结
print(f"\n── 训练总结 ──")
print(f"   初始 Train Loss : {train_losses[0]:.4f}")
print(f"   最终 Train Loss : {train_losses[-1]:.4f}")
if eval_losses:
    print(f"   最低 Eval Loss  : {min(eval_losses):.4f}  (epoch {eval_losses.index(min(eval_losses)) + 1})")
    print(f"   最终 Eval Loss  : {eval_losses[-1]:.4f}")

plt.show()
