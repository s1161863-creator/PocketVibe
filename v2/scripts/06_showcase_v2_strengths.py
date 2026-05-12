#!/usr/bin/env python3
"""
=================================================================
PocketVibe V2+ — V2 强项展示对比（Showcase）
=================================================================
目标：用 2 条专门针对 V2 训练目标设计的"超纲题"，让 V1/V2 同台输出，
     直观展示 V2 在跨类组合 + 长HTML + 复杂状态管理 + 视觉风格控制
     四个维度上的优势。

两题设计依据：
  S1 = 跨类组合 + 长HTML + localStorage （对应 §3.2 策略 C / §3.3 Evol COMBINATION）
  S2 = 视觉风格精准 + 游戏逻辑 + 计分状态管理 （对应 §3.2 策略 A 四风格 + §3.3 Evol DEPTH）

输出路径: data/eval/
  showcase_S1_morning_routine_V1.html / _V2.html
  showcase_S2_tictactoe_neon_V1.html / _V2.html
  showcase_results.md / .csv / .json

运行：在 HPC 上 sbatch slurm/eval_showcase.slurm
=================================================================
本版本（修复版）：使用 pv_scoring 实际导出的 API：
  - score_html_100(code, test_case) -> dict
  - generate_best_of_n(tokenizer, model, test_case, label) -> (best_html, candidates)
  - load_tokenizer() / load_model_with_adapter(adapter_dir)
  - SYSTEM_PROMPT
=================================================================
"""

import os, sys, json, csv, gc
import torch

# 共享评分 / 推理模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_scoring import (
    score_html_100,
    generate_best_of_n,
    load_tokenizer,
    load_model_with_adapter,
    SYSTEM_PROMPT,
)

# ================= 路径配置 =================
os.environ.setdefault("HF_HOME", "/opt/shared/model-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

V1_ADAPTER = os.path.expanduser("~/PocketVibe/outputs/qlora-run1/final_adapter")
V2_ADAPTER = os.path.expanduser("~/PocketVibe/outputs/qlora-v2-run1/final_adapter")
EVAL_DIR   = os.path.expanduser("~/PocketVibe/data/eval")
os.makedirs(EVAL_DIR, exist_ok=True)

# ================= Showcase 题目 =================
# 注意：expected_features 格式必须和 pv_scoring.TEST_CASES 保持一致
# [(功能描述, [关键词或正则...]), ...]
SHOWCASES = [
    {
        "tag": "S1_morning_routine",
        "category": "CROSS_CATEGORY + LONG_HTML",
        "title": "晨间例行三合一工具",
        "instruction": (
            "做一个「晨间例行」手机页面，包含三个模块："
            "（1）待办清单，可以添加任务、勾选完成、删除；"
            "（2）5 分钟冥想计时器，有开始/暂停按钮，倒计时过程中显示呼吸引导文字（吸气/呼气交替，每 4 秒切换一次）；"
            "（3）今日心情打卡，5 个表情按钮（😀😊😐😔😡）选一个记录当天心情。"
            "所有数据（待办列表、今日心情）都要用 localStorage 持久化，刷新页面后不丢失。"
            "整体用温暖的橙色到粉色渐变背景，卡片分三块垂直排列。"
        ),
        "expected_features": [
            ("添加待办",          ["add", "push", "添加", "新增"]),
            ("勾选待办",          ["check", "toggle", "done", "勾选", "☑", "✅"]),
            ("删除待办",          ["remove", "splice", "delete", "删除", "×"]),
            ("冥想计时启动",      ["start", "开始", "setInterval", "setTimeout"]),
            ("5 分钟倒计时",      ["300", "5*60", "60*5", "5 *", "5*60"]),
            ("呼吸引导文字切换",  ["吸气", "呼气", "inhale", "exhale", "breath"]),
            ("4 秒切换",          ["4000", "4 *", "*4", "4000)"]),
            ("心情 5 表情",       ["😀", "😊", "😐", "😔", "😡", "mood", "心情"]),
            ("localStorage 持久化", ["localStorage", "setItem", "getItem"]),
            ("橙色/粉色渐变",     ["linear-gradient", "orange", "pink", "#f", "#ff"]),
            ("三块垂直卡片",      ["card", "section", "flex-direction", "column", "margin"]),
        ],
    },
    {
        "tag": "S2_tictactoe_neon",
        "category": "VISUAL_STYLE + GAME_LOGIC + SCORING",
        "title": "暗色霓虹风井字棋 + 计分板",
        "instruction": (
            "做一个井字棋小游戏，要求如下："
            "（1）暗色背景（纯黑到深紫渐变），3x3 棋盘格子用霓虹青色边框，玩家落子 X 用霓虹粉色，AI 落子 O 用霓虹绿色；"
            "（2）玩家先手，点击空格落子后 AI 随机选一个空格回应；"
            "（3）每局结束（三连或平局）自动判定胜负并弹出提示，然后 1 秒后自动开始新一局；"
            "（4）右侧或下方显示累计比分板：玩家胜 / AI 胜 / 平局三个计数器，数字也用霓虹发光效果；"
            "（5）有一个「重置比分」按钮。"
        ),
        "expected_features": [
            ("3x3 棋盘",          ["grid", "3", "9", "board", "cell"]),
            ("暗色/深紫渐变",     ["#0", "#1", "#2", "purple", "black", "linear-gradient"]),
            ("霓虹青色边框",      ["cyan", "#0ff", "#00ffff", "aqua", "border"]),
            ("X 粉色",            ["pink", "#f", "magenta", "#ff00ff", "X"]),
            ("O 绿色",            ["green", "#0f0", "#00ff00", "lime", "O"]),
            ("AI 随机出招",       ["Math.random", "random", "floor"]),
            ("胜负判定",          ["win", "check", "line", "三连", "row", "col", "diag"]),
            ("平局判定",          ["draw", "tie", "full", "平局"]),
            ("1 秒后新局",        ["1000", "setTimeout", "reset", "新局"]),
            ("比分累计",          ["score", "count", "++", "player", "ai", "draw"]),
            ("发光效果",          ["text-shadow", "box-shadow", "glow", "filter"]),
            ("重置比分",          ["reset", "clear", "重置", "清零"]),
            ("动态 DOM 更新",     ["textContent", "innerHTML", "innerText"]),
        ],
    },
]


def run_one(model, tokenizer, showcase: dict, model_label: str):
    """为一条 showcase 题目生成 HTML（Best-of-N）"""
    tag = showcase["tag"]
    print(f"\n---- [{model_label}] Showcase: {tag} ----")
    print(f"     场景: {showcase['title']}")

    # pv_scoring.generate_best_of_n 的签名: (tokenizer, model, test_case, label)
    # test_case 需要 "instruction" + "expected_features" 两个字段
    best_html, candidates = generate_best_of_n(tokenizer, model, showcase, label=model_label)

    # 重新评一次获取完整 dims
    best_score = score_html_100(best_html, showcase)

    # 存最佳结果
    fname = f"showcase_{tag}_{model_label}.html"
    fpath = os.path.join(EVAL_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(best_html)
    print(f"     已保存: {fname}  |  总分 {best_score['total']}/100")
    print(f"     维度: J={best_score['J_js_depth']} I={best_score['I_instruct']} "
          f"F={best_score['F_functional']} C={best_score['C_css']} S={best_score['S_structure']}")

    return {
        "model":        model_label,
        "tag":          tag,
        "title":        showcase["title"],
        "category":     showcase["category"],
        "total_score":  best_score["total"],
        "J":            best_score["J_js_depth"],
        "I":            best_score["I_instruct"],
        "F":            best_score["F_functional"],
        "C":            best_score["C_css"],
        "S":            best_score["S_structure"],
        "html_file":    fname,
        "html_length":  len(best_html),
        "instruction":  showcase["instruction"],
        "candidates_count": len(candidates),
    }


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    all_results = []

    # 加载 tokenizer（两个模型共用）
    print(">>> 加载 tokenizer...")
    tokenizer = load_tokenizer()

    # ---------- V1 阶段 ----------
    print("\n" + "=" * 60)
    print(">>> V1 阶段：加载 V1 adapter")
    print(f"    路径: {V1_ADAPTER}")
    print("=" * 60)
    v1_model = load_model_with_adapter(V1_ADAPTER)
    for sc in SHOWCASES:
        all_results.append(run_one(v1_model, tokenizer, sc, "V1"))
    del v1_model
    free_gpu()

    # ---------- V2 阶段 ----------
    print("\n" + "=" * 60)
    print(">>> V2 阶段：加载 V2 adapter")
    print(f"    路径: {V2_ADAPTER}")
    print("=" * 60)
    v2_model = load_model_with_adapter(V2_ADAPTER)
    for sc in SHOWCASES:
        all_results.append(run_one(v2_model, tokenizer, sc, "V2"))
    del v2_model
    free_gpu()

    # ---------- 落盘报表 ----------
    # JSON
    with open(os.path.join(EVAL_DIR, "showcase_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # CSV
    with open(os.path.join(EVAL_DIR, "showcase_results.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tag", "title", "total_score", "J", "I", "F", "C", "S",
                    "html_length", "html_file"])
        for r in all_results:
            w.writerow([
                r["model"], r["tag"], r["title"], r["total_score"],
                r["J"], r["I"], r["F"], r["C"], r["S"],
                r["html_length"], r["html_file"],
            ])

    # Markdown 对比
    by_tag = {}
    for r in all_results:
        by_tag.setdefault(r["tag"], {})[r["model"]] = r

    md_lines = [
        "# V2 强项展示对比（Showcase）",
        "",
        "> 两条题目均针对 V2 训练目标（跨类组合 + 长HTML + 视觉风格 + 复杂状态）设计。",
        "",
        "## 分数对比（100 分制）",
        "",
        "| 题号 | 场景 | V1 总分 | V2 总分 | Δ (V2-V1) |",
        "|------|------|---------|---------|-----------|",
    ]
    for tag, pair in by_tag.items():
        v1 = pair.get("V1", {})
        v2 = pair.get("V2", {})
        v1s = v1.get("total_score", 0)
        v2s = v2.get("total_score", 0)
        delta = v2s - v1s
        title = v2.get("title") or v1.get("title", tag)
        md_lines.append(f"| {tag} | {title} | {v1s} | {v2s} | **{'+' if delta >= 0 else ''}{delta}** |")

    md_lines += [
        "",
        "## 维度细分 J/I/F/C/S",
        "",
        "| 题号 | 模型 | J(30) | I(25) | F(20) | C(15) | S(10) | 总分 |",
        "|------|------|-------|-------|-------|-------|-------|------|",
    ]
    for r in all_results:
        md_lines.append(
            f"| {r['tag']} | **{r['model']}** | "
            f"{r['J']} | {r['I']} | {r['F']} | {r['C']} | {r['S']} | "
            f"**{r['total_score']}** |"
        )

    md_lines += [
        "",
        "## HTML 长度（反映输出完整度）",
        "",
        "| 题号 | V1 字符数 | V2 字符数 | V2/V1 |",
        "|------|-----------|-----------|-------|",
    ]
    for tag, pair in by_tag.items():
        v1len = pair.get("V1", {}).get("html_length", 0)
        v2len = pair.get("V2", {}).get("html_length", 0)
        ratio = f"{v2len/v1len:.2f}x" if v1len else "N/A"
        md_lines.append(f"| {tag} | {v1len} | {v2len} | {ratio} |")

    md_lines += ["", "## 产物清单（data/eval/ 下）", ""]
    for r in all_results:
        md_lines.append(f"- `{r['html_file']}` — {r['model']} / {r['tag']} / {r['total_score']}分")

    with open(os.path.join(EVAL_DIR, "showcase_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 控制台汇总
    print("\n" + "=" * 60)
    print("V2 强项展示对比 — 最终汇总")
    print("=" * 60)
    for tag, pair in by_tag.items():
        v1s = pair.get("V1", {}).get("total_score", 0)
        v2s = pair.get("V2", {}).get("total_score", 0)
        print(f"  {tag}:  V1={v1s}/100   V2={v2s}/100   Δ={v2s-v1s:+d}")
    print("=" * 60)
    print(f"\n所有产物已保存到: {EVAL_DIR}")


if __name__ == "__main__":
    main()
