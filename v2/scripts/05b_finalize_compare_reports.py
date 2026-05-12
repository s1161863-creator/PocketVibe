#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
PocketVibe V2+ — 05b: 离线补全 compare 报表（不需要 GPU）
=================================================================
背景：
  Job 1448 / 1455 因 bitsandbytes CUDA 运行期故障在生成 C4_v2 之前崩溃，
  已存盘 8 个 HTML (V1 全部 5 个 + V2 只有 C1/C2/C3)，但 .csv/.md/.json 报表
  都还没写出。重跑代价大，不如复用已有产物离线补齐。

本脚本做的事：
  1) 扫描 data/eval/compare_v1v2p_{tag}_v{1,2}.html
  2) 能找到的 HTML 用 pv_scoring.score_html_100 正常打分
  3) 找不到的（C4_v2 / C5_v2）记为 MISSING（分数按 0 处理，用 "--" 展示）
  4) 产出与 05 脚本格式一致的 3 份报表 + 控制台汇总

只做文件 I/O + 正则打分，不加载模型 → 登录节点直接 python 跑即可。

运行：
    cd ~/PocketVibe
    python scripts/05b_finalize_compare_reports.py
=================================================================
"""
import os, json, csv

# pv_scoring 是纯 Python 的打分/测试集定义，不 import torch/bnb
from pv_scoring import BASE_MODEL, TEST_CASES, SAMPLE_KWARGS, BEST_OF_N, score_html_100

HOME           = os.path.expanduser("~")
EVAL_DIR       = os.path.join(HOME, "PocketVibe", "data", "eval")
V1_ADAPTER_DIR = os.path.join(HOME, "PocketVibe", "outputs", "qlora-run1",    "final_adapter")
V2_ADAPTER_DIR = os.path.join(HOME, "PocketVibe", "outputs", "qlora-v2-run1", "final_adapter")


def read_html(tag: str, label: str):
    path = os.path.join(EVAL_DIR, f"compare_v1v2p_{tag}_{label.lower()}.html")
    if not os.path.isfile(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return f.read(), path


def missing_score(case):
    """占位 0 分结构，形状与 score_html_100 保持一致以便表格渲染"""
    return {
        "total": 0,
        "J_js_depth":   0,
        "I_instruct":   0,
        "F_functional": 0,
        "C_css":        0,
        "S_structure":  0,
        "length":       0,
        "breakdown": {"I": {"hit_count": 0, "total": len(case.get("instruction_points", []))}},
        "_missing":   True,
    }


def fmt_delta(d):
    if d > 0: return f"+{d}"
    if d < 0: return f"{d}"
    return "+-0"


def fmt_cell(v, miss):
    return "--" if miss else f"{v}"


def main():
    print("=" * 64)
    print("PocketVibe V2+ -- offline finalize compare reports (no GPU needed)")
    print("=" * 64)

    rows, v1_results, v2_results = [], [], []
    missing_files = []

    for case in TEST_CASES:
        tag = case["tag"]
        print(f"\n--- {tag} ---")

        v1_code, v1_path = read_html(tag, "V1")
        v2_code, v2_path = read_html(tag, "V2")

        if v1_code is None:
            print(f"  V1 MISSING: {v1_path}")
            v1_score = missing_score(case); missing_files.append(v1_path)
        else:
            v1_score = score_html_100(v1_code, case)
            print(f"  V1 OK  -> total={v1_score['total']}/100  ({len(v1_code)} chars)")

        if v2_code is None:
            print(f"  V2 MISSING: {v2_path}")
            v2_score = missing_score(case); missing_files.append(v2_path)
        else:
            v2_score = score_html_100(v2_code, case)
            print(f"  V2 OK  -> total={v2_score['total']}/100  ({len(v2_code)} chars)")

        v1_results.append({"case": case, "score": v1_score, "code": v1_code})
        v2_results.append({"case": case, "score": v2_score, "code": v2_code})

        rows.append({
            "tag":         tag,
            "category":    case["category"],
            "instruction": case["instruction"][:50],
            "v1_missing":  v1_code is None,
            "v2_missing":  v2_code is None,
            "v1_total":    v1_score["total"],
            "v2_total":    v2_score["total"],
            "delta_total": v2_score["total"] - v1_score["total"],
            "v1_J": v1_score["J_js_depth"],   "v2_J": v2_score["J_js_depth"],
            "v1_I": v1_score["I_instruct"],   "v2_I": v2_score["I_instruct"],
            "v1_F": v1_score["F_functional"], "v2_F": v2_score["F_functional"],
            "v1_C": v1_score["C_css"],        "v2_C": v2_score["C_css"],
            "v1_S": v1_score["S_structure"],  "v2_S": v2_score["S_structure"],
            "v1_len": v1_score["length"],     "v2_len": v2_score["length"],
            "v1_i_hits":   v1_score["breakdown"]["I"].get("hit_count", 0),
            "v2_i_hits":   v2_score["breakdown"]["I"].get("hit_count", 0),
            "i_total_pts": v1_score["breakdown"]["I"].get("total",     0),
        })

    # ------------------------------------------------------------------
    # 控制台汇总（与 05 一致；缺失项展示 --）
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("V1 vs V2 COMPARISON SUMMARY (offline finalize)")
    print(f"{'=' * 80}")
    print(f"{'case':<32} {'V1':>7} {'V2':>7} {'delta':>6}  dJ dI dF dC dS")
    print("-" * 80)
    for r in rows:
        dJ = r["v2_J"] - r["v1_J"]
        dI = r["v2_I"] - r["v1_I"]
        dF = r["v2_F"] - r["v1_F"]
        dC = r["v2_C"] - r["v1_C"]
        dS = r["v2_S"] - r["v1_S"]
        v1_cell = "--" if r["v1_missing"] else f"{r['v1_total']:>3}/100"
        v2_cell = "--" if r["v2_missing"] else f"{r['v2_total']:>3}/100"
        delta   = "--" if (r["v1_missing"] or r["v2_missing"]) else fmt_delta(r["delta_total"])
        print(f"{r['tag']:<32} {v1_cell:>7} {v2_cell:>7} {delta:>6}  "
              f"{fmt_delta(dJ):>3} {fmt_delta(dI):>3} {fmt_delta(dF):>3} "
              f"{fmt_delta(dC):>3} {fmt_delta(dS):>3}")

    # 只统计「两版都存在」的案例的均值，避免 MISSING 拖累平均
    valid = [r for r in rows if not r["v1_missing"] and not r["v2_missing"]]
    if valid:
        v1_avg = sum(r["v1_total"] for r in valid) / len(valid)
        v2_avg = sum(r["v2_total"] for r in valid) / len(valid)
        v2_wins = sum(1 for r in valid if r["delta_total"] > 0)
        tied    = sum(1 for r in valid if r["delta_total"] == 0)
        v1_wins = sum(1 for r in valid if r["delta_total"] < 0)
    else:
        v1_avg = v2_avg = 0.0
        v2_wins = tied = v1_wins = 0

    print("-" * 80)
    print(f"avg (valid cases only, n={len(valid)}):  "
          f"V1 = {v1_avg:.1f}/100   V2 = {v2_avg:.1f}/100   "
          f"delta = {v2_avg - v1_avg:+.1f}")
    print(f"wins (valid only): V2={v2_wins}  tie={tied}  V1={v1_wins}")
    if missing_files:
        print(f"\nMISSING HTMLs (skipped, counted as 0 in reports):")
        for p in missing_files:
            print(f"  - {p}")

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(EVAL_DIR, "compare_v1v2p_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    md_path = os.path.join(EVAL_DIR, "compare_v1v2p_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PocketVibe — V1 vs V2 Fine-grained Comparison (100-pt)\n\n")
        f.write(f"- **Base**: {BASE_MODEL}\n")
        f.write("- **V1**: `outputs/qlora-run1/final_adapter` (original / aligned with local Enoch repo)\n")
        f.write("- **V2**: `outputs/qlora-v2-run1/final_adapter` (Version2 new training, SLURM 1442)\n")
        f.write(f"- **Inference**: Best-of-{BEST_OF_N}, temp={SAMPLE_KWARGS['temperature']}, "
                f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
                f"max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}\n")
        f.write("- **Note**: Job 1448/1455 was interrupted by a runtime bitsandbytes CUDA fault "
                "after generating V2 for cases C1/C2/C3. C4_v2 / C5_v2 are marked `--` (MISSING). "
                "Averages and win counts are computed over **valid cases only**.\n\n")
        f.write(f"**Avg (n={len(valid)} valid cases)**: "
                f"V1 = {v1_avg:.1f}/100  |  V2 = {v2_avg:.1f}/100  "
                f"|  delta = {v2_avg - v1_avg:+.1f}\n\n")
        f.write(f"**Wins (valid only)**: V2={v2_wins} | tie={tied} | V1={v1_wins}\n\n")
        if missing_files:
            f.write("**Missing HTMLs**:\n\n")
            for p in missing_files:
                f.write(f"- `{os.path.basename(p)}`\n")
            f.write("\n")

        f.write("## Total score comparison\n\n")
        f.write("| Case | Category | V1 | V2 | delta |\n")
        f.write("|------|----------|----|----|----|\n")
        for r in rows:
            v1c = "--" if r["v1_missing"] else f"{r['v1_total']}/100"
            v2c = "--" if r["v2_missing"] else f"**{r['v2_total']}/100**"
            dlt = "--" if (r["v1_missing"] or r["v2_missing"]) else fmt_delta(r["delta_total"])
            f.write(f"| {r['tag']} | {r['category']} | {v1c} | {v2c} | {dlt} |\n")

        f.write("\n## Dimension breakdown (V1 -> V2)\n\n")
        f.write("| Case | J(30) | I(25) | F(20) | C(15) | S(10) | length | I hit-rate |\n")
        f.write("|------|-------|-------|-------|-------|-------|--------|------------|\n")
        for r in rows:
            def arrow(v1, v2, m1, m2):
                a = "--" if m1 else str(v1)
                b = "--" if m2 else str(v2)
                return f"{a}->{b}"

            # 命中率单独拼，避免 f-string 嵌套引号（兼容 Python 3.10）
            v1_hit = "--" if r["v1_missing"] else "{}/{}".format(r["v1_i_hits"], r["i_total_pts"])
            v2_hit = "--" if r["v2_missing"] else "{}/{}".format(r["v2_i_hits"], r["i_total_pts"])
            hit_col = "{} -> {}".format(v1_hit, v2_hit)

            f.write(f"| {r['tag']} | "
                    f"{arrow(r['v1_J'], r['v2_J'], r['v1_missing'], r['v2_missing'])} | "
                    f"{arrow(r['v1_I'], r['v2_I'], r['v1_missing'], r['v2_missing'])} | "
                    f"{arrow(r['v1_F'], r['v2_F'], r['v1_missing'], r['v2_missing'])} | "
                    f"{arrow(r['v1_C'], r['v2_C'], r['v1_missing'], r['v2_missing'])} | "
                    f"{arrow(r['v1_S'], r['v2_S'], r['v1_missing'], r['v2_missing'])} | "
                    f"{arrow(r['v1_len'], r['v2_len'], r['v1_missing'], r['v2_missing'])} | "
                    f"{hit_col} |\n")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(EVAL_DIR, "compare_v1v2p_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model":    BASE_MODEL,
            "v1_adapter":    V1_ADAPTER_DIR,
            "v2_adapter":    V2_ADAPTER_DIR,
            "sample_kwargs": SAMPLE_KWARGS,
            "best_of_n":     BEST_OF_N,
            "note": ("Offline finalize after interrupted job 1448/1455. "
                     "C4_v2 and C5_v2 are MISSING and excluded from averages."),
            "valid_case_count": len(valid),
            "v1_avg":        v1_avg,
            "v2_avg":        v2_avg,
            "delta_avg":     v2_avg - v1_avg,
            "v2_wins":       v2_wins,
            "ties":          tied,
            "v1_wins":       v1_wins,
            "missing":       [os.path.basename(p) for p in missing_files],
            "details": [
                {"tag": r["case"]["tag"], "case": r["case"],
                 "v1": {"score": r["score"], "present": r["code"] is not None}}
                for r in v1_results
            ] + [
                {"tag": r["case"]["tag"], "case": r["case"],
                 "v2": {"score": r["score"], "present": r["code"] is not None}}
                for r in v2_results
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nCSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
