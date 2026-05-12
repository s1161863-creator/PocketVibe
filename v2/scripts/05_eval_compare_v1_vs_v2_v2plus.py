#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
PocketVibe V2+ — 05: V1 vs V2 Fine-grained Comparison
=================================================================
对比对象（必须严格对齐身份，脚本启动会打印 adapter_config.json 供核对）:
  V1 = ~/PocketVibe/outputs/qlora-run1/final_adapter
       -> 交接版：PocketVibe PDF 指南原始训练脚本产物
       -> 对应本地 Enoch 仓库最早版本
  V2 = ~/PocketVibe/outputs/qlora-v2-run1/final_adapter
       -> 本工作：Version2 方法论训练（SLURM job 1442 成功产物）

评分维度（100 分制，详见 pv_scoring.py）:
  J(30) JS 深度 | I(25) 指令遵循 | F(20) 功能可运行 | C(15) CSS 质量 | S(10) 结构合规

测试集（5 条高难度 held-out 指令，覆盖 Evol-Instruct 四方向 + 跨类组合）
  与 04 脚本共用同一份 TEST_CASES，两版推理参数完全一致以保证公平。

运行:
    python scripts/05_eval_compare_v1_vs_v2_v2plus.py
输出:
    data/eval/compare_v1v2p_{tag}_v1.html / _v2.html
    data/eval/compare_v1v2p_results.csv
    data/eval/compare_v1v2p_results.md
    data/eval/compare_v1v2p_results.json
=================================================================
"""
import os, json, csv, torch

from pv_scoring import (
    BASE_MODEL, TEST_CASES,
    SAMPLE_KWARGS, BEST_OF_N,
    score_html_100,
    generate_best_of_n,
    load_tokenizer, load_model_with_adapter,
)

os.environ["HF_HOME"]        = os.environ.get("HF_HOME", "/opt/shared/model-cache")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

HOME           = os.path.expanduser("~")
V1_ADAPTER_DIR = os.path.join(HOME, "PocketVibe", "outputs", "qlora-run1",    "final_adapter")
V2_ADAPTER_DIR = os.path.join(HOME, "PocketVibe", "outputs", "qlora-v2-run1", "final_adapter")
EVAL_DIR       = os.path.join(HOME, "PocketVibe", "data", "eval")


def print_adapter_identity(adapter_dir: str, label: str):
    """打印 adapter_config.json，供日志核验身份"""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        print(f"  [{label}] WARN adapter_config.json missing! path: {cfg_path}")
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"  [{label} identity] {adapter_dir}")
    for k in ("r", "lora_alpha", "target_modules", "lora_dropout",
              "base_model_name_or_path", "task_type"):
        if k in cfg:
            print(f"       {k}: {cfg[k]}")


def run_single_version(adapter_dir: str, label: str, tokenizer):
    """把一版 LoRA 在 5 条指令上全部跑一遍，返回结果列表"""
    print(f"\n{'#' * 64}")
    print(f"# start eval {label}  ->  {adapter_dir}")
    print(f"{'#' * 64}")
    print_adapter_identity(adapter_dir, label)

    model = load_model_with_adapter(adapter_dir)
    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'-' * 55}")
        print(f"[{label}][{i}/{len(TEST_CASES)}] {case['tag']}")
        print(f"{'-' * 55}")
        best_code, _ = generate_best_of_n(tokenizer, model, case, label=label)
        score        = score_html_100(best_code, case)

        html_path = os.path.join(EVAL_DIR,
                                 f"compare_v1v2p_{case['tag']}_{label.lower()}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(best_code)
        print(f"  -> saved HTML: {html_path}")

        results.append({"case": case, "code": best_code, "score": score})

    del model
    torch.cuda.empty_cache()
    return results


def fmt_delta(d):
    if d > 0:  return f"+{d}"
    if d < 0:  return f"{d}"
    return "+-0"


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    for name, path in [("V1", V1_ADAPTER_DIR), ("V2", V2_ADAPTER_DIR)]:
        if not os.path.isdir(path):
            raise SystemExit(f"[ERROR] {name} adapter not found: {path}")

    print("=" * 64)
    print("PocketVibe V2+ -- V1 vs V2 Fine-grained Comparison")
    print(f"  base model: {BASE_MODEL}")
    print(f"  V1 = qlora-run1    (original / aligned with local Enoch repo)")
    print(f"  V2 = qlora-v2-run1 (Version2 new training / SLURM 1442 output)")
    print(f"  sampling: temp={SAMPLE_KWARGS['temperature']}, "
          f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
          f"Best-of-{BEST_OF_N}, max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}")
    print("=" * 64)

    tokenizer = load_tokenizer()

    # 顺序跑：先 V1 释放后再 V2，避免双 LoRA 撑爆显存
    v1_results = run_single_version(V1_ADAPTER_DIR, "V1", tokenizer)
    v2_results = run_single_version(V2_ADAPTER_DIR, "V2", tokenizer)

    # ==============================================================
    # 对比分析
    # ==============================================================
    rows = []
    for v1_r, v2_r in zip(v1_results, v2_results):
        case = v1_r["case"]
        v1_s = v1_r["score"]
        v2_s = v2_r["score"]
        rows.append({
            "tag":           case["tag"],
            "category":      case["category"],
            "instruction":   case["instruction"][:50],
            "v1_total":      v1_s["total"],
            "v2_total":      v2_s["total"],
            "delta_total":   v2_s["total"] - v1_s["total"],
            "v1_J":          v1_s["J_js_depth"],
            "v2_J":          v2_s["J_js_depth"],
            "v1_I":          v1_s["I_instruct"],
            "v2_I":          v2_s["I_instruct"],
            "v1_F":          v1_s["F_functional"],
            "v2_F":          v2_s["F_functional"],
            "v1_C":          v1_s["C_css"],
            "v2_C":          v2_s["C_css"],
            "v1_S":          v1_s["S_structure"],
            "v2_S":          v2_s["S_structure"],
            "v1_len":        v1_s["length"],
            "v2_len":        v2_s["length"],
            "v1_i_hits":     v1_s["breakdown"]["I"].get("hit_count", 0),
            "v2_i_hits":     v2_s["breakdown"]["I"].get("hit_count", 0),
            "i_total_pts":   v1_s["breakdown"]["I"].get("total",     0),
        })

    # 控制台总表
    print(f"\n{'=' * 80}")
    print("V1 vs V2 COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'case':<32} {'V1':>6} {'V2':>6} {'delta':>6}  dJ dI dF dC dS")
    print("-" * 80)
    for r in rows:
        dJ = r["v2_J"] - r["v1_J"]
        dI = r["v2_I"] - r["v1_I"]
        dF = r["v2_F"] - r["v1_F"]
        dC = r["v2_C"] - r["v1_C"]
        dS = r["v2_S"] - r["v1_S"]
        print(f"{r['tag']:<32} {r['v1_total']:>3}/100 {r['v2_total']:>3}/100 "
              f"{fmt_delta(r['delta_total']):>6}  "
              f"{fmt_delta(dJ):>3} {fmt_delta(dI):>3} {fmt_delta(dF):>3} "
              f"{fmt_delta(dC):>3} {fmt_delta(dS):>3}")

    v1_avg = sum(r["v1_total"] for r in rows) / len(rows)
    v2_avg = sum(r["v2_total"] for r in rows) / len(rows)
    v2_wins = sum(1 for r in rows if r["delta_total"] > 0)
    tied    = sum(1 for r in rows if r["delta_total"] == 0)
    v1_wins = sum(1 for r in rows if r["delta_total"] < 0)
    print("-" * 80)
    print(f"avg:  V1 = {v1_avg:.1f}/100   V2 = {v2_avg:.1f}/100   "
          f"delta = {v2_avg - v1_avg:+.1f}")
    print(f"wins: V2={v2_wins}  tie={tied}  V1={v1_wins}")

    # ---- CSV ----
    csv_path = os.path.join(EVAL_DIR, "compare_v1v2p_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Markdown ----
    md_path = os.path.join(EVAL_DIR, "compare_v1v2p_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PocketVibe — V1 vs V2 Fine-grained Comparison (100-pt)\n\n")
        f.write(f"- **Base**: {BASE_MODEL}\n")
        f.write("- **V1**: `outputs/qlora-run1/final_adapter` (original / aligned with local Enoch repo)\n")
        f.write("- **V2**: `outputs/qlora-v2-run1/final_adapter` (Version2 new training)\n")
        f.write(f"- **Inference**: Best-of-{BEST_OF_N}, temp={SAMPLE_KWARGS['temperature']}, "
                f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
                f"max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}\n\n")
        f.write(f"**Avg**: V1 = {v1_avg:.1f}/100  |  V2 = {v2_avg:.1f}/100  "
                f"|  delta = {v2_avg - v1_avg:+.1f}\n\n")
        f.write(f"**Wins**: V2={v2_wins} | tie={tied} | V1={v1_wins}\n\n")

        f.write("## Total score comparison\n\n")
        f.write("| Case | Category | V1 | V2 | delta |\n")
        f.write("|------|----------|----|----|----|\n")
        for r in rows:
            f.write(f"| {r['tag']} | {r['category']} | {r['v1_total']}/100 | "
                    f"**{r['v2_total']}/100** | {fmt_delta(r['delta_total'])} |\n")

        f.write("\n## Dimension breakdown (V1 -> V2)\n\n")
        f.write("| Case | J(30) | I(25) | F(20) | C(15) | S(10) | length | I hit-rate |\n")
        f.write("|------|-------|-------|-------|-------|-------|--------|------------|\n")
        for r in rows:
            f.write(f"| {r['tag']} | "
                    f"{r['v1_J']}->{r['v2_J']} | "
                    f"{r['v1_I']}->{r['v2_I']} | "
                    f"{r['v1_F']}->{r['v2_F']} | "
                    f"{r['v1_C']}->{r['v2_C']} | "
                    f"{r['v1_S']}->{r['v2_S']} | "
                    f"{r['v1_len']}->{r['v2_len']} | "
                    f"{r['v1_i_hits']}/{r['i_total_pts']} -> "
                    f"{r['v2_i_hits']}/{r['i_total_pts']} |\n")

    # ---- JSON ----
    json_path = os.path.join(EVAL_DIR, "compare_v1v2p_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model":    BASE_MODEL,
            "v1_adapter":    V1_ADAPTER_DIR,
            "v2_adapter":    V2_ADAPTER_DIR,
            "sample_kwargs": SAMPLE_KWARGS,
            "best_of_n":     BEST_OF_N,
            "v1_avg":        v1_avg,
            "v2_avg":        v2_avg,
            "delta_avg":     v2_avg - v1_avg,
            "v2_wins":       v2_wins,
            "ties":          tied,
            "v1_wins":       v1_wins,
            "details": [
                {"tag": r["case"]["tag"], "case": r["case"], "v1": {"score": r["score"]}}
                for r in v1_results
            ] + [
                {"tag": r["case"]["tag"], "case": r["case"], "v2": {"score": r["score"]}}
                for r in v2_results
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nCSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print(f"JSON: {json_path}")
    print(f"HTMLs: {EVAL_DIR}/compare_v1v2p_*.html  ({len(rows)*2} files)")


if __name__ == "__main__":
    main()
