#!/usr/bin/env python3
"""
PocketVibe V2+ — 04: V2 单模型细粒度推理测试
=================================================================
目标：验证 V2 LoRA 适配器在高难度 held-out 指令上的表现。
      （不与任何版本对比，只产出"V2 能做到什么程度"的证据）

对 V1 vs V2 的横向对比请运行: 05_eval_compare_v1_vs_v2_v2plus.py

评分维度（共 100 分，详见 pv_scoring.py）:
  J (30) = JS 代码深度
  I (25) = 指令遵循度（复杂题的每个功能点是否落地）
  F (20) = 功能可运行性（DOM/脚本闭合性）
  C (15) = CSS 质量 & 移动端适配
  S (10) = HTML 结构合规性

测试用例：5 条 V2 倾向的高难度指令（来自 pv_scoring.TEST_CASES）
  C1 DEPTH         - 分段秒表+最快/慢高亮+10段上限
  C2 BREADTH       - 游泳训练秒表（大字号+防水配色+距离估算）
  C3 REASONING     - 科学计算器（括号优先级+实时错误检测）
  C4 COMBINATION   - 待办+番茄钟融合
  C5 CROSS_CATEGORY- 石头剪刀布+实时记分板

运行:
    python scripts/04_inference_test_v2plus.py
输出:
    data/eval/v2plus_{tag}.html          # 5 份最佳 HTML（报告截图用）
    data/eval/v2plus_results.csv         # 细粒度分数表
    data/eval/v2plus_results.md          # Markdown 报告
    data/eval/v2plus_results.json        # 完整评分细节（含所有 breakdown）
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

# ---- HF 缓存 ----
os.environ["HF_HOME"]        = os.environ.get("HF_HOME", "/opt/shared/model-cache")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

HOME        = os.path.expanduser("~")
V2_ADAPTER  = os.path.join(HOME, "PocketVibe", "outputs", "qlora-v2-run1", "final_adapter")
EVAL_DIR    = os.path.join(HOME, "PocketVibe", "data", "eval")


def print_adapter_identity(adapter_dir: str, label: str):
    """打印 adapter_config.json 内容，方便日志里自证身份"""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        print(f"  [{label}] ⚠ 没有找到 adapter_config.json")
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"  [{label} 身份自证] adapter_config.json:")
    for k in ("r", "lora_alpha", "target_modules", "lora_dropout",
              "base_model_name_or_path", "task_type"):
        if k in cfg:
            print(f"       {k}: {cfg[k]}")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    if not os.path.isdir(V2_ADAPTER):
        raise SystemExit(f"❌ V2 adapter 不存在: {V2_ADAPTER}")

    print("=" * 64)
    print("PocketVibe V2+ — V2 单模型细粒度推理")
    print(f"  基座模型: {BASE_MODEL}")
    print(f"  V2 适配器: {V2_ADAPTER}")
    print(f"  采样: temp={SAMPLE_KWARGS['temperature']}, "
          f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
          f"Best-of-{BEST_OF_N}, max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}")
    print("=" * 64)

    print_adapter_identity(V2_ADAPTER, "V2")

    tokenizer = load_tokenizer()
    print("\n>>> 加载 V2 模型 (bf16 + LoRA)")
    model = load_model_with_adapter(V2_ADAPTER)

    rows, full_results = [], []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 64}")
        print(f"[{i}/{len(TEST_CASES)}] {case['tag']}  ({case['category']}, "
              f"{case['difficulty']})")
        print(f"指令: {case['instruction'][:80]}...")
        print(f"{'─' * 64}")

        best_code, all_attempts = generate_best_of_n(tokenizer, model, case, label="V2")
        best_score = score_html_100(best_code, case)

        # ---- 保存 HTML ----
        html_path = os.path.join(EVAL_DIR, f"v2plus_{case['tag']}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(best_code)

        # ---- 格式化打印 ----
        print(f"\n  ✅ 最佳得分: {best_score['total']}/100")
        print(f"      J (JS深度):      {best_score['J_js_depth']:2d}/30   "
              f"{best_score['breakdown']['J']}")
        print(f"      I (指令遵循):    {best_score['I_instruct']:2d}/25   "
              f"hit {best_score['breakdown']['I'].get('hit_count', 0)}/"
              f"{best_score['breakdown']['I'].get('total', 0)}")
        if best_score['breakdown']['I'].get('missed'):
            print(f"          未命中: {best_score['breakdown']['I']['missed']}")
        print(f"      F (可运行):      {best_score['F_functional']:2d}/20   "
              f"{best_score['breakdown']['F']}")
        print(f"      C (CSS质量):     {best_score['C_css']:2d}/15   "
              f"{best_score['breakdown']['C']}")
        print(f"      S (结构合规):    {best_score['S_structure']:2d}/10   "
              f"{best_score['breakdown']['S']}")
        print(f"      代码长度: {best_score['length']} 字符")
        print(f"      → HTML 已保存: {html_path}")

        rows.append({
            "tag":          case['tag'],
            "category":     case['category'],
            "difficulty":   case['difficulty'],
            "total":        best_score['total'],
            "J_js":         best_score['J_js_depth'],
            "I_instruct":   best_score['I_instruct'],
            "F_func":       best_score['F_functional'],
            "C_css":        best_score['C_css'],
            "S_struct":     best_score['S_structure'],
            "length":       best_score['length'],
            "i_hits":       best_score['breakdown']['I'].get('hit_count', 0),
            "i_total":      best_score['breakdown']['I'].get('total', 0),
        })
        full_results.append({
            "case":         case,
            "best_score":   best_score,
            "all_attempts": [
                {"mode": a["mode"], "total": a["score"]["total"],
                 "length": a["score"]["length"]}
                for a in all_attempts
            ],
        })

    # ---- 释放显存（下一个 job 准备加载 V1） ----
    del model
    torch.cuda.empty_cache()

    # ================================================================
    # 汇总
    # ================================================================
    avg = sum(r["total"] for r in rows) / len(rows)
    print(f"\n{'=' * 64}")
    print(f"📊 V2 单模型测试完成  |  平均分: {avg:.1f}/100")
    print(f"{'=' * 64}")
    for r in rows:
        print(f"  [{r['tag']:<32}] {r['total']:3d}/100 "
              f"(J{r['J_js']:2d} I{r['I_instruct']:2d} F{r['F_func']:2d} "
              f"C{r['C_css']:2d} S{r['S_struct']:2d})")

    # ---- CSV ----
    csv_path = os.path.join(EVAL_DIR, "v2plus_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Markdown ----
    md_path = os.path.join(EVAL_DIR, "v2plus_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PocketVibe V2+ — V2 单模型细粒度推理结果\n\n")
        f.write(f"**基座**: {BASE_MODEL}  |  **LoRA**: qlora-v2-run1\n\n")
        f.write(f"**推理**: Best-of-{BEST_OF_N}, temp={SAMPLE_KWARGS['temperature']}, "
                f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}\n\n")
        f.write(f"**平均分**: {avg:.1f}/100\n\n")
        f.write("| 用例 | 类别 | 总分 | J(30) | I(25) | F(20) | C(15) | S(10) | 长度 | I命中率 |\n")
        f.write("|------|------|------|-------|-------|-------|-------|-------|------|---------|\n")
        for r in rows:
            f.write(f"| {r['tag']} | {r['category']} | **{r['total']}** | "
                    f"{r['J_js']} | {r['I_instruct']} | {r['F_func']} | "
                    f"{r['C_css']} | {r['S_struct']} | {r['length']} | "
                    f"{r['i_hits']}/{r['i_total']} |\n")

    # ---- JSON（完整 breakdown） ----
    json_path = os.path.join(EVAL_DIR, "v2plus_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_model":    BASE_MODEL,
            "adapter":       V2_ADAPTER,
            "sample_kwargs": SAMPLE_KWARGS,
            "best_of_n":     BEST_OF_N,
            "avg_total":     avg,
            "results":       full_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 CSV:  {csv_path}")
    print(f"📄 MD:   {md_path}")
    print(f"📄 JSON: {json_path}")
    print(f"📄 HTMLs: {EVAL_DIR}/v2plus_*.html")


if __name__ == "__main__":
    main()
