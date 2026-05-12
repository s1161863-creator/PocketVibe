#!/usr/bin/env python3
"""
PocketVibe V2 — 05: 微调前后对比评估（优化版 2026.05）
=================================================================
优化项（与 04_inference_test.py 同步）:
  1. 官方参数: temperature=0.7, top_p=0.8, top_k=20
  2. 移除 repetition_penalty（代码生成场景禁用）
  3. Best-of-3 采样: 原版/微调版各自独立采 3 次选优
  4. 代码清洗: 去 markdown 包裹 / 截取真正的 HTML
  5. 贪心兜底: 3 次采样均失败时用 do_sample=False
  6. 输出详细对比 CSV（供报告直接粘贴）+ 逐条 HTML 对比文件

对比维度（报告得分项）:
  - 可运行性 (runnable)
  - Viewport 移动端适配 (viewport)
  - 响应式布局 (mobile)
  - 无外部依赖 (no_cdn)
  - 有 style 标签 (has_style)
  - 有 script 标签 (has_script)
  - 综合质量得分 (score/6)
  - 代码长度 (length)

运行（HPC 上）：python scripts/05_eval_compare.py
输出：data/eval/compare_results.csv  +  data/eval/compare_*.html
=================================================================
"""
import os, json, re, csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_HOME"]        = os.environ.get("HF_HOME", "/opt/shared/model-cache")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

HOME        = os.path.expanduser("~")
BASE_MODEL  = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR = os.path.join(HOME, "PocketVibe_v2", "outputs", "qlora-v2-run1", "final_adapter")
EVAL_DIR    = os.path.join(HOME, "PocketVibe_v2", "data", "eval")

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# ================================================================
# 推理参数（与 04 保持一致）
# ================================================================
SAMPLE_KWARGS = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
    repetition_penalty=1.0,
    max_new_tokens=2048,
)
GREEDY_KWARGS = dict(
    do_sample=False,
    max_new_tokens=2048,
    repetition_penalty=1.0,
)
BEST_OF_N = 3

# ================================================================
# 对比测试指令（5 条，需要与训练集有重叠以证明学到了，
# 同时也包含新的变种以评估泛化）
# ================================================================
# 选择策略：
#   3 条"域内"（类似训练集的简单指令）→ 证明微调有效
#   2 条"域外"（新类别/复杂描述）       → 评估泛化能力
TEST_INSTRUCTIONS = [
    # ----- 域内指令（训练集覆盖类型）-----
    ("in_01",  "做一个秒表，有开始停止和清零功能"),
    ("in_02",  "帮我做个掷骰子工具，点击按钮随机出1到6的点数"),
    ("in_03",  "做一个简单的计算器，支持加减乘除"),
    # ----- 域外指令（新类别 / 复杂表述）-----
    ("out_01", "做一个颜色随机生成器，每次点击显示一个随机颜色并可以复制hex值"),
    ("out_02", "帮我做个单位换算工具，厘米和英寸可以互相换算，粉色可爱风格"),
]


# ================================================================
# 工具函数（与 04 完全一致，便于 HPC 独立运行）
# ================================================================

def clean_code(raw: str) -> str:
    code = raw.strip()
    if code.startswith("```"):
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code, count=1)
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    code = code.strip()
    idx = code.upper().find("<!DOCTYPE")
    if idx > 0:
        code = code[idx:]
    end_idx = code.lower().rfind("</html>")
    if end_idx != -1:
        code = code[:end_idx + len("</html>")]
    return code.strip()


def score_html(code: str) -> int:
    score = 0
    c = code.lower()
    if "<!doctype" in c:
        score += 1
    if "</html>" in c:
        score += 1
    if "viewport" in c:
        score += 1
    if "max-width" in c or "100%" in c or "100vw" in c:
        score += 1
    if "cdn." not in c and "unpkg" not in c and "jsdelivr" not in c:
        score += 1
    if 300 <= len(code) <= 8000:
        score += 1
    return score


def check_html(code: str) -> dict:
    c = code.lower()
    return {
        "runnable":   "<!doctype" in c and "</html>" in c,
        "viewport":   "viewport" in c,
        "mobile":     any(k in c for k in ["max-width", "100%", "100vw"]),
        "no_cdn":     "cdn." not in c and "unpkg" not in c and "jsdelivr" not in c,
        "has_style":  "<style" in c,
        "has_script": "<script" in c,
        "score":      score_html(code),
        "length":     len(code),
    }


# ================================================================
# 模型加载
# ================================================================

def load_base_model(tokenizer):
    print(f"  >>> 加载基础模型（原版，不加 LoRA）...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_finetuned_model(tokenizer):
    print(f"  >>> 加载基础模型...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  >>> 挂载 LoRA 适配器：{ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return model


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ================================================================
# 推理（Best-of-N + 贪心兜底，与 04 相同逻辑）
# ================================================================

def generate_one(tokenizer, model, instruction: str, gen_kwargs: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_code(raw)


def generate_best(tokenizer, model, instruction: str, label: str = "") -> str:
    """Best-of-N + 贪心兜底，返回最佳 HTML 代码"""
    candidates = []
    for i in range(BEST_OF_N):
        code = generate_one(tokenizer, model, instruction, SAMPLE_KWARGS)
        candidates.append(code)
        sc = score_html(code)
        print(f"      [{label} 采样 {i+1}/{BEST_OF_N}] score={sc} len={len(code)}")
        if sc == 6:
            break

    best = max(candidates, key=score_html)
    if score_html(best) < 4:
        print(f"      [{label} 贪心兜底] ...")
        greedy = generate_one(tokenizer, model, instruction, GREEDY_KWARGS)
        if score_html(greedy) >= score_html(best):
            best = greedy
        print(f"      [{label} 贪心兜底] score={score_html(best)}")

    return best


# ================================================================
# 主程序
# ================================================================

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    tokenizer = load_tokenizer()

    print("\n" + "=" * 60)
    print("📊 PocketVibe V2 — 微调前后对比评估")
    print(f"   参数: temperature={SAMPLE_KWARGS['temperature']}, "
          f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}")
    print(f"   Best-of-N={BEST_OF_N}  |  代码清洗: 启用  |  贪心兜底: 启用")
    print("=" * 60)

    rows = []

    # ---- 对每条指令先用原版模型，再用微调模型 ----
    for tag, instruction in TEST_INSTRUCTIONS:
        print(f"\n{'─'*55}")
        print(f"[{tag}] {instruction}")
        print(f"{'─'*55}")

        # 1) 原版模型
        print("  [原版模型]")
        base_model = load_base_model(tokenizer)
        base_code  = generate_best(tokenizer, base_model, instruction, label="原版")
        base_chk   = check_html(base_code)
        # 节省显存：生成完立即释放原版模型
        del base_model
        torch.cuda.empty_cache()

        # 2) 微调模型
        print("  [微调模型]")
        ft_model  = load_finetuned_model(tokenizer)
        ft_code   = generate_best(tokenizer, ft_model, instruction, label="微调")
        ft_chk    = check_html(ft_code)
        del ft_model
        torch.cuda.empty_cache()

        # 3) 保存 HTML（供报告截图）
        base_path = os.path.join(EVAL_DIR, f"compare_{tag}_base.html")
        ft_path   = os.path.join(EVAL_DIR, f"compare_{tag}_ft.html")
        with open(base_path, "w", encoding="utf-8") as f:
            f.write(base_code)
        with open(ft_path, "w", encoding="utf-8") as f:
            f.write(ft_code)

        # 4) 打印对比
        print(f"\n  ┌{'─'*45}┐")
        print(f"  │{'指标':<12} {'原版':>10}  {'微调':>10}  {'改进':>8}│")
        print(f"  ├{'─'*45}┤")
        metrics = ["runnable", "viewport", "mobile", "no_cdn", "has_style", "has_script"]
        for m in metrics:
            bv = "✅" if base_chk[m] else "❌"
            fv = "✅" if ft_chk[m]   else "❌"
            imp = "⬆" if (not base_chk[m] and ft_chk[m]) else ("⬇" if (base_chk[m] and not ft_chk[m]) else " ")
            print(f"  │  {m:<14} {bv:>6}     {fv:>6}     {imp:>4}  │")
        print(f"  │  {'质量得分':<14} {base_chk['score']:>6}/6   {ft_chk['score']:>6}/6         │")
        print(f"  │  {'代码长度':<14} {base_chk['length']:>6}     {ft_chk['length']:>6}         │")
        print(f"  └{'─'*45}┘")

        rows.append({
            "tag":             tag,
            "instruction":     instruction,
            **{f"base_{k}": v for k, v in base_chk.items()},
            **{f"ft_{k}":   v for k, v in ft_chk.items()},
            "score_delta":     ft_chk["score"] - base_chk["score"],
            "length_delta":    ft_chk["length"] - base_chk["length"],
        })

    # ================================================================
    # 输出汇总 CSV（直接粘贴到报告 Table）
    # ================================================================
    csv_path = os.path.join(EVAL_DIR, "compare_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ================================================================
    # 输出 Markdown 表格（直接粘贴到报告正文）
    # ================================================================
    md_path = os.path.join(EVAL_DIR, "compare_results.md")
    metrics = ["runnable", "viewport", "mobile", "no_cdn", "has_style", "has_script", "score", "length"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PocketVibe V2 — 微调前后对比\n\n")
        f.write(f"推理参数: temperature={SAMPLE_KWARGS['temperature']}, "
                f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
                f"Best-of-{BEST_OF_N}\n\n")
        f.write("| 指令 | 指标 | 原版模型 | 微调模型 | 改进 |\n")
        f.write("|------|------|---------|---------|------|\n")
        for row in rows:
            for m in metrics:
                bv = row.get(f"base_{m}")
                fv = row.get(f"ft_{m}")
                if isinstance(bv, bool):
                    bstr = "✅" if bv else "❌"
                    fstr = "✅" if fv else "❌"
                    imp  = "⬆" if (not bv and fv) else ("⬇" if (bv and not fv) else "–")
                else:
                    bstr = str(bv)
                    fstr = str(fv)
                    imp  = f"{fv - bv:+d}" if isinstance(fv, int) and isinstance(bv, int) else "–"
                f.write(f"| {row['instruction'][:18]} | {m} | {bstr} | {fstr} | {imp} |\n")
            f.write("|\n")

    # ================================================================
    # 打印最终汇总
    # ================================================================
    print(f"\n{'='*60}")
    print(f"📊 对比评估完成！共 {len(rows)} 条指令")

    base_avg = sum(r["base_score"] for r in rows) / len(rows)
    ft_avg   = sum(r["ft_score"]   for r in rows) / len(rows)
    print(f"   原版模型平均质量得分: {base_avg:.2f}/6")
    print(f"   微调模型平均质量得分: {ft_avg:.2f}/6")
    print(f"   平均提升: {ft_avg - base_avg:+.2f}")

    improved  = sum(1 for r in rows if r["score_delta"] > 0)
    unchanged = sum(1 for r in rows if r["score_delta"] == 0)
    degraded  = sum(1 for r in rows if r["score_delta"] < 0)
    print(f"   提升: {improved}条  持平: {unchanged}条  下降: {degraded}条")

    print(f"\n📄 CSV 报告: {csv_path}")
    print(f"📄 Markdown 报告: {md_path}")
    print(f"📄 HTML 对比文件: {EVAL_DIR}/compare_*.html")
    print(f"\n⏭️  下一步：python scripts/07_plot_loss.py")


if __name__ == "__main__":
    main()
