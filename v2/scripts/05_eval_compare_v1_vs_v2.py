#!/usr/bin/env python3
"""
PocketVibe — 05 (v1_vs_v2): 两版 LoRA 适配器对比评估
=================================================================
本脚本加载同一个基座模型 (Qwen2.5-Coder-1.5B-Instruct)，
分别挂载 V1 / V2 两个 LoRA 适配器，使用相同的推理参数、
相同的测试指令，输出可以直接粘贴到报告的对比表。

对比维度（report Section 3 得分项）:
  - runnable   : HTML 是否可直接运行 (<!DOCTYPE + </html>)
  - viewport   : 是否包含 <meta name=viewport>
  - mobile     : 是否有响应式布局关键字
  - no_cdn     : 是否脱离所有外部依赖
  - has_style  : 是否存在 <style> 块
  - has_script : 是否存在 <script> 块（未被截断）
  - score/6    : 综合质量分
  - length     : 代码字符数

运行:
    python scripts/05_eval_compare_v1_vs_v2.py
输出:
    data/eval/compare_v1v2_results.csv
    data/eval/compare_v1v2_results.md
    data/eval/compare_v1v2_<tag>_v1.html
    data/eval/compare_v1v2_<tag>_v2.html
=================================================================
"""
import os, json, re, csv, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_HOME"]        = os.environ.get("HF_HOME", "/opt/shared/model-cache")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

HOME         = os.path.expanduser("~")
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# ---- 两个 LoRA 适配器路径 ----
# V1: 项目初版交接的 LoRA 适配器
V1_ADAPTER_DIR = os.path.join(HOME, "PocketVibe", "outputs", "qlora-run1",    "final_adapter")
# V2: 本工作训练的 LoRA 适配器
V2_ADAPTER_DIR = os.path.join(HOME, "PocketVibe", "outputs", "qlora-v2-run1", "final_adapter")

EVAL_DIR       = os.path.join(HOME, "PocketVibe", "data", "eval")

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# ================================================================
# 推理参数（两版保持完全一致以保证公平）
# ================================================================
SAMPLE_KWARGS = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
    repetition_penalty=1.0,
    max_new_tokens=3072,   # 统一预算，避免因截断影响评分
)
GREEDY_KWARGS = dict(
    do_sample=False,
    max_new_tokens=3072,
    repetition_penalty=1.0,
)
BEST_OF_N = 3

# ================================================================
# 对比测试指令
# 选择原则：均为 PocketVibe 项目指南中列出的基础 / 中等难度工具，
#          两版 LoRA 的训练数据都覆盖此类任务。
# ================================================================
TEST_INSTRUCTIONS = [
    ("t1_stopwatch",   "做一个秒表，有开始停止和清零功能"),
    ("t2_pomodoro",    "做一个番茄钟，25分钟工作5分钟休息，自动切换"),
    ("t3_todo",        "做一个待办事项列表，可以添加勾选和删除"),
    ("t4_password",    "做一个随机密码生成器，可以调节长度"),
    ("t5_guess_num",   "做一个猜数字游戏，1到100之间猜"),
]


# ================================================================
# 工具函数
# ================================================================

def clean_code(raw: str) -> str:
    """去除 markdown 包裹 + 截取真正的 HTML 段。"""
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
    if 300 <= len(code) <= 12000:
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
        "has_script": "<script" in c and "</script>" in c,
        "score":      score_html(code),
        "length":     len(code),
    }


# ================================================================
# 模型加载
# ================================================================

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_with_adapter(adapter_dir: str):
    """加载基座 + 挂载指定 LoRA 适配器（两版走完全相同的加载路径）。"""
    print(f"  >>> 加载基础模型 (bf16) ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  >>> 挂载 LoRA 适配器: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


# ================================================================
# 推理（Best-of-N + 贪心兜底）
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
    """Best-of-N 采样 + 贪心兜底，返回最佳 HTML 代码。"""
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
        print(f"      [{label} 贪心兜底完成] score={score_html(best)}")

    return best


# ================================================================
# 主程序
# ================================================================

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    # 前置检查：两个 adapter 必须都存在
    for name, path in [("V1", V1_ADAPTER_DIR), ("V2", V2_ADAPTER_DIR)]:
        if not os.path.isdir(path):
            raise SystemExit(f"❌ {name} adapter 不存在: {path}")
        else:
            print(f"✅ 找到 {name} adapter: {path}")

    tokenizer = load_tokenizer()

    print("\n" + "=" * 60)
    print("📊 PocketVibe — V1 vs V2 两版 LoRA 适配器对比")
    print(f"   基座: {BASE_MODEL}")
    print(f"   采样: temperature={SAMPLE_KWARGS['temperature']}, "
          f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}")
    print(f"   Best-of-N={BEST_OF_N}  |  max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}")
    print("=" * 60)

    rows = []

    for tag, instruction in TEST_INSTRUCTIONS:
        print(f"\n{'─'*55}")
        print(f"[{tag}] {instruction}")
        print(f"{'─'*55}")

        # ---- V1 ----
        print("  [V1 适配器]")
        v1_model = load_model_with_adapter(V1_ADAPTER_DIR)
        v1_code  = generate_best(tokenizer, v1_model, instruction, label="V1")
        v1_chk   = check_html(v1_code)
        del v1_model
        torch.cuda.empty_cache()

        # ---- V2 ----
        print("  [V2 适配器]")
        v2_model = load_model_with_adapter(V2_ADAPTER_DIR)
        v2_code  = generate_best(tokenizer, v2_model, instruction, label="V2")
        v2_chk   = check_html(v2_code)
        del v2_model
        torch.cuda.empty_cache()

        # ---- 保存 HTML（供报告截图） ----
        v1_path = os.path.join(EVAL_DIR, f"compare_v1v2_{tag}_v1.html")
        v2_path = os.path.join(EVAL_DIR, f"compare_v1v2_{tag}_v2.html")
        with open(v1_path, "w", encoding="utf-8") as f: f.write(v1_code)
        with open(v2_path, "w", encoding="utf-8") as f: f.write(v2_code)

        # ---- 打印对比表 ----
        print(f"\n  ┌{'─'*45}┐")
        print(f"  │{'指标':<12} {'V1':>10}  {'V2':>10}  {'变化':>8}│")
        print(f"  ├{'─'*45}┤")
        bool_metrics = ["runnable", "viewport", "mobile", "no_cdn", "has_style", "has_script"]
        for m in bool_metrics:
            b1 = "✅" if v1_chk[m] else "❌"
            b2 = "✅" if v2_chk[m] else "❌"
            if not v1_chk[m] and v2_chk[m]:
                delta = "⬆"
            elif v1_chk[m] and not v2_chk[m]:
                delta = "⬇"
            else:
                delta = "–"
            print(f"  │  {m:<14} {b1:>6}     {b2:>6}     {delta:>4}  │")
        print(f"  │  {'质量得分':<14} {v1_chk['score']:>6}/6   {v2_chk['score']:>6}/6      {v2_chk['score']-v1_chk['score']:+d}  │")
        print(f"  │  {'代码长度':<14} {v1_chk['length']:>6}     {v2_chk['length']:>6}     {v2_chk['length']-v1_chk['length']:+d}│")
        print(f"  └{'─'*45}┘")

        rows.append({
            "tag":          tag,
            "instruction":  instruction,
            **{f"v1_{k}": v for k, v in v1_chk.items()},
            **{f"v2_{k}": v for k, v in v2_chk.items()},
            "score_delta":  v2_chk["score"]  - v1_chk["score"],
            "length_delta": v2_chk["length"] - v1_chk["length"],
        })

    # ================================================================
    # CSV
    # ================================================================
    csv_path = os.path.join(EVAL_DIR, "compare_v1v2_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # ================================================================
    # Markdown
    # ================================================================
    md_path = os.path.join(EVAL_DIR, "compare_v1v2_results.md")
    metrics = ["runnable", "viewport", "mobile", "no_cdn", "has_style", "has_script", "score", "length"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PocketVibe — V1 vs V2 两版 LoRA 对比\n\n")
        f.write(f"推理参数: temperature={SAMPLE_KWARGS['temperature']}, "
                f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
                f"Best-of-{BEST_OF_N}, max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}\n\n")
        f.write("| 指令 | 指标 | V1 | V2 | 变化 |\n")
        f.write("|------|------|----|----|------|\n")
        for row in rows:
            for m in metrics:
                b1 = row.get(f"v1_{m}")
                b2 = row.get(f"v2_{m}")
                if isinstance(b1, bool):
                    s1 = "✅" if b1 else "❌"
                    s2 = "✅" if b2 else "❌"
                    delta = "⬆" if (not b1 and b2) else ("⬇" if (b1 and not b2) else "–")
                else:
                    s1 = str(b1); s2 = str(b2)
                    delta = f"{b2 - b1:+d}" if isinstance(b1, int) and isinstance(b2, int) else "–"
                f.write(f"| {row['instruction'][:18]} | {m} | {s1} | {s2} | {delta} |\n")

    # ================================================================
    # 汇总统计
    # ================================================================
    print(f"\n{'='*60}")
    print(f"📊 V1 vs V2 对比完成！共 {len(rows)} 条指令")

    v1_avg = sum(r["v1_score"] for r in rows) / len(rows)
    v2_avg = sum(r["v2_score"] for r in rows) / len(rows)
    print(f"   V1 平均质量得分: {v1_avg:.2f}/6")
    print(f"   V2 平均质量得分: {v2_avg:.2f}/6")
    print(f"   平均变化: {v2_avg - v1_avg:+.2f}")

    v2_better = sum(1 for r in rows if r["score_delta"] > 0)
    tied      = sum(1 for r in rows if r["score_delta"] == 0)
    v1_better = sum(1 for r in rows if r["score_delta"] < 0)
    print(f"   V2 胜: {v2_better}  持平: {tied}  V1 胜: {v1_better}")

    print(f"\n📄 CSV:      {csv_path}")
    print(f"📄 Markdown: {md_path}")
    print(f"📄 HTML:     {EVAL_DIR}/compare_v1v2_*.html")


if __name__ == "__main__":
    main()
