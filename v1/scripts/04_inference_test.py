#!/usr/bin/env python3
"""
PocketVibe — 推理测试 + 自动质量评估
运行: python scripts/04_inference_test.py
输出: data/eval/test_*.html  +  data/eval/eval_report.json
"""
import os, json, re, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["HF_HOME"] = "/opt/shared/model-cache"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE_MODEL  = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR = os.path.expanduser("~/PocketVibe/outputs/qlora-run2/final_adapter")
EVAL_DIR    = Path(os.path.expanduser("~/PocketVibe/data/eval"))
EVAL_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# 10条测试指令：简单(1-3) / 中等(4-7) / 挑战(8-10)，均不在训练集中
TEST_PROMPTS = [
    # 简单
    "做一个硬币抛掷工具，正面反面各显示不同颜色",
    "帮我做个摄氏和开尔文温度换算器",
    "做一个简单的乘法练习，随机出题",
    # 中等
    "做一个体重指数追踪器，可以添加每天的体重，用列表显示历史记录",
    "做个每日步数记录工具，输入步数，显示离10000步目标还差多少",
    "帮我做一个读书记录工具，可以添加书名和阅读进度",
    "做个番茄工作法计时器，暗色主题，显示今天完成了几个番茄",
    # 挑战
    "做一个24小时作息规划表，每个时间段可以填写活动，用颜色区分工作休息娱乐",
    "帮我做个简易收支统计图，用柱状图显示本月每天的收支情况",
    "做一个单词记忆闪卡，有正面（英文）背面（中文），支持标记已掌握和不会",
]

# ── 加载模型（匹配训练时配置：bf16 全精度 LoRA）──
print(">>> 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(">>> 加载基座模型 (bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(">>> 挂载 LoRA 适配器 (run2)...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
print(">>> 模型加载完成\n")


def generate(instruction: str) -> tuple:
    """生成 HTML，返回 (代码, 耗时秒)"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,          # 确定性解码，生成最稳定的结果
            repetition_penalty=1.1,   # 防止重复输出
        )
    elapsed = time.time() - t0

    code = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # 截断到 </html> 防止尾部乱码
    end_idx = code.lower().rfind("</html>")
    if end_idx != -1:
        code = code[:end_idx + 7]

    return code, elapsed


def evaluate(code: str) -> dict:
    """5 项自动质量评估，返回评分字典（每项1分，满分5分）"""
    checks = {
        "html_complete":   code.strip().upper().startswith("<!DOCTYPE") and "</html>" in code.lower(),
        "has_viewport":    "viewport" in code,
        "mobile_friendly": bool(re.search(r"max-width|100%|100vw|flex|grid", code)),
        "no_external_cdn": not bool(re.search(r"cdn\.|unpkg\.com|jsdelivr|googleapis", code)),
        "has_gradient_ui": bool(re.search(r"linear-gradient|border-radius|box-shadow", code)),
    }
    score = sum(checks.values())
    checks["score"] = f"{score}/5"
    checks["score_int"] = score
    return checks


# ── 主循环 ──
results = []
total_score = 0

print(f"{'='*60}")
print(f"PocketVibe 推理评估 — 模型: qlora-run2")
print(f"{'='*60}\n")

for i, prompt in enumerate(TEST_PROMPTS):
    difficulty = "简单" if i < 3 else ("中等" if i < 7 else "挑战")
    print(f"[{i+1:02d}/{len(TEST_PROMPTS)}] [{difficulty}] {prompt}")

    code, elapsed = generate(prompt)
    metrics = evaluate(code)
    total_score += metrics["score_int"]

    # 保存 HTML 文件
    html_path = EVAL_DIR / f"test_{i+1:02d}.html"
    html_path.write_text(code, encoding="utf-8")

    # 打印摘要
    status_icons = {k: ("✅" if v else "❌") for k, v in metrics.items()
                    if k not in ("score", "score_int")}
    print(f"    耗时: {elapsed:.1f}s | 字符数: {len(code):,} | 评分: {metrics['score']}")
    print(f"    HTML完整:{status_icons['html_complete']} "
          f"viewport:{status_icons['has_viewport']} "
          f"移动端:{status_icons['mobile_friendly']} "
          f"无CDN:{status_icons['no_external_cdn']} "
          f"渐变UI:{status_icons['has_gradient_ui']}")
    print()

    results.append({
        "id": i + 1,
        "difficulty": difficulty,
        "instruction": prompt,
        "html_file": str(html_path),
        "char_count": len(code),
        "elapsed_sec": round(elapsed, 2),
        "metrics": metrics,
    })

# ── 汇总报告 ──
avg_score = total_score / len(TEST_PROMPTS)
pass_rate = sum(1 for r in results if r["metrics"]["score_int"] >= 4) / len(TEST_PROMPTS) * 100

report = {
    "model": "Qwen2.5-Coder-1.5B-Instruct + QLoRA run2",
    "adapter": ADAPTER_DIR,
    "test_count": len(TEST_PROMPTS),
    "avg_score": round(avg_score, 2),
    "pass_rate_4plus": f"{pass_rate:.0f}%",
    "total_score": f"{total_score}/{len(TEST_PROMPTS)*5}",
    "results": results,
}

report_path = EVAL_DIR / "eval_report.json"
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"{'='*60}")
print(f"✅ 评估完成!")
print(f"   测试数量: {len(TEST_PROMPTS)} 条")
print(f"   平均评分: {avg_score:.2f}/5")
print(f"   达标率(≥4分): {pass_rate:.0f}%")
print(f"   总分: {total_score}/{len(TEST_PROMPTS)*5}")
print(f"   HTML文件: {EVAL_DIR}/test_*.html")
print(f"   评估报告: {report_path}")
