#!/usr/bin/env python3
"""
PocketVibe — 微调前后对比评估 v2（10分制多维度评分）
================================================================
评分维度说明:
  1. html_structure    (1分) HTML结构完整性
  2. no_markdown_wrap  (1分) 直接输出代码，不包markdown代码块
  3. viewport_meta     (1分) 包含移动端viewport标签
  4. pocketvibe_style  (2分) PocketVibe风格匹配（card布局+渐变+圆角+emoji等）
  5. js_functional     (2分) JavaScript功能完整性（事件绑定+函数定义数量）
  6. no_external_dep   (1分) 无外部CDN/库依赖
  7. code_efficiency   (1分) 代码长度合理（200~5000字符）
  8. chinese_ui        (1分) 中文界面（中文字符占比>5%）

总分: 10分
原版模型预期: 5-7分（结构对但风格/格式不符）
微调模型预期: 8-10分（风格统一、格式规范）

运行: python scripts/05_eval_compare.py
输出:
  data/eval/compare_results_v2.csv
  data/eval/compare_report_v2.json
  data/eval/base_v2_*.html
  data/eval/ft_v2_*.html
"""

import os, json, re, csv, torch, time
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

# ================================================================
# 10条对比测试指令（不在训练集中，覆盖多种表达方式）
# ================================================================
TEST_INSTRUCTIONS = [
    # 简单直接型 (3条)
    "做一个秒表，有开始、停止和清零按钮",
    "弄个随机颜色生成器，要能复制颜色值",
    "搞个摄氏度华氏度互转的工具",
    # 口语化/模糊型 (3条) — 原版模型最容易翻车的场景
    "我要个能记事的小本本",
    "做个小游戏玩玩，就随便一个",
    "帮我搞个倒计时，暗色的那种",
    # 中等复杂 + 视觉要求 (2条)
    "做一个待办事项，粉色可爱风格，可以勾选和删除",
    "帮我做一个BMI计算器，结果要显示健康建议",
    # 挑战级 (2条)
    "做一个简易记账本，要分收入和支出，能看到余额",
    "做一个番茄钟，暗色主题，自动在工作和休息之间切换，记录今日完成数",
]

# ================================================================
# 评分函数（满分10分）
# ================================================================
def evaluate(code: str) -> dict:
    """
    10分制多维度评分，每项均有明确检测逻辑
    返回: {维度名: 分值, ..., "total": 总分, "total_str": "X/10"}
    """
    scores = {}

    # ── 1. HTML结构完整性 (1分) ──────────────────────────────────
    # 必须以<!DOCTYPE html>开头且有</html>闭合
    scores["html_structure"] = int(
        code.strip().upper().startswith("<!DOCTYPE") and
        "</html>" in code.lower()
    )

    # ── 2. 无Markdown包裹 (1分) ──────────────────────────────────
    # 原版模型常把代码包在 ```html ... ``` 中，微调后不会
    scores["no_markdown_wrap"] = int(
        not code.strip().startswith("```") and
        not code.strip().startswith("~~~")
    )

    # ── 3. Viewport移动端标签 (1分) ──────────────────────────────
    scores["viewport_meta"] = int("viewport" in code)

    # ── 4. PocketVibe风格匹配度 (0/1/2分) ────────────────────────
    # 检查6个风格特征，命中≥5个得2分，命中3-4个得1分，<3个得0分
    style_checks = [
        bool(re.search(r"linear-gradient", code)),           # 渐变背景
        bool(re.search(r"border-radius\s*:\s*1[2-9]|border-radius\s*:\s*2\d", code)),  # 圆角≥12px
        bool(re.search(r"box-shadow", code)),                # 阴影
        bool(re.search(r"max-width\s*:\s*3[6-9]\d|max-width\s*:\s*4\d\d", code)),     # card宽度360~480px
        bool(re.search(r"[\U0001F300-\U0001FFFF]|[\u2600-\u27FF]", code)),           # emoji标题装饰
        bool(re.search(r"class=[\"']card[\"']|\.card\s*{", code)),                   # card类名
    ]
    hit = sum(style_checks)
    scores["pocketvibe_style"] = 2 if hit >= 5 else (1 if hit >= 3 else 0)

    # ── 5. JavaScript功能完整性 (0/1/2分) ────────────────────────
    # 检查: 有script标签 + 事件绑定数量 + function定义数量
    has_script   = "<script" in code.lower()
    event_count  = len(re.findall(r"onclick|oninput|onchange|addEventListener", code))
    func_count   = len(re.findall(r"function\s+\w+\s*\(", code))
    if not has_script:
        scores["js_functional"] = 0
    elif event_count >= 2 and func_count >= 2:
        scores["js_functional"] = 2   # 有多个事件绑定和函数：功能完整
    elif event_count >= 1 or func_count >= 1:
        scores["js_functional"] = 1   # 有JS但偏简单
    else:
        scores["js_functional"] = 0   # script存在但几乎是空的

    # ── 6. 无外部CDN依赖 (1分) ───────────────────────────────────
    scores["no_external_dep"] = int(
        not bool(re.search(r"cdn\.|unpkg\.com|jsdelivr|googleapis|bootstrap|jquery", code, re.I))
    )

    # ── 7. 代码长度合理 (1分) ────────────────────────────────────
    # 200字符以下是骨架，5000字符以上通常是废话重复
    scores["code_efficiency"] = int(200 <= len(code) <= 5000)

    # ── 8. 中文界面 (1分) ────────────────────────────────────────
    # 中文字符数 / 总字符数 > 3%（避免全英文输出）
    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", code))
    scores["chinese_ui"] = int(cn_chars / max(len(code), 1) > 0.03)

    # ── 总分 ─────────────────────────────────────────────────────
    total = sum(scores.values())
    scores["total"] = total
    scores["total_str"] = f"{total}/10"
    return scores


def dim_label(key: str) -> str:
    """评分维度的中文标签"""
    labels = {
        "html_structure":   "HTML完整",
        "no_markdown_wrap": "无MD包裹",
        "viewport_meta":    "Viewport",
        "pocketvibe_style": "风格匹配",
        "js_functional":    "JS功能",
        "no_external_dep":  "无外部CDN",
        "code_efficiency":  "长度合理",
        "chinese_ui":       "中文界面",
    }
    return labels.get(key, key)


# ================================================================
# 加载模型
# ================================================================
print(">>> 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(">>> 加载原版基座模型 (bf16)...")
base_model_obj = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model_obj.eval()

print(">>> 加载微调模型 (LoRA run2)...")
ft_model_obj = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ),
    ADAPTER_DIR,
)
ft_model_obj.eval()
print(">>> 模型加载完成\n")


def gen(model, instruction: str) -> str:
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
            max_new_tokens=2048,
            do_sample=False,
            repetition_penalty=1.1,
        )
    code = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    # 截断到 </html>
    idx = code.lower().rfind("</html>")
    if idx != -1:
        code = code[:idx + 7]
    return code


# ================================================================
# 主对比循环
# ================================================================
DIM_KEYS = [
    "html_structure", "no_markdown_wrap", "viewport_meta",
    "pocketvibe_style", "js_functional", "no_external_dep",
    "code_efficiency", "chinese_ui",
]
MAX_PER_DIM = {k: 2 if k in ("pocketvibe_style", "js_functional") else 1 for k in DIM_KEYS}

rows = []
base_total = 0
ft_total   = 0

header_dims = "  ".join(f"{dim_label(k):^6}" for k in DIM_KEYS)
print(f"{'='*80}")
print(f"{'指令':^30} | {'原版':^5} | {'微调':^5} | {'提升':^5}")
print(f"{'':^30}   {header_dims}")
print(f"{'='*80}")

for i, inst in enumerate(TEST_INSTRUCTIONS):
    print(f"\n[{i+1:02d}/{len(TEST_INSTRUCTIONS)}] {inst}")

    base_code = gen(base_model_obj, inst)
    ft_code   = gen(ft_model_obj,   inst)

    base_sc = evaluate(base_code)
    ft_sc   = evaluate(ft_code)

    base_total += base_sc["total"]
    ft_total   += ft_sc["total"]
    delta = ft_sc["total"] - base_sc["total"]

    # 保存HTML
    (EVAL_DIR / f"base_v2_{i+1:02d}.html").write_text(base_code, encoding="utf-8")
    (EVAL_DIR / f"ft_v2_{i+1:02d}.html").write_text(ft_code,   encoding="utf-8")

    # 逐维度对比行
    base_dim_str = "  ".join(f"{base_sc[k]:^6}" for k in DIM_KEYS)
    ft_dim_str   = "  ".join(f"{ft_sc[k]:^6}" for k in DIM_KEYS)
    sign = "+" if delta > 0 else ("=" if delta == 0 else "")
    print(f"  原版 {base_sc['total_str']:>5}  [{base_dim_str}]")
    print(f"  微调 {ft_sc['total_str']:>5}  [{ft_dim_str}]   {sign}{delta if delta != 0 else '0'}")

    rows.append({
        "指令": inst,
        "原版总分": base_sc["total"],
        "微调总分": ft_sc["total"],
        "提升": delta,
        **{f"原版_{dim_label(k)}": base_sc[k] for k in DIM_KEYS},
        **{f"微调_{dim_label(k)}": ft_sc[k]   for k in DIM_KEYS},
    })

# ================================================================
# 汇总
# ================================================================
max_possible = len(TEST_INSTRUCTIONS) * 10
print(f"\n{'='*80}")
print(f"{'指令':<30} | {'原版':>5} | {'微调':>5} | {'提升':>5}")
print(f"{'-'*80}")
for r in rows:
    sign = f"+{r['提升']}" if r['提升'] > 0 else str(r['提升'])
    print(f"  {r['指令'][:28]:<30} | {r['原版总分']:>5} | {r['微调总分']:>5} | {sign:>5}")
print(f"{'='*80}")
print(f"{'原版总分':>40}: {base_total}/{max_possible}  ({base_total/max_possible*100:.1f}%)")
print(f"{'微调总分':>40}: {ft_total}/{max_possible}  ({ft_total/max_possible*100:.1f}%)")
print(f"{'总提升':>40}: +{ft_total - base_total} 分\n")

# CSV 输出
csv_path = EVAL_DIR / "compare_results_v2.csv"
with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# JSON 报告
report = {
    "model_base":     BASE_MODEL,
    "model_finetuned": f"{BASE_MODEL} + QLoRA run2",
    "adapter":        ADAPTER_DIR,
    "scoring_system": {
        "total_per_sample": 10,
        "dimensions": {k: {"max": MAX_PER_DIM[k], "label": dim_label(k)} for k in DIM_KEYS},
    },
    "test_count": len(TEST_INSTRUCTIONS),
    "base_total":   base_total,
    "ft_total":     ft_total,
    "max_possible": max_possible,
    "base_rate":    f"{base_total/max_possible*100:.1f}%",
    "ft_rate":      f"{ft_total/max_possible*100:.1f}%",
    "improvement":  ft_total - base_total,
    "results": rows,
}

json_path = EVAL_DIR / "compare_report_v2.json"
json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"✅ 对比完成!")
print(f"   CSV  → {csv_path}")
print(f"   JSON → {json_path}")
print(f"   HTML → {EVAL_DIR}/base_v2_*.html  vs  ft_v2_*.html")
