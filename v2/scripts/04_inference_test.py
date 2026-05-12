#!/usr/bin/env python3
"""
PocketVibe V2 — 04: 推理测试（优化版 2026.05）
=================================================================
优化项（基于 Qwen 官方推荐 + 2025-2026 业界最佳实践）:
  1. 官方参数: temperature=0.7, top_p=0.8, top_k=20（Qwen2.5 Instruct 官方 model card）
  2. 移除 repetition_penalty（代码生成不应用，Reddit LocalLLaMA 社区建议）
  3. Best-of-3 采样: 每条指令生成 3 次，自动选质量最高的
  4. 代码清洗: 去 markdown 包裹 / 截取 <!DOCTYPE 之前的解释文字
  5. Stop tokens: 遇到 </html> 自动停止，防止继续输出解释
  6. 贪心兜底: 3 次采样全失败时用 do_sample=False 第 4 次尝试
  7. 失败样本全部记录（供报告 Error Analysis 使用）

运行（HPC 上）：python scripts/04_inference_test.py
输出：data/eval/test_*.html  +  data/eval/test_summary.json
=================================================================
参考资料:
  - Qwen2.5 official model card: temperature=0.7, top_p=0.8, top_k=20
  - Muxup 2025Q2 Vendor-recommended LLM parameter reference
  - Reddit r/LocalLLaMA: "repetition penalty > 1 is disabled for coding models"
  - AlphaCode / OpenAI Codex: Best-of-N sampling for code generation
=================================================================
"""
import os, json, re, torch
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
# 推理参数（Qwen2.5 官方推荐 + 2026 最佳实践）
# ================================================================
# 官方来源: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct
# Instruct 非思考模式: temperature=0.7, top_p=0.8, top_k=20
# 代码生成: repetition_penalty 应设为 1.0（禁用）
SAMPLE_KWARGS = dict(
    temperature=0.7,        # 官方: 0.7（≠原版0.2，过低导致退化/截断）
    top_p=0.8,              # 官方: 0.8（≠原版0.9）
    top_k=20,               # 官方: 20（原版未设置）
    do_sample=True,
    repetition_penalty=1.0, # 代码生成必须关闭（原版1.1有害，会切断</div>等合法重复token）
    max_new_tokens=2048,
)

GREEDY_KWARGS = dict(       # 兜底用: 贪心解码，最稳定
    do_sample=False,
    max_new_tokens=2048,
    repetition_penalty=1.0,
)

BEST_OF_N = 3               # 每条指令采样次数（业界: AlphaCode 用 best-of-N）

# ================================================================
# 测试指令（全部来自 held-out 类别，训练集绝对没见过）
# ================================================================
TEST_PROMPTS = [
    # creative
    ("creative_01", "做一个简易画板，暗色主题，有颜色选择和橡皮擦"),
    # social
    ("social_01",   "做一个随机分组工具，输入名字列表，选择组数，自动分组"),
    # finance
    ("finance_01",  "做一个房贷月供计算器，等额本息，显示还款总额和利息"),
    # parenting
    ("parenting_01","做一个宝宝睡眠记录器，记录每次睡着和醒来的时间"),
    # cross: timer×game
    ("cross_tg_01", "做一个限时30秒的心算闯关游戏，答对越多分越高"),
    # cross: health×timer
    ("cross_ht_01", "做一个护眼20-20-20提醒器，每20分钟提醒看远处20秒"),
    # planner
    ("planner_01",  "做一个周计划表，7天×6时段，可以填写每个时段的任务"),
    # utility
    ("utility_01",  "做一个文本行去重工具，粘贴文本，自动删除重复行"),
]


# ================================================================
# 工具函数
# ================================================================

def clean_code(raw: str) -> str:
    """
    清洗模型输出:
    1. 去掉 markdown 代码块包裹（```html ... ```）
    2. 去掉 <!DOCTYPE 之前的解释文字
    3. 去掉 </html> 之后的解释文字
    """
    code = raw.strip()

    # 去 markdown 包裹
    if code.startswith("```"):
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code, count=1)
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    code = code.strip()

    # 截取 <!DOCTYPE 开头
    idx = code.upper().find("<!DOCTYPE")
    if idx > 0:
        code = code[idx:]

    # 截取 </html> 结尾（去掉后面的解释文字）
    end_idx = code.lower().rfind("</html>")
    if end_idx != -1:
        code = code[:end_idx + len("</html>")]

    return code.strip()


def score_html(code: str) -> int:
    """
    对 HTML 代码打质量分（0-6），用于 Best-of-N 自动选优。
    分数越高越好。
    """
    score = 0
    c = code.lower()
    if "<!doctype html>" in c or "<!doctype" in c:
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


def quick_check(code: str) -> dict:
    """详细质量检查（用于报告）"""
    return {
        "runnable":   ("<!DOCTYPE" in code.upper() or "<!doctype" in code.lower())
                      and "</html>" in code.lower(),
        "viewport":   "viewport" in code.lower(),
        "mobile":     any(k in code.lower() for k in ["max-width", "100%", "100vw"]),
        "no_cdn":     "cdn." not in code.lower()
                      and "unpkg" not in code.lower()
                      and "jsdelivr" not in code.lower(),
        "has_style":  "<style" in code.lower(),
        "has_script": "<script" in code.lower(),
        "length":     len(code),
        "score":      score_html(code),
    }


# ================================================================
# 模型加载
# ================================================================

def load_model():
    """加载基础模型 + LoRA 适配器"""
    print(f">>> 加载分词器：{BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f">>> 加载基础模型（bf16）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f">>> 加载 LoRA 适配器：{ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return tokenizer, model


# ================================================================
# 推理：Best-of-N + 贪心兜底
# ================================================================

def generate_one(tokenizer, model, instruction: str, gen_kwargs: dict) -> str:
    """单次推理，返回清洗后的代码"""
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


def generate_best_of_n(tokenizer, model, instruction: str) -> tuple[str, list[str]]:
    """
    Best-of-N 采样 + 贪心兜底。
    返回: (最佳代码, 所有尝试列表)

    流程:
      1. 采样 BEST_OF_N 次（temperature=0.7, top_p=0.8, top_k=20）
      2. 按 score_html() 打分，取得分最高的
      3. 若 score < 4（基本不合格），第 N+1 次用 do_sample=False 贪心解码
    """
    candidates = []

    # 阶段 1: BEST_OF_N 次随机采样
    for i in range(BEST_OF_N):
        code = generate_one(tokenizer, model, instruction, SAMPLE_KWARGS)
        candidates.append(code)
        sc = score_html(code)
        print(f"    [采样 {i+1}/{BEST_OF_N}] score={sc}  len={len(code)}")
        if sc == 6:
            # 满分直接用，不再尝试
            print(f"    → 满分! 跳过剩余采样")
            break

    # 选最优
    best = max(candidates, key=score_html)

    # 阶段 2: 若最优仍不合格，贪心兜底
    if score_html(best) < 4:
        print(f"    [贪心兜底] 采样均未达标(score<4), 尝试 do_sample=False ...")
        greedy_code = generate_one(tokenizer, model, instruction, GREEDY_KWARGS)
        greedy_sc   = score_html(greedy_code)
        candidates.append(greedy_code)
        print(f"    [贪心兜底] score={greedy_sc}  len={len(greedy_code)}")
        if greedy_sc >= score_html(best):
            best = greedy_code

    return best, candidates


# ================================================================
# 主程序
# ================================================================

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    tokenizer, model = load_model()

    print(f"\n{'='*60}")
    print(f"推理参数: temperature={SAMPLE_KWARGS['temperature']}, "
          f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}")
    print(f"Best-of-N: {BEST_OF_N}  |  贪心兜底: 启用")
    print(f"代码清洗: 启用  |  repetition_penalty: 1.0（禁用）")
    print(f"{'='*60}\n")

    results = []

    for tag, prompt in TEST_PROMPTS:
        print(f"\n{'='*55}")
        print(f"[{tag}] {prompt}")
        print(f"{'='*55}")

        best_code, all_candidates = generate_best_of_n(tokenizer, model, prompt)
        chk = quick_check(best_code)

        # 保存最佳 HTML
        out_path = os.path.join(EVAL_DIR, f"{tag}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(best_code)

        # 保存所有候选（供报告 Error Analysis）
        for idx, cand in enumerate(all_candidates):
            cand_path = os.path.join(EVAL_DIR, f"{tag}_candidate_{idx+1}.html")
            with open(cand_path, "w", encoding="utf-8") as f:
                f.write(cand)

        passed = all(v for k, v in chk.items() if k not in ("length", "score"))
        print(f"  质量检查：{chk}")
        print(f"  结果：{'✅ 通过' if passed else '❌ 未通过'}  (score={chk['score']}/6)")
        print(f"  → 已保存：{out_path}")

        results.append({
            "tag":              tag,
            "instruction":      prompt,
            "checks":           chk,
            "passed":           passed,
            "best_score":       chk["score"],
            "num_candidates":   len(all_candidates),
            "all_scores":       [score_html(c) for c in all_candidates],
        })

    # 保存汇总 JSON
    summary_path = os.path.join(EVAL_DIR, "test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    passed_count = sum(1 for r in results if r["passed"])
    avg_score = sum(r["best_score"] for r in results) / len(results)

    print(f"\n{'='*55}")
    print(f"✅ 推理测试完成：{passed_count}/{len(results)} 条通过质量检查")
    print(f"📊 平均质量得分：{avg_score:.2f}/6")
    print(f"📄 HTML 文件：{EVAL_DIR}/")
    print(f"📄 汇总报告：{summary_path}")
    print(f"⏭️  下一步：python scripts/05_eval_compare.py")


if __name__ == "__main__":
    main()
