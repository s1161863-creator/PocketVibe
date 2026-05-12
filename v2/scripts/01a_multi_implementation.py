#!/usr/bin/env python3
"""
PocketVibe V2 — 01a: 一指令 × 四风格 HTML 实现
=================================================================
修复 V1 核心问题：同一 HTML × 10 指令（伪增强）
V2 新策略：同一指令 → 4 种不同风格的 HTML 实现（真增强）

原理：迫使模型学习"功能语义 → 代码结构"的映射，而非记忆模板
输出：data/processed/multi_impl.jsonl（约 200 条）
=================================================================
运行：python scripts/01a_multi_implementation.py
"""
import json, os, time, re
from openai import OpenAI

API_KEY  = "sk-10da60b9c960415992756ade04853606"
BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"

SEED_FILE   = "data/seed/seed_examples.jsonl"
OUTPUT_FILE = "data/processed/multi_impl.jsonl"

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 四种视觉风格的 prompt 修饰词
STYLE_VARIANTS = [
    {
        "name": "simple-light",
        "desc": "简约白底风格：背景白色或浅灰，卡片白色，字体黑色，颜色点缀用单一主色调，代码简洁≤100行",
    },
    {
        "name": "dark-neon",
        "desc": "暗色霓虹风格：背景深色(#0f0c29或类似)，文字浅色，强调色用霓虹蓝/紫/青，按钮有发光效果，加入微动画(transition/transform)",
    },
    {
        "name": "colorful-emoji",
        "desc": "多彩活泼风格：背景用多色渐变，卡片彩色，大量使用emoji装饰标题和按钮，字体活泼，圆角更大(24px+)",
    },
    {
        "name": "minimal-mono",
        "desc": "极简单色风格：只用黑白灰三色，无渐变，边框线条感强，字体偏向等宽/系统字体，留白充足，克制优雅",
    },
]

GENERATION_PROMPT = """请根据以下需求，以【{style_desc}】的视觉风格，直接输出完整的、可独立运行的HTML文件。

严格要求：
1. 完整HTML文件，从<!DOCTYPE html>开始到</html>结束
2. 必须包含 <meta name="viewport" content="width=device-width,initial-scale=1.0">
3. 所有CSS内联在<head>的<style>标签中（不能引用外部CSS）
4. 所有JavaScript内联在<body>末尾的<script>标签中（不能引用外部JS）
5. 不引用任何外部CDN、库、字体或API（纯原生HTML/CSS/JS）
6. 按钮padding≥12px，字体≥16px，使用max-width:380px居中卡片布局
7. 功能逻辑必须完整正确，所有按钮都有对应功能
8. 只输出纯HTML代码，不要任何解释或Markdown标记

视觉风格要求：{style_desc}

用户需求：{instruction}"""


def load_seeds():
    seeds = []
    with open(SEED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            msgs = item["messages"]
            seeds.append({
                "system":      msgs[0]["content"],
                "instruction": msgs[1]["content"],
                "html_v1":     msgs[2]["content"],  # V1 的原始实现，供参考
            })
    return seeds


def generate_html(instruction: str, style: dict, retries: int = 2) -> str | None:
    prompt = GENERATION_PROMPT.format(
        style_desc=style["desc"],
        instruction=instruction
    )
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,    # 稍高温度保证风格多样性
                max_tokens=3500,
            )
            code = resp.choices[0].message.content.strip()
            # 清理 markdown 代码块
            if code.startswith("```"):
                code = re.sub(r'^```[a-zA-Z]*\n?', '', code)
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            code = code.strip()
            # 快速验证
            if ("<!DOCTYPE" in code.upper() or "<!doctype" in code.lower()) and \
               "</html>" in code.lower() and "viewport" in code:
                return code
            elif attempt < retries:
                print(f"    ⚠ 第{attempt+1}次格式不对，重试...")
                time.sleep(1)
        except Exception as e:
            print(f"    [API错误] {e}")
            if attempt < retries:
                time.sleep(3)
    return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    seeds = load_seeds()
    print(f"📖 加载了 {len(seeds)} 条种子数据")
    print(f"🎯 目标：每条生成 {len(STYLE_VARIANTS)} 种风格 → 共 {len(seeds) * len(STYLE_VARIANTS)} 条")
    print(f"🤖 模型：{MODEL} @ {BASE_URL}\n")

    # 加载已生成的（断点续传）
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                key = item.get("_source_key", "")
                done_keys.add(key)
        print(f"♻️  已找到 {len(done_keys)} 条已生成数据，将跳过重复\n")

    success = 0
    fail = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i, seed in enumerate(seeds):
            for style in STYLE_VARIANTS:
                key = f"{i}_{style['name']}"
                if key in done_keys:
                    success += 1
                    continue

                print(f"[{i+1}/{len(seeds)}][{style['name']}] {seed['instruction'][:40]}...")
                code = generate_html(seed["instruction"], style)

                if code:
                    entry = {
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": seed["instruction"]},
                            {"role": "assistant", "content": code},
                        ],
                        "_source_key":  key,
                        "_style":       style["name"],
                        "_category":    "seed_multi_impl",
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()
                    success += 1
                    print(f"    ✅ 成功 ({len(code)} 字符)")
                else:
                    fail += 1
                    print(f"    ❌ 失败，跳过")

                time.sleep(0.8)

    print(f"\n{'='*50}")
    print(f"✅ 01a 完成！成功: {success} | 失败: {fail}")
    print(f"📄 输出：{OUTPUT_FILE}")
    print(f"⏭️  下一步：python scripts/01b_evol_instruct_tools.py")


if __name__ == "__main__":
    main()
