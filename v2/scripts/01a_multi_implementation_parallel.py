#!/usr/bin/env python3
"""
PocketVibe V2 — 01a 并发版：一指令 × 四风格 HTML 实现（asyncio）
=================================================================
与原 01a_multi_implementation.py 完全等价（同样的 STYLE_VARIANTS / Prompt）
仅把串行 API 调用改成 asyncio 并发，吞吐量提升约 15-20 倍。

关键保障：
- 断点续传：沿用同一个 multi_impl.jsonl，通过 _source_key 跳过已完成条目
- 文件写入加锁
- 和原脚本完全兼容，下游 01d/01e/02 不用改

预计时间：50 种子 × 4 风格 = 200 条，并发=15，约 5-10 分钟完成
=================================================================
运行：python scripts/01a_multi_implementation_parallel.py
"""
import json, os, asyncio, re, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _api_client import MultiKeyAsyncClient, API_KEYS

# ================================================================
# 配置（与原脚本保持一致）
# ================================================================
BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"

SEED_FILE   = "data/seed/seed_examples.jsonl"
OUTPUT_FILE = "data/processed/multi_impl.jsonl"

MAX_CONCURRENT = 15

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

client = MultiKeyAsyncClient(API_KEYS, base_url=BASE_URL)

# 四种视觉风格（与原脚本完全一致）
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
                "html_v1":     msgs[2]["content"],
            })
    return seeds


async def generate_html_async(semaphore, instruction: str, style: dict) -> str | None:
    prompt = GENERATION_PROMPT.format(style_desc=style["desc"], instruction=instruction)
    async with semaphore:
        try:
            resp = await client.chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=3500,
            )
            code = resp.choices[0].message.content.strip()
            if code.startswith("```"):
                code = re.sub(r'^```[a-zA-Z]*\n?', '', code)
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            code = code.strip()
            if ("<!DOCTYPE" in code.upper() or "<!doctype" in code.lower()) and \
               "</html>" in code.lower() and "viewport" in code:
                return code
            return None
        except Exception as e:
            print(f"    [生成失败] {str(e)[:100]}")
            return None


async def process_task(semaphore, file_lock, fout, task, progress):
    i, seed, style = task["i"], task["seed"], task["style"]
    code = await generate_html_async(semaphore, seed["instruction"], style)
    if not code:
        progress["fail"] += 1
        return

    entry = {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": seed["instruction"]},
            {"role": "assistant", "content": code},
        ],
        "_source_key":  f"{i}_{style['name']}",
        "_style":       style["name"],
        "_category":    "seed_multi_impl",
    }

    async with file_lock:
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fout.flush()
        progress["success"] += 1
        total = progress["success"] + progress["fail"]
        if total % 10 == 0 or total == progress["total_todo"]:
            elapsed = time.time() - progress["start_time"]
            rate = total / elapsed if elapsed > 0 else 0
            eta = (progress["total_todo"] - total) / rate if rate > 0 else 0
            print(f"  [{total}/{progress['total_todo']}] ✅{progress['success']} ❌{progress['fail']} | "
                  f"速度:{rate:.1f}条/秒 | 剩余:{eta/60:.1f}分钟")


async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    seeds = load_seeds()

    # 断点续传
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    done_keys.add(item.get("_source_key", ""))
                except Exception:
                    pass
        print(f"♻️  已有 {len(done_keys)} 条，将跳过\n")

    # 构建所有任务
    all_tasks = []
    for i, seed in enumerate(seeds):
        for style in STYLE_VARIANTS:
            key = f"{i}_{style['name']}"
            if key not in done_keys:
                all_tasks.append({"i": i, "seed": seed, "style": style})

    total_target = len(seeds) * len(STYLE_VARIANTS)
    print(f"📖 种子数：{len(seeds)}")
    print(f"🎨 风格数：{len(STYLE_VARIANTS)}")
    print(f"🎯 目标总数：{total_target} 条")
    print(f"⏭️  待生成：{len(all_tasks)} 条（{len(done_keys)} 条已完成）")
    print(f"⚡ 并发度：{MAX_CONCURRENT}")
    print()

    if not all_tasks:
        print("✅ 全部完成，无需处理")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    file_lock = asyncio.Lock()
    progress = {
        "success": 0,
        "fail": 0,
        "total_todo": len(all_tasks),
        "start_time": time.time(),
    }

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        coros = [process_task(semaphore, file_lock, fout, task, progress) for task in all_tasks]
        await asyncio.gather(*coros)

    elapsed = time.time() - progress["start_time"]
    print(f"\n{'='*50}")
    print(f"✅ 01a 并发版完成！")
    print(f"   本次成功: {progress['success']} | 失败: {progress['fail']}")
    print(f"   耗时: {elapsed/60:.1f} 分钟 | 平均速度: {progress['success']/elapsed:.2f} 条/秒")
    print(f"📄 输出：{OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
