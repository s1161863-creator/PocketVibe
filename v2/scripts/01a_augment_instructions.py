#!/usr/bin/env python3
"""Generate instruction paraphrases for existing seed examples with DeepSeek."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / "data" / "seed" / "seed_examples.jsonl"
OUTPUT_FILE = ROOT / "data" / "processed" / "augmented_instructions.jsonl"
NUM_VARIANTS = int(os.getenv("PV_NUM_VARIANTS", "10"))
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

AUGMENT_PROMPT = """你是一个用户行为模拟器。
以下是一个手机小工具的标准功能描述：「{instruction}」

请模拟 {n} 个不同中国用户的真实输入习惯，为上述需求生成 {n} 条不同的指令表达。

严格要求：
1. 长度多样：至少 2 条 <= 6 个字，至少 2 条 >= 15 个字
2. 风格多样：同时包含口语化表达和略正式表达
3. 至少 2 条包含视觉偏好关键词，比如“暗色主题”“粉色系”“简约风格”“可爱一点”“高级感”
4. 至少 1 条包含具体参数或场景，比如“3分钟”“上班用的”“送女朋友用的”
5. 功能含义必须与原始描述完全一致，不能新增或删减核心功能

直接输出 {n} 条指令，每行一条，不要编号，不要解释。
"""


def build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请先设置环境变量，例如："
            ' $env:DEEPSEEK_API_KEY="sk-..."'
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_seeds() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with SEED_FILE.open("r", encoding="utf-8") as fh:
      for line in fh:
            row = json.loads(line)
            messages = row["messages"]
            items.append(
                {
                    "system": messages[0]["content"],
                    "instruction": messages[1]["content"],
                    "html": messages[2]["content"],
                }
            )
    return items


def clean_variants(raw: str) -> list[str]:
    variants: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^[\d]+[.)、]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line)
        line = line.strip("“”\"' ")
        if len(line) >= 2:
            variants.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped[:NUM_VARIANTS]


def generate_variants(client: OpenAI, instruction: str) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": AUGMENT_PROMPT.format(instruction=instruction, n=NUM_VARIANTS)}],
        temperature=0.9,
        max_tokens=600,
    )
    content = response.choices[0].message.content or ""
    return clean_variants(content)


def main() -> None:
    client = build_client()
    seeds = load_seeds()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        for seed in seeds:
            payload = {
                "messages": [
                    {"role": "system", "content": seed["system"]},
                    {"role": "user", "content": seed["instruction"]},
                    {"role": "assistant", "content": seed["html"]},
                ]
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            total += 1

        print(f"加载 {len(seeds)} 条种子数据")
        print(f"使用模型: {MODEL} @ {BASE_URL}")
        print(f"每条种子扩充 {NUM_VARIANTS} 条指令变体")

        for index, seed in enumerate(seeds, start=1):
            print(f"[{index:>2}/{len(seeds)}] {seed['instruction']}")
            variants = generate_variants(client, seed["instruction"])
            for variant in variants:
                payload = {
                    "messages": [
                        {"role": "system", "content": seed["system"]},
                        {"role": "user", "content": variant},
                        {"role": "assistant", "content": seed["html"]},
                    ]
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                total += 1
            print(f"    生成 {len(variants)} 条，累计 {total} 条")
            time.sleep(0.5)

    print(f"\n阶段 A 完成，共 {total} 条数据")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
