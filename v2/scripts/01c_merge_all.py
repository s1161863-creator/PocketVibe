#!/usr/bin/env python3
"""Merge augmented instructions and new-tool samples into one dataset."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FILES = [
    PROCESSED_DIR / "augmented_instructions.jsonl",
    PROCESSED_DIR / "new_tools.jsonl",
    *sorted(PROCESSED_DIR.glob("new_tools.shard*.jsonl")),
]
OUTPUT = PROCESSED_DIR / "train_generated.jsonl"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0

    with OUTPUT.open("w", encoding="utf-8") as out:
        for path in FILES:
            if not path.exists():
                print(f"文件不存在，跳过：{path}")
                continue
            count = 0
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    item = json.loads(line)
                    instruction = item["messages"][1]["content"].strip()
                    if instruction in seen:
                        continue
                    seen.add(instruction)
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
            total += count
            print(f"{path.name}: 写入 {count} 条")

    print(f"\n合并完成，共 {total} 条唯一数据")
    print(f"输出文件: {OUTPUT}")


if __name__ == "__main__":
    main()
