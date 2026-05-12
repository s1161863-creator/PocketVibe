#!/usr/bin/env python3
"""
PocketVibe V2 — Preflight 检查
===============================================
在启动并发版之前，验证：
1. DeepSeek API Key 是否有效 + 是否付费账号（查余额）
2. 15 并发压测是否稳定（不触发 429）
3. 已生成的 JSONL 数据完整性（JSON 能否解析 / HTML 是否有效）
===============================================
运行: python scripts/00_preflight_check.py
"""
import os, sys, json, asyncio, time, re
import urllib.request, urllib.error
from openai import AsyncOpenAI

# Windows 控制台默认 GBK，强制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

API_KEY  = "sk-10da60b9c960415992756ade04853606"
BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"

DATA_FILES = [
    "data/processed/multi_impl.jsonl",
    "data/processed/evol_tools.jsonl",
    "data/processed/cross_cat.jsonl",
]

print("=" * 60)
print("PocketVibe V2 - Preflight Check")
print("=" * 60)

# ================================================================
# 1. 查账户余额
# ================================================================
print("\n[1/3] 查询 DeepSeek 账户余额...")
try:
    req = urllib.request.Request(
        f"{BASE_URL}/user/balance",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body)

    print(f"  原始响应: {json.dumps(data, ensure_ascii=False)[:300]}")

    is_available = data.get("is_available", False)
    balance_infos = data.get("balance_infos", [])

    print(f"  账号可用: {'✅ 是' if is_available else '❌ 否'}")

    total_usd = 0.0
    for info in balance_infos:
        currency = info.get("currency", "?")
        total    = info.get("total_balance", "0")
        granted  = info.get("granted_balance", "0")
        topped   = info.get("topped_up_balance", "0")
        print(f"    {currency}: 总余额={total} (赠送={granted}, 充值={topped})")
        if currency == "CNY":
            try:
                total_usd += float(total) / 7.25
            except Exception:
                pass
        elif currency == "USD":
            try:
                total_usd += float(total)
            except Exception:
                pass

    has_paid = any(float(info.get("topped_up_balance", "0") or 0) > 0
                    for info in balance_infos)
    print(f"  付费账号: {'✅ 是（有充值余额）' if has_paid else '⚠️  否（仅免费额度）'}")
    print(f"  总余额估算: ~${total_usd:.2f}")

except urllib.error.HTTPError as e:
    print(f"  ❌ HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}")
    print(f"  → API Key 可能无效或已过期")
except Exception as e:
    print(f"  ❌ 查询失败: {e}")


# ================================================================
# 2. 并发压测
# ================================================================
print("\n[2/3] 15 并发压测（发送 15 个最小请求）...")

async def ping(client, sem, idx, result):
    async with sem:
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "回复ok"}],
                max_tokens=5,
                temperature=0,
            )
            result["success"] += 1
            result["latencies"].append(time.time() - t0)
        except Exception as e:
            result["fail"] += 1
            err_msg = str(e)[:200]
            result["errors"].append(f"#{idx}: {err_msg}")
            if "429" in err_msg or "rate" in err_msg.lower():
                result["rate_limited"] += 1


async def run_pressure_test():
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(15)
    result = {"success": 0, "fail": 0, "rate_limited": 0,
              "latencies": [], "errors": []}
    t_start = time.time()
    await asyncio.gather(*[ping(client, sem, i, result) for i in range(15)])
    total_time = time.time() - t_start

    print(f"  成功: {result['success']}/15 | 失败: {result['fail']}/15")
    print(f"  总耗时: {total_time:.2f}s")
    if result["latencies"]:
        avg = sum(result["latencies"]) / len(result["latencies"])
        mn, mx = min(result["latencies"]), max(result["latencies"])
        print(f"  延迟: 平均 {avg:.2f}s | 最快 {mn:.2f}s | 最慢 {mx:.2f}s")
    if result["rate_limited"] > 0:
        print(f"  ⚠️  限流次数: {result['rate_limited']} → 建议并发降到 10")
    else:
        print(f"  ✅ 无 429 限流")
    if result["errors"]:
        print(f"  失败详情（前 3 条）:")
        for err in result["errors"][:3]:
            print(f"    {err}")

    if result["success"] == 15 and result["rate_limited"] == 0:
        print(f"  👉 推荐并发度: 15")
    elif result["success"] >= 12:
        print(f"  👉 推荐并发度: 10（有少量失败）")
    else:
        print(f"  👉 推荐并发度: 5（稳定性不足，建议降级）")


try:
    asyncio.run(run_pressure_test())
except Exception as e:
    print(f"  ❌ 压测失败: {e}")


# ================================================================
# 3. 扫描已有 JSONL 完整性
# ================================================================
print("\n[3/3] 扫描已有 JSONL 文件...")

def scan_jsonl(path):
    if not os.path.exists(path):
        return None

    total_lines = 0
    valid_json = 0
    valid_html = 0
    html_lens  = []
    bad_samples = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_lines += 1
            try:
                item = json.loads(line)
                valid_json += 1
            except Exception as e:
                if len(bad_samples) < 3:
                    bad_samples.append(f"行{lineno}: JSON 解析失败 ({str(e)[:60]})")
                continue

            # 检查 HTML 内容
            try:
                html = item["messages"][-1]["content"]
            except Exception:
                if len(bad_samples) < 3:
                    bad_samples.append(f"行{lineno}: 缺少 messages[-1].content")
                continue

            html_len = len(html)
            html_lens.append(html_len)

            up = html.upper()
            if ("<!DOCTYPE" in up) and ("</HTML>" in up) and ("viewport" in html):
                valid_html += 1
            else:
                if len(bad_samples) < 3:
                    missing = []
                    if "<!DOCTYPE" not in up: missing.append("DOCTYPE")
                    if "</HTML>" not in up:   missing.append("</html>")
                    if "viewport" not in html: missing.append("viewport")
                    bad_samples.append(f"行{lineno}: HTML 缺少 {missing} (长度={html_len})")

    return {
        "total": total_lines,
        "valid_json": valid_json,
        "valid_html": valid_html,
        "html_lens": html_lens,
        "bad_samples": bad_samples,
    }


for path in DATA_FILES:
    print(f"\n  📄 {path}")
    r = scan_jsonl(path)
    if r is None:
        print(f"    ⚠️  文件不存在")
        continue
    print(f"    总行数: {r['total']}  |  合法 JSON: {r['valid_json']}  |  合法 HTML: {r['valid_html']}")
    if r["html_lens"]:
        mn = min(r["html_lens"])
        mx = max(r["html_lens"])
        mean = sum(r["html_lens"]) / len(r["html_lens"])
        print(f"    HTML 长度: min={mn} / max={mx} / 均值={mean:.0f}")
    if r["bad_samples"]:
        print(f"    ❌ 发现 {r['total'] - r['valid_html']} 条异常：")
        for s in r["bad_samples"]:
            print(f"      - {s}")
    else:
        print(f"    ✅ 全部通过校验")


print("\n" + "=" * 60)
print("Preflight 检查完成")
print("=" * 60)
