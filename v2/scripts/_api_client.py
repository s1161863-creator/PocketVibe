#!/usr/bin/env python3
"""
PocketVibe V2 — 多 Key Fallback 异步客户端
=================================================================
功能:
  - 维护多个 DeepSeek API Key
  - 当前 key 余额不足 (402 / Insufficient Balance) 时自动切换到下一个
  - 完全透明: 调用方只需 await client.chat_completion(...)
  - 线程/协程安全: 用 asyncio.Lock 保护切换动作
=================================================================
用法:
    from _api_client import MultiKeyAsyncClient, API_KEYS

    client = MultiKeyAsyncClient(API_KEYS, base_url="https://api.deepseek.com")
    resp = await client.chat_completion(
        model="deepseek-chat",
        messages=[...],
        temperature=0.6,
        max_tokens=3500,
    )
    text = resp.choices[0].message.content
"""
import asyncio
import threading
import time
from openai import AsyncOpenAI, OpenAI

# ================================================================
# 两个 Key:  主 Key(小余额) 放前面当日常, 大余额 Key 作备胎兜底
# 如需调换顺序, 直接改这个列表即可
# ================================================================
API_KEYS = [
    "sk-10da60b9c960415992756ade04853606",  # 主 Key, 余额小, 先用完
    "sk-93a3f2ae45674ee1ac518c2af8e4258f",  # 备用 Key, 充足, 兜底
]


# 识别"余额不足"类型错误的关键词
BALANCE_ERROR_KEYWORDS = (
    "insufficient balance",
    "insufficient_quota",
    "insufficient funds",
    "payment required",
    "balance not enough",
    "余额不足",
    " 402",
    "(402",
    "status 402",
    "status_code=402",
)


def _is_balance_error(err: Exception) -> bool:
    """判断异常是否由"余额不足"引起"""
    msg = str(err).lower()
    for kw in BALANCE_ERROR_KEYWORDS:
        if kw.lower() in msg:
            return True
    return False


class MultiKeyAsyncClient:
    """
    多 Key 异步客户端:
      - 初始化时用 keys[0]
      - 调用失败且为"余额不足"时, 自动切到 keys[1], 重试
      - 全部 key 耗尽时抛出最后一次异常
    """

    def __init__(self, keys: list[str], base_url: str, verbose: bool = True):
        if not keys:
            raise ValueError("至少需要 1 个 API Key")
        self.keys = keys
        self.base_url = base_url
        self.verbose = verbose

        self._current_idx = 0
        self._client = AsyncOpenAI(api_key=keys[0], base_url=base_url)
        self._lock = asyncio.Lock()

    @property
    def current_index(self) -> int:
        return self._current_idx

    @property
    def current_key_tail(self) -> str:
        return self.keys[self._current_idx][-8:]

    async def _switch_to_next_key(self, reason: str) -> bool:
        """
        尝试切换到下一个 Key. 返回 True 表示切换成功, False 表示无更多 Key.
        多协程同时触发时用 Lock 保证只切一次.
        """
        async with self._lock:
            # 可能其他协程已经切过了, 检查一下
            # 通过传入的 reason 里面的 idx 判断? 简单起见, 每次都推进一格
            if self._current_idx + 1 >= len(self.keys):
                if self.verbose:
                    print(f"    ❌ 所有 {len(self.keys)} 个 Key 都已耗尽")
                return False

            old_idx = self._current_idx
            old_tail = self.keys[old_idx][-8:]
            self._current_idx += 1
            new_tail = self.keys[self._current_idx][-8:]
            self._client = AsyncOpenAI(
                api_key=self.keys[self._current_idx],
                base_url=self.base_url,
            )
            if self.verbose:
                print(
                    f"\n    💸 Key #{old_idx} (...{old_tail}) 余额耗尽 → "
                    f"切换到 Key #{self._current_idx} (...{new_tail})  |  原因: {reason[:80]}\n"
                )
            return True

    async def chat_completion(self, **kwargs):
        """
        调用 chat.completions.create, 透明处理:
          - 余额不足 → 切 Key 重试
          - 临时错误(429/网络) → 指数退避重试 3 次
          - 彻底失败 → 抛异常
        """
        last_err = None

        # 最多尝试 len(keys) * 3 次
        max_total_attempts = len(self.keys) * 3
        attempt_in_key = 0
        total_attempts = 0

        while total_attempts < max_total_attempts:
            total_attempts += 1
            try:
                # 捕获当前 client 的引用, 避免切 Key 的那一瞬间引用错乱
                cur_idx_before = self._current_idx
                cur_client = self._client
                return await cur_client.chat.completions.create(**kwargs)

            except Exception as e:
                last_err = e

                if _is_balance_error(e):
                    # 余额不足: 立刻切 Key
                    switched = await self._switch_to_next_key(reason=str(e))
                    if switched:
                        attempt_in_key = 0  # 新 Key 重新计数
                        continue
                    else:
                        # 没有更多 Key 了, 直接抛
                        raise

                # 其它错误: 指数退避重试
                attempt_in_key += 1
                if attempt_in_key < 3:
                    await asyncio.sleep(2 ** attempt_in_key)
                    continue
                else:
                    # 当前 Key 连续 3 次错误, 尝试切下一个 Key
                    switched = await self._switch_to_next_key(
                        reason=f"连续 3 次非余额错误: {str(e)[:60]}"
                    )
                    if switched:
                        attempt_in_key = 0
                        continue
                    else:
                        raise

        # 理论上不会走到这里
        if last_err:
            raise last_err
        raise RuntimeError("chat_completion: 未知错误, 未产生响应")


class MultiKeySyncClient:
    """
    同步版多 Key 客户端, 给非 async 脚本用 (如 01c).
    - 调用 chat_completion(**kwargs)  
    - 同样支持 "余额不足 → 自动切 Key" + "其它错误 → 指数退避重试"
    """

    def __init__(self, keys: list[str], base_url: str, verbose: bool = True):
        if not keys:
            raise ValueError("至少需要 1 个 API Key")
        self.keys = keys
        self.base_url = base_url
        self.verbose = verbose

        self._current_idx = 0
        self._client = OpenAI(api_key=keys[0], base_url=base_url)
        self._lock = threading.Lock()

    def _switch_to_next_key(self, reason: str) -> bool:
        with self._lock:
            if self._current_idx + 1 >= len(self.keys):
                if self.verbose:
                    print(f"    ❌ 所有 {len(self.keys)} 个 Key 都已耗尽")
                return False
            old_idx = self._current_idx
            self._current_idx += 1
            self._client = OpenAI(
                api_key=self.keys[self._current_idx],
                base_url=self.base_url,
            )
            if self.verbose:
                print(
                    f"\n    💸 Key #{old_idx} (...{self.keys[old_idx][-8:]}) 余额耗尽 → "
                    f"切换到 Key #{self._current_idx} (...{self.keys[self._current_idx][-8:]})  |  原因: {reason[:80]}\n"
                )
            return True

    def chat_completion(self, **kwargs):
        last_err = None
        max_total_attempts = len(self.keys) * 3
        attempt_in_key = 0
        total_attempts = 0

        while total_attempts < max_total_attempts:
            total_attempts += 1
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                last_err = e
                if _is_balance_error(e):
                    switched = self._switch_to_next_key(reason=str(e))
                    if switched:
                        attempt_in_key = 0
                        continue
                    else:
                        raise
                attempt_in_key += 1
                if attempt_in_key < 3:
                    time.sleep(2 ** attempt_in_key)
                    continue
                else:
                    switched = self._switch_to_next_key(
                        reason=f"连续 3 次非余额错误: {str(e)[:60]}"
                    )
                    if switched:
                        attempt_in_key = 0
                        continue
                    else:
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("chat_completion: 未知错误")
