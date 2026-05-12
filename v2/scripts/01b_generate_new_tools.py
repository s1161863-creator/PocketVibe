#!/usr/bin/env python3
"""Generate new tool examples with full HTML using DeepSeek."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = ROOT / "data" / "processed" / "new_tools.jsonl"
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
NUM_SHARDS = max(1, int(os.getenv("PV_NUM_SHARDS", "1")))
SHARD_INDEX = int(os.getenv("PV_SHARD_INDEX", "0"))
OUTPUT_FILE = Path(os.getenv("PV_OUTPUT_FILE", str(DEFAULT_OUTPUT_FILE))).resolve()

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

NEW_TOOL_INSTRUCTIONS = [
    "做一个卡路里计算器，选择食物和份量算出热量",
    "帮我做个血压记录工具，可以输入收缩压舒张压和心率，显示历史趋势",
    "做一个体重变化追踪器，记录每天体重画出趋势线",
    "帮我弄个睡眠时长计算器，输入入睡和起床时间算出睡了多久",
    "做一个简易视力测试表，显示不同大小的字母让用户辨认",
    "做个体脂率计算器，输入身高体重腰围计算体脂",
    "做一个运动消耗热量计算器，选择运动类型和时间算消耗",
    "帮我做个心率区间计算工具，输入年龄算出各个训练区间",
    "做个拉伸计时器，有多个动作每个30秒自动切换下一个",
    "做一个跑步配速计算器，输入距离和时间算出每公里配速",
    "做一个房贷计算器，支持等额本息和等额本金两种方式",
    "帮我做个复利计算器，输入本金利率和年数算出最终金额",
    "做一个信用卡分期计算器，算出分期总利息和每月还款",
    "做个365天存钱挑战工具，第1天存1块第2天存2块可以打卡",
    "帮我做个工资税后计算器，输入税前工资算出到手多少",
    "做一个记账本，暗色主题，分类显示收入支出",
    "做个零钱凑整工具，输入总金额和面额算出怎么凑",
    "帮我做一个预算分配器，输入月收入按比例分配各项支出",
    "做一个存款目标追踪器，设定目标金额和日期显示进度",
    "做个简单的股票收益计算器，输入买入卖出价格和数量",
    "做一个九九乘法表，点击可以测验，可爱风格",
    "帮我做个GPA计算器，输入各科成绩和学分算出绩点",
    "做一个英语单词拼写练习，显示中文让用户拼出英文",
    "做个数学口算练习工具，随机出加减乘除题目计时作答",
    "帮我做个课程表，可以填写每天每节课的内容",
    "做一个成绩等级转换器，百分制转ABCD等级",
    "做个古诗词背诵卡片，显示上句填下句",
    "帮我做一个英语时态练习工具，判断句子用什么时态",
    "做一个化学元素周期表速查工具，简约风格",
    "做个考试倒计时工具，可以同时设置多个考试日期",
    "做一个情侣纪念日计算器，输入在一起的日期显示天数和各种纪念日",
    "帮我做个宝宝月龄计算器，输入出生日期显示几个月几天了",
    "做一个怀孕周数计算器，输入末次月经日期算出孕周和预产期",
    "做个旅行行李打包清单，可以添加物品和勾选",
    "帮我做一个节日倒数工具，同时显示到春节中秋国庆的天数",
    "做个人生进度条，显示今年已过去百分之多少今天剩余百分之多少",
    "帮我做一个快递费用估算器，输入重量和距离算运费",
    "做个垃圾分类查询工具，输入物品名称告诉你是什么垃圾",
    "帮我做一个穿衣建议工具，输入温度推荐穿什么",
    "做个简易电费计算器，输入用电量和单价算出电费",
    "做一个烹饪多段计时器，可以同时设多个计时各自倒计时",
    "帮我做个食材配比计算器，按人数比例调整用量",
    "做一个饮品调配工具，选择基酒和配料显示配方比例",
    "做个烘焙温度换算器，烤箱温度和时间按模具大小调整",
    "帮我做个每日食谱推荐，随机展示一道菜的做法步骤",
    "做一个真心话大冒险工具，随机出题目",
    "帮我做个谁是卧底词语生成器，给出平民词和卧底词",
    "做一个随机表白情话生成器，粉色可爱风格",
    "做个幸运转盘，可以添加选项然后转动选择",
    "帮我做一个随机抽签工具，输入名字列表随机抽取",
    "做一个你画我猜题目生成器，随机出词让大家猜",
    "帮我做个喝酒骰子游戏，显示不同惩罚",
    "做一个星座速查工具，输入生日显示星座和性格特点",
    "做个生肖年份查询器，输入年份显示对应生肖",
    "帮我做一个塔罗牌抽取工具，随机展示一张牌和含义",
    "做一个演讲计时器，分段提醒绿黄红灯",
    "帮我做个密码强度检测工具，输入密码实时显示强度",
    "做一个文本去重工具，粘贴文本自动去除重复行",
    "帮我做个简易Markdown预览工具，左边输入右边预览",
    "做一个番茄钟统计工具，暗色主题，记录每天完成了多少个番茄",
    "帮我做个时区换算器，选两个城市显示当前时间差",
    "做个会议时长计算器，输入开始结束时间算出总时长",
    "帮我做个专注力测试，显示不同颜色的文字让用户快速回答颜色",
    "做一个井字棋游戏，可以两个人轮流下",
    "帮我做个简易打地鼠游戏，随机出现点击得分",
    "做一个记忆翻牌游戏，翻两张相同的就消除",
    "帮我做一个贪吃蛇小游戏，简约风格",
    "做一个弹球游戏，控制底部挡板弹球打砖块",
    "做个数字华容道游戏，把数字排成正确顺序",
    "做个扫雷精简版，5x5的小棋盘",
    "做一个宝宝喂奶记录器，记录每次喂奶时间和量",
    "帮我做个儿童涂色板，有几个简单图案可以选颜色涂",
    "做一个乘法口诀练习游戏，答对加分答错提示，适合小朋友",
    "帮我做个儿童认数字游戏，显示数字让小朋友点击对应数量的星星",
    "做个讲故事计时器，家长用的睡前故事控制时间",
    "做一个班级随机分组工具，输入人名和组数自动分组",
    "帮我做个座位随机安排器，输入人名随机排座位表",
    "做一个抽奖工具，输入参与者名单随机抽出中奖者有动画效果",
    "帮我做个红包金额随机分配器，输入总金额和人数随机分",
    "做个团建游戏随机选择器，内置多种团建游戏随机推荐",
    "做个鞋码换算器，中国码美国码欧洲码互转",
    "帮我做个衣服尺码对照表，SML和具体尺寸对照",
    "做一个RGB和HEX颜色值互转工具",
    "帮我做个功率单位换算器，瓦特马力千瓦互转",
    "做个数据存储单位换算，B KB MB GB TB互转",
]

GENERATION_PROMPT = """请根据以下用户需求，直接输出一个完整的、可独立运行的 HTML 文件。

严格要求，全部必须满足：
1. 完整 HTML 文件，从 <!DOCTYPE html> 到 </html> 结束
2. 必须包含 <meta name="viewport" content="width=device-width,initial-scale=1.0">
3. 所有 CSS 用 <style> 标签内联在 <head> 中
4. 所有 JavaScript 用 <script> 标签内联在 <body> 末尾
5. 不引用任何外部 CDN、库、字体或 API
6. 适配手机屏幕：按钮 padding >= 12px，字体 >= 16px，使用 max-width: 380px 的卡片布局
7. 现代 UI：渐变背景、圆角、阴影、醒目的按钮样式
8. 中文界面，标题带 emoji
9. 功能逻辑完整，所有按钮都有效
10. 代码控制在 200 行以内

只输出纯 HTML 代码，不要解释，不要 Markdown。

用户需求：{instruction}
"""


def build_client() -> OpenAI:
    if not API_KEY:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请先设置环境变量，例如："
            ' $env:DEEPSEEK_API_KEY="sk-..."'
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def log(message: str) -> None:
    print(message, flush=True)


def load_completed_instructions(*paths: Path) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                completed.add(row["messages"][1]["content"])
    return completed


def clean_html(raw: str) -> str:
    code = raw.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def generate_html(client: OpenAI, instruction: str, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": GENERATION_PROMPT.format(instruction=instruction)}],
                temperature=0.4,
                max_tokens=4000,
            )
            code = clean_html(response.choices[0].message.content or "")
            if (
                "<!DOCTYPE" in code.upper()
                and "</html>" in code.lower()
                and "viewport" in code.lower()
                and "<style" in code.lower()
                and "<script" in code.lower()
            ):
                return code
            log(f"    输出结构异常，第 {attempt + 1} 次重试")
        except Exception as exc:  # noqa: BLE001
            log(f"    [API 错误] {exc}")
        time.sleep(2)
    return None


def main() -> None:
    client = build_client()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_instructions(DEFAULT_OUTPUT_FILE, OUTPUT_FILE)
    pending_all = [item for item in NEW_TOOL_INSTRUCTIONS if item not in completed]
    pending = [
        item for index, item in enumerate(pending_all)
        if index % NUM_SHARDS == SHARD_INDEX
    ]

    log(f"使用模型: {MODEL} @ {BASE_URL}")
    log(f"输出文件: {OUTPUT_FILE}")
    log(f"分片设置: shard={SHARD_INDEX + 1}/{NUM_SHARDS}")
    log(f"已存在 {len(completed)} 条，剩余总待生成 {len(pending_all)} 条，本分片负责 {len(pending)} 条")

    success = 0
    failure = 0
    with OUTPUT_FILE.open("a", encoding="utf-8") as fh:
        for index, instruction in enumerate(pending, start=1):
            log(f"[{index:>2}/{len(pending)}] {instruction}")
            html = generate_html(client, instruction)
            if html is None:
                failure += 1
                log("    失败")
                continue

            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": html},
                ]
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            fh.flush()
            success += 1
            log(f"    成功，HTML 长度 {len(html)}")
            time.sleep(1)

    log("\n阶段 B 完成")
    log(f"成功: {success} 条 | 失败: {failure} 条")
    log(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
