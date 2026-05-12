#!/usr/bin/env python3
"""
PocketVibe V2 — 01c: 跨类组合指令生成
=================================================================
手工设计 60 条跨类组合指令（计算器×游戏、计时器×记录等）
用 DeepSeek API 扩写完整 HTML，打破"功能类型→固定模板"的死链
支持断点续传

输出：data/processed/cross_cat.jsonl（约 100 条）
=================================================================
运行：python scripts/01c_cross_category.py
"""
import json, os, time, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _api_client import MultiKeySyncClient, API_KEYS

BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"
OUTPUT_FILE = "data/processed/cross_cat.jsonl"

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

client = MultiKeySyncClient(API_KEYS, base_url=BASE_URL)

# ================================================================
# 60 条跨类组合指令
# 每条标注组合了哪两个类别，便于 held-out 切分时归类
# ================================================================
CROSS_CATEGORY_TOOLS = [
    # ===== 计时器 × 游戏 =====
    ("timer_game", "做一个番茄钟游戏，每完成一个番茄解锁一个成就徽章"),
    ("timer_game", "做一个反应速度计时测试，显示历史最快记录"),
    ("timer_game", "做一个限时猜数字游戏，30秒内尽量多猜"),
    ("timer_game", "做一个打字速度比赛计时器，显示WPM和倒计时"),
    ("timer_game", "做一个倒计时炸弹游戏，时间到了显示爆炸动画"),

    # ===== 计算器 × 游戏 =====
    ("calculator_game", "做一个心算训练游戏，随机出算术题计时作答，答对加分"),
    ("calculator_game", "做一个24点游戏，给出4个数字让用户算出24"),
    ("calculator_game", "做一个数学猜猜看，显示答案让用户猜算式"),
    ("calculator_game", "做一个乘法速算挑战，看谁能连续答对最多题"),
    ("calculator_game", "做一个估算游戏，猜测结果在范围内就得分"),

    # ===== 计时器 × 记录 =====
    ("timer_tracker", "做一个工作时间记录器，可以开始结束并保存每段工时"),
    ("timer_tracker", "做一个番茄钟+每日完成计数，记录今天做了几个番茄"),
    ("timer_tracker", "做一个运动打卡计时器，记录每次锻炼时长"),
    ("timer_tracker", "做一个冥想计时器，保存历史冥想记录"),
    ("timer_tracker", "做一个阅读计时器，记录每本书花了多少时间"),

    # ===== 换算 × 生活 =====
    ("converter_lifestyle", "做一个购物换算器，输入外国价格和汇率算人民币"),
    ("converter_lifestyle", "做一个食谱单位换算，克和杯勺互转"),
    ("converter_lifestyle", "做一个旅行费用换算器，同时显示多种货币"),
    ("converter_lifestyle", "做一个穿衣温度建议器，输入华氏温度给出建议"),
    ("converter_lifestyle", "做一个快递尺寸换算器，厘米英寸加上重量换算"),

    # ===== 记录 × 健康 =====
    ("tracker_health", "做一个每日水分和卡路里双追踪器"),
    ("tracker_health", "做一个运动+体重联合记录工具"),
    ("tracker_health", "做一个血糖记录器，标注餐前餐后并显示趋势"),
    ("tracker_health", "做一个睡眠质量日记，记录入睡时间和心情评分"),
    ("tracker_health", "做一个步数目标追踪，输入今日步数显示完成进度"),

    # ===== 游戏 × 社交 =====
    ("game_social", "做一个多人猜数字游戏，两个人轮流猜同一个数"),
    ("game_social", "做一个真心话大冒险+计时器，每人限时15秒完成"),
    ("game_social", "做一个团队随机PK挑战选择器，随机出题两队对抗"),
    ("game_social", "做一个你画我猜计时题目器，随机词加30秒倒计时"),
    ("game_social", "做一个班级积分游戏板，多个队伍可以加减分"),

    # ===== 教育 × 游戏 =====
    ("education_game", "做一个英语单词消消乐，拖动字母拼出单词"),
    ("education_game", "做一个数学闯关游戏，每关出一道题答对进入下一关"),
    ("education_game", "做一个成语接龙游戏，输入成语最后一个字开头的成语"),
    ("education_game", "做一个地理知识问答，随机出省份首府问题"),
    ("education_game", "做一个化学元素猜谜，显示原子序数让用户猜元素符号"),

    # ===== 健康 × 计时器 =====
    ("health_timer", "做一个间歇性禁食计时器，显示进食窗口和禁食时段"),
    ("health_timer", "做一个喝水提醒计时器，每隔设定时间提醒喝水"),
    ("health_timer", "做一个护眼20-20-20计时器，每20分钟提醒看20米外20秒"),
    ("health_timer", "做一个高强度间歇训练计时器，运动和休息自动交替"),
    ("health_timer", "做一个冥想引导呼吸计时器，吸气4秒屏气7秒呼气8秒"),

    # ===== 生活 × 计算器 =====
    ("lifestyle_calculator", "做一个外卖费用计算器，输入菜价加配送费算总价"),
    ("lifestyle_calculator", "做一个节日礼金预算计算器，输入人数和关系推荐金额"),
    ("lifestyle_calculator", "做一个装修面积费用计算器，输入面积和单价"),
    ("lifestyle_calculator", "做一个超市价格对比器，输入两款商品的克数和价格"),
    ("lifestyle_calculator", "做一个打车费用估算器，输入公里数算大概费用"),

    # ===== 娱乐 × 记录 =====
    ("entertainment_tracker", "做一个电影观看记录器，可以打星评分和标注是否看过"),
    ("entertainment_tracker", "做一个游戏成就收集记录，可以标记哪些成就已解锁"),
    ("entertainment_tracker", "做一个每日签到打卡+随机奖励，打卡获得随机励志语"),
    ("entertainment_tracker", "做一个心情日历+趋势图，显示本月情绪分布"),
    ("entertainment_tracker", "做一个书单追踪器，记录想读已读在读状态"),

    # ===== 理财 × 生活 =====
    ("finance_lifestyle", "做一个咖啡消费追踪器，记录每次喝咖啡的花费"),
    ("finance_lifestyle", "做一个每日零花钱预算追踪，设定日预算显示剩余"),
    ("finance_lifestyle", "做一个拼单AA计算器，多人各自选了多少钱自动分摊"),
    ("finance_lifestyle", "做一个外出就餐消费记录，分类统计外卖和堂食花费"),
    ("finance_lifestyle", "做一个购物冲动冷静器，输入想买的东西30天后提醒"),

    # ===== 育儿 × 游戏 =====
    ("parenting_game", "做一个亲子问答游戏，家长出题孩子抢答"),
    ("parenting_game", "做一个儿童数数游戏，点击苹果数数量答对加分"),
    ("parenting_game", "做一个宝宝认颜色游戏，显示颜色块让孩子选对应名字"),
    ("parenting_game", "做一个亲子石头剪刀布，有可爱动画和欢呼效果"),
    ("parenting_game", "做一个儿童拼音学习游戏，点击拼音积木拼出汉字"),
]

GENERATION_PROMPT = """请根据以下用户需求，直接输出一个完整的、可独立运行的HTML文件。

严格要求：
1. 完整HTML文件，从<!DOCTYPE html>开始到</html>结束
2. 必须包含 <meta name="viewport" content="width=device-width,initial-scale=1.0">
3. 所有CSS内联在<head>的<style>标签中
4. 所有JavaScript内联在<body>末尾的<script>标签中
5. 不引用任何外部CDN、库或API（纯原生HTML/CSS/JS）
6. 按钮padding≥12px，字体≥16px，max-width:380px居中卡片布局
7. body用linear-gradient渐变背景，卡片+圆角+阴影
8. 功能逻辑完整，所有交互都能正常工作
9. 中文界面，标题用emoji装饰
10. 只输出纯HTML代码，不要任何解释

用户需求：{instruction}"""


def generate_html(instruction: str) -> str | None:
    try:
        resp = client.chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": GENERATION_PROMPT.format(instruction=instruction)}],
            temperature=0.4,
            max_tokens=3500,
        )
        code = resp.choices[0].message.content.strip()
        if code.startswith("```"):
            code = re.sub(r'^```[a-zA-Z]*\n?', '', code)
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        code = code.strip()
        if "<!DOCTYPE" in code.upper() and "</html>" in code.lower() and "viewport" in code:
            return code
        return None
    except Exception as e:
        print(f"    [API错误] {str(e)[:100]}")
        return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 断点续传
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done_keys.add(item.get("_source_key", ""))
        print(f"♻️  已有 {len(done_keys)} 条，将跳过\n")

    print(f"📋 跨类组合指令数：{len(CROSS_CATEGORY_TOOLS)}")
    print(f"🤖 模型：{MODEL} @ {BASE_URL}\n")

    success = 0
    fail = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i, (cross_cat, instruction) in enumerate(CROSS_CATEGORY_TOOLS):
            key = f"cross_{i}"
            if key in done_keys:
                success += 1
                continue

            print(f"[{i+1}/{len(CROSS_CATEGORY_TOOLS)}][{cross_cat}] {instruction[:45]}...")
            code = generate_html(instruction)

            if code:
                entry = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": instruction},
                        {"role": "assistant", "content": code},
                    ],
                    "_source_key": key,
                    "_category":   "cross_category",
                    "_cross_type": cross_cat,
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()
                success += 1
                print(f"    ✅ ({len(code)} 字符)")
            else:
                fail += 1
                print(f"    ❌ 失败")

            time.sleep(0.8)

    print(f"\n{'='*50}")
    print(f"✅ 01c 完成！成功: {success} | 失败: {fail}")
    print(f"📄 输出：{OUTPUT_FILE}")
    print(f"⏭️  下一步：python scripts/01d_merge_and_dedupe.py")


if __name__ == "__main__":
    main()
