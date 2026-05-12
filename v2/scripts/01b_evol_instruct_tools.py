#!/usr/bin/env python3
"""
PocketVibe V2 — 01b: Evol-Instruct 工具数据生成
=================================================================
参考 WizardCoder (ICLR 2024) 的 Evol-Instruct 方法
对 150 个基础功能描述做 4 方向进化 + 直接生成完整 HTML
支持断点续传（中途断了直接重新运行）

输出：data/processed/evol_tools.jsonl（约 750 条）
=================================================================
运行：python scripts/01b_evol_instruct_tools.py
"""
import json, os, time, re
from openai import OpenAI

API_KEY  = "sk-10da60b9c960415992756ade04853606"
BASE_URL = "https://api.deepseek.com"
MODEL    = "deepseek-chat"
OUTPUT_FILE = "data/processed/evol_tools.jsonl"

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================================================================
# 150 条基础工具指令（覆盖 V1 的 90 条 + 新增 60 条）
# 分 15 个类别，便于后续 held-out 切分
# ================================================================
BASE_TOOLS = [
    # ===== 计时器类 (category: timer) =====
    ("timer", "做一个倒计时器，可以设定分钟数，有开始暂停和重置按钮"),
    ("timer", "帮我做个秒表，有开始停止和清零功能"),
    ("timer", "做一个番茄钟，25分钟工作5分钟休息，自动轮次切换"),
    ("timer", "做一个组间休息计时器，可以设定休息秒数"),
    ("timer", "做一个拉伸计时器，有多个动作每个30秒自动切换"),
    ("timer", "做一个演讲计时器，分段提醒绿黄红灯"),
    ("timer", "做一个烹饪多段计时器，可以同时设多个倒计时"),
    ("timer", "做一个番茄钟统计工具，记录每天完成了多少个番茄"),
    ("timer", "做一个讲故事计时器，可以设定时间提醒家长睡前故事"),
    ("timer", "做一个会议时长计算器，记录每个发言者的时间"),

    # ===== 计算器类 (category: calculator) =====
    ("calculator", "做一个简单的计算器，加减乘除"),
    ("calculator", "做一个BMI计算器，输入身高体重就能算"),
    ("calculator", "做一个小费计算器，支持人数分摊"),
    ("calculator", "做一个百分比计算器"),
    ("calculator", "做一个面积计算器，支持正方形长方形和圆形"),
    ("calculator", "做一个折扣计算器，输入原价和折扣算最终价格"),
    ("calculator", "帮我做个AA制算账工具"),
    ("calculator", "做一个房贷计算器，支持等额本息和等额本金"),
    ("calculator", "做一个复利计算器，输入本金利率和年数"),
    ("calculator", "做一个工资税后计算器，输入税前工资算到手多少"),

    # ===== 换算工具类 (category: converter) =====
    ("converter", "做一个摄氏和华氏温度换算器"),
    ("converter", "做一个厘米和英寸互转工具"),
    ("converter", "做一个公斤和磅的换算"),
    ("converter", "做一个进制转换器，支持二进制十进制十六进制"),
    ("converter", "做一个汇率换算器，人民币和美元互转"),
    ("converter", "做个鞋码换算器，中国码美国码欧洲码互转"),
    ("converter", "做一个RGB和HEX颜色值互转工具"),
    ("converter", "做个数据存储单位换算，B KB MB GB TB互转"),
    ("converter", "做一个功率单位换算器，瓦特马力千瓦互转"),
    ("converter", "做一个烘焙温度换算器，华氏和摄氏互转"),

    # ===== 游戏类 (category: game) =====
    ("game", "做一个猜数字游戏，1到100之间猜"),
    ("game", "做一个石头剪刀布游戏"),
    ("game", "做一个颜色记忆游戏，记住顺序后点击"),
    ("game", "做一个打字速度测试工具"),
    ("game", "做一个反应速度测试，点击绿色方块"),
    ("game", "做一个井字棋游戏，两个人轮流下"),
    ("game", "帮我做个简易打地鼠游戏，随机出现点击得分"),
    ("game", "做一个记忆翻牌游戏，翻两张相同的就消除"),
    ("game", "做个数字华容道游戏，把数字排成正确顺序"),
    ("game", "做一个骰子游戏，点击按钮随机出1到6点数"),

    # ===== 记录追踪类 (category: tracker) =====
    ("tracker", "做一个待办事项列表，可以添加勾选和删除"),
    ("tracker", "做一个简易记账本，可以记收入和支出"),
    ("tracker", "做一个打卡日历，点击日期标记已完成"),
    ("tracker", "做一个喝水记录工具，目标8杯"),
    ("tracker", "做一个习惯打卡工具，可以添加多个习惯"),
    ("tracker", "做一个心情记录器，每天选一个心情保存"),
    ("tracker", "做一个购物清单，可以添加物品标记已购买"),
    ("tracker", "做一个宝宝喂奶记录器，记录时间和喂奶量"),
    ("tracker", "做一个体重变化追踪器，记录每天体重"),
    ("tracker", "做一个血压记录工具，可以输入收缩压舒张压"),

    # ===== 健康类 (category: health) =====
    ("health", "做一个卡路里计算器，选择食物和份量算热量"),
    ("health", "做一个体脂率计算器，输入身高体重腰围计算"),
    ("health", "做一个运动消耗热量计算器，选择运动类型和时间"),
    ("health", "做一个心率区间计算工具，输入年龄算训练区间"),
    ("health", "做一个跑步配速计算器，输入距离和时间算配速"),
    ("health", "帮我做个睡眠时长计算器，输入入睡和起床时间"),
    ("health", "做一个呼吸练习引导工具，有吸气屏气呼气步骤"),
    ("health", "做一个密码强度检测工具，输入密码实时显示强度"),
    ("health", "做一个年龄计算器，输入生日算出年龄"),
    ("health", "做个怀孕周数计算器，输入末次月经日期算孕周"),

    # ===== 生活工具类 (category: lifestyle) =====
    ("lifestyle", "做一个随机选择器，帮我决定今天吃什么"),
    ("lifestyle", "做一个倒数日工具，输入日期显示还有几天"),
    ("lifestyle", "做一个字数统计工具"),
    ("lifestyle", "做一个简单的备忘录工具，支持本地保存"),
    ("lifestyle", "做一个人生进度条，显示今年已过去多少"),
    ("lifestyle", "帮我做个节日倒数工具，显示到春节中秋国庆的天数"),
    ("lifestyle", "做个旅行行李打包清单"),
    ("lifestyle", "做一个穿衣建议工具，输入温度推荐穿什么"),
    ("lifestyle", "做个简易电费计算器，输入用电量和单价"),
    ("lifestyle", "做一个情侣纪念日计算器，显示在一起多少天"),

    # ===== 教育学习类 (category: education) =====
    ("education", "做一个闪卡记忆工具，可以翻转查看答案"),
    ("education", "做一个九九乘法表，可以测验"),
    ("education", "做一个GPA计算器，输入各科成绩和学分"),
    ("education", "做个数学口算练习工具，随机出加减乘除题目"),
    ("education", "做个古诗词背诵卡片，显示上句填下句"),
    ("education", "做一个成绩等级转换器，百分制转ABCD等级"),
    ("education", "做一个考试倒计时工具，可以设置多个考试日期"),
    ("education", "做一个乘法口诀练习游戏，答对加分"),
    ("education", "做一个化学元素周期表速查工具"),
    ("education", "做一个单词记忆卡片，显示英文单词让用户猜中文"),

    # ===== 创意娱乐类 (category: entertainment) =====
    ("entertainment", "做一个随机密码生成器"),
    ("entertainment", "做一个随机颜色生成器"),
    ("entertainment", "做一个抛硬币的小工具"),
    ("entertainment", "做一个随机数生成器，可以设定范围"),
    ("entertainment", "做一个真心话大冒险工具，随机出题"),
    ("entertainment", "做一个幸运转盘，可以添加选项然后转动"),
    ("entertainment", "做一个随机表白情话生成器，粉色可爱风格"),
    ("entertainment", "做一个星座速查工具，输入生日显示星座"),
    ("entertainment", "做一个塔罗牌抽取工具，随机展示牌面"),
    ("entertainment", "做个生肖年份查询器，输入年份显示生肖"),

    # ===== 绘画创作类 (category: creative) =====
    ("creative", "做一个简易画板可以画画，有颜色选择和橡皮擦"),
    ("creative", "做一个评分卡，可以给1到5星评分"),
    ("creative", "做一个简易日记本，可以保存每日记录"),
    ("creative", "做一个白噪音播放器，有几种不同环境音"),
    ("creative", "做一个节拍器，可以调节BPM"),
    ("creative", "做一个颜色调色板，可以混合颜色查看效果"),
    ("creative", "做一个随机故事开头生成器，点击获取新开头"),
    ("creative", "做一个心情日历，可以给每天涂不同颜色"),
    ("creative", "做一个文字加密解密工具，简单凯撒密码"),
    ("creative", "做一个随机背景渐变生成器，每次刷新换一个"),

    # ===== 社交团队类 (category: social) =====
    ("social", "做一个投票器，可以添加选项让大家投票"),
    ("social", "做一个记分板，两队计分可以加减分"),
    ("social", "做一个班级随机分组工具，输入人名和组数"),
    ("social", "做一个抽奖工具，输入名单随机抽出中奖者"),
    ("social", "帮我做个红包金额随机分配器"),
    ("social", "做一个座位随机安排器，输入人名随机排座位"),
    ("social", "做一个谁是卧底词语生成器，给出平民词和卧底词"),
    ("social", "做一个团建游戏随机选择器，内置多种游戏"),
    ("social", "做一个简单投票工具，实时显示百分比"),
    ("social", "做一个随机抽签工具，输入名字列表随机抽"),

    # ===== 金融理财类 (category: finance) =====
    ("finance", "做一个信用卡分期计算器，算出分期总利息"),
    ("finance", "做一个365天存钱挑战，第1天存1块第2天存2块"),
    ("finance", "做一个记账本，分类显示收入支出"),
    ("finance", "做一个预算分配器，输入月收入按比例分配"),
    ("finance", "做一个存款目标追踪器，设定目标金额显示进度"),
    ("finance", "做个简单股票收益计算器，输入买入卖出价"),
    ("finance", "做一个零钱凑整工具，输入总金额和面额"),
    ("finance", "做一个贷款利率对比工具，对比不同利率还款额"),
    ("finance", "做个理财收益计算器，输入年化收益率和本金"),
    ("finance", "做一个汇率换算工具，支持多种货币"),

    # ===== 育儿亲子类 (category: parenting) =====
    ("parenting", "做一个宝宝月龄计算器，输入出生日期显示月龄"),
    ("parenting", "帮我做个儿童涂色板，有简单图案可以选颜色"),
    ("parenting", "做一个乘法口诀练习游戏，适合小朋友"),
    ("parenting", "帮我做个儿童认数字游戏，点击对应数量的星星"),
    ("parenting", "做一个讲故事计时器，睡前故事时间控制"),
    ("parenting", "做一个儿童单词记忆游戏，配图单词卡片"),
    ("parenting", "做一个宝宝成长里程碑记录工具"),
    ("parenting", "做一个儿童视力测试，显示不同大小字母"),
    ("parenting", "做一个数数练习工具，显示图片让孩子数数量"),
    ("parenting", "做一个儿童番茄学习时钟，可爱风格"),

    # ===== 日程规划类 (category: planner) =====
    ("planner", "做一个日程表，可以给每个小时添加事项"),
    ("planner", "做一个课程表，可以填写每天每节课内容"),
    ("planner", "做一个每日目标清单，每天可以重置"),
    ("planner", "做一个周计划工具，七天每天可以填写事项"),
    ("planner", "做一个时区换算器，选两个城市显示时间差"),
    ("planner", "做一个倒计时日历，标记重要日期"),
    ("planner", "做一个时间块规划工具，拖拽分配时间块"),
    ("planner", "做一个每日反思日记，固定几个提问"),
    ("planner", "做一个读书清单，记录想读和已读的书"),
    ("planner", "做一个年度目标追踪器，12个月分别填写目标"),

    # ===== 工具效率类 (category: utility) =====
    ("utility", "做一个文本去重工具，粘贴文本自动去除重复行"),
    ("utility", "做一个二维码生成器，输入文字或链接生成二维码"),
    ("utility", "做一个简易Markdown预览，左边输入右边预览"),
    ("utility", "帮我做个密码强度检测工具"),
    ("utility", "做一个专注力测试，显示颜色文字让用户快速回答"),
    ("utility", "做一个随机IP地址生成器"),
    ("utility", "做一个字符编码工具，URL编码和Base64转换"),
    ("utility", "做一个简单的正则表达式测试工具"),
    ("utility", "做一个时间戳转换工具，Unix时间戳和日期互转"),
    ("utility", "做一个颜色对比度检测工具，输入两个颜色算对比度"),
]

# 四种 Evol-Instruct 进化策略
EVOL_STRATEGIES = [
    {
        "name": "depth",
        "prompt": """你是一个指令进化器。将以下手机小工具需求进行【深度进化】：
在原有功能基础上添加1-2个新的功能约束或细节要求，使需求更加具体复杂，但仍是同类工具。
只输出进化后的需求描述，不要任何解释。控制在40字以内。

原始需求：{instruction}"""
    },
    {
        "name": "breadth",
        "prompt": """你是一个指令进化器。将以下手机小工具需求进行【广度进化】：
保持核心功能不变，但加入一个具体的使用场景或用户身份，使需求更有代入感。
只输出进化后的需求描述，不要任何解释。控制在40字以内。

原始需求：{instruction}"""
    },
    {
        "name": "reasoning",
        "prompt": """你是一个指令进化器。将以下手机小工具需求进行【推理逻辑进化】：
在原有功能基础上添加一个需要条件判断或状态反馈的逻辑需求（如：超过某阈值提醒、错误输入提示等）。
只输出进化后的需求描述，不要任何解释。控制在50字以内。

原始需求：{instruction}"""
    },
    {
        "name": "combination",
        "prompt": """你是一个指令进化器。将以下手机小工具需求进行【组合进化】：
把原有功能与另一个相关但不同的小功能融合，创造一个"二合一"的工具需求。
只输出进化后的需求描述，不要任何解释。控制在50字以内。

原始需求：{instruction}"""
    },
]

GENERATION_PROMPT = """请根据以下用户需求，直接输出一个完整的、可独立运行的HTML文件。

严格要求：
1. 完整HTML文件，从<!DOCTYPE html>开始到</html>结束
2. 必须包含 <meta name="viewport" content="width=device-width,initial-scale=1.0">
3. 所有CSS内联在<head>的<style>标签中（不能引用外部CSS）
4. 所有JavaScript内联在<body>末尾的<script>标签中（不能引用外部JS）
5. 不引用任何外部CDN、库、字体或API（纯原生HTML/CSS/JS）
6. 按钮padding≥12px，字体≥16px，max-width:380px居中卡片布局
7. body用linear-gradient渐变背景，卡片白色/深色+圆角+阴影
8. 功能逻辑完整正确，所有按钮都有对应JavaScript功能
9. 中文界面，标题用emoji装饰
10. 只输出纯HTML代码，不要任何解释或Markdown标记

用户需求：{instruction}"""


def evolve_instruction(instruction: str, strategy: dict) -> str | None:
    """用API进化一条指令"""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": strategy["prompt"].format(instruction=instruction)}],
            temperature=0.8,
            max_tokens=150,
        )
        evolved = resp.choices[0].message.content.strip()
        # 去掉可能的引号
        evolved = evolved.strip('"\'「」')
        return evolved if len(evolved) >= 5 else None
    except Exception as e:
        print(f"    [进化错误] {e}")
        return None


def generate_html(instruction: str, retries: int = 2) -> str | None:
    """生成完整HTML"""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
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
            if ("<!DOCTYPE" in code.upper()) and "</html>" in code.lower() and "viewport" in code:
                return code
            elif attempt < retries:
                time.sleep(1)
        except Exception as e:
            print(f"    [生成错误] {e}")
            if attempt < retries:
                time.sleep(3)
    return None


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 断点续传：加载已完成的 key
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done_keys.add(item.get("_source_key", ""))
        print(f"♻️  已有 {len(done_keys)} 条，将跳过\n")

    print(f"📋 基础工具数：{len(BASE_TOOLS)}")
    print(f"🔄 进化策略数：{len(EVOL_STRATEGIES)}")
    print(f"🎯 目标：{len(BASE_TOOLS)} 基础 + {len(BASE_TOOLS)*len(EVOL_STRATEGIES)} 进化 = {len(BASE_TOOLS)*(1+len(EVOL_STRATEGIES))} 条\n")

    success = 0
    fail = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:

        for i, (category, base_instr) in enumerate(BASE_TOOLS):

            # --- 1. 先生成基础版（不进化）---
            base_key = f"base_{i}"
            if base_key not in done_keys:
                print(f"[{i+1}/{len(BASE_TOOLS)}][base][{category}] {base_instr[:38]}...")
                code = generate_html(base_instr)
                if code:
                    entry = {
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": base_instr},
                            {"role": "assistant", "content": code},
                        ],
                        "_source_key": base_key,
                        "_evol_type":  "base",
                        "_category":   category,
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()
                    success += 1
                    print(f"    ✅ 基础版 ({len(code)} 字符)")
                else:
                    fail += 1
                    print(f"    ❌ 基础版失败")
                time.sleep(0.8)

            # --- 2. 四方向进化 ---
            for strategy in EVOL_STRATEGIES:
                evol_key = f"evol_{i}_{strategy['name']}"
                if evol_key in done_keys:
                    success += 1
                    continue

                # Step 1: 进化指令
                evolved = evolve_instruction(base_instr, strategy)
                if not evolved:
                    fail += 1
                    continue
                time.sleep(0.3)

                # Step 2: 根据进化后的指令生成 HTML
                print(f"[{i+1}/{len(BASE_TOOLS)}][{strategy['name']}] {evolved[:45]}...")
                code = generate_html(evolved)
                if code:
                    entry = {
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": evolved},
                            {"role": "assistant", "content": code},
                        ],
                        "_source_key":    evol_key,
                        "_evol_type":     strategy["name"],
                        "_base_instr":    base_instr,
                        "_category":      category,
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()
                    success += 1
                    print(f"    ✅ ({len(code)} 字符)")
                else:
                    fail += 1
                    print(f"    ❌ 生成失败")

                time.sleep(0.8)

            # 每 10 条打印进度
            if (i + 1) % 10 == 0:
                print(f"\n--- 进度 {i+1}/{len(BASE_TOOLS)} | 成功:{success} 失败:{fail} ---\n")

    print(f"\n{'='*50}")
    print(f"✅ 01b 完成！成功: {success} | 失败: {fail}")
    print(f"📄 输出：{OUTPUT_FILE}")
    print(f"⏭️  下一步：python scripts/01c_cross_category.py")


if __name__ == "__main__":
    main()
