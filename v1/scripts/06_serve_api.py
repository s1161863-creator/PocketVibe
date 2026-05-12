#!/usr/bin/env python3
"""
PocketVibe — 推理 API 服务 v3
改进: 黄金模板兜底 / 扩展指令补全 / 静态文件服务 / 不依赖 Gradio
运行: sbatch slurm/serve.slurm
访问: http://localhost:8000 (手机UI) | POST /generate (API)
"""
import os, re, json, torch
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

os.environ["HF_HOME"]        = "/opt/shared/model-cache"
os.environ["HF_HUB_OFFLINE"] = "1"

BASE   = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPT  = os.path.expanduser("~/PocketVibe/outputs/qlora-run2/final_adapter")
APP_DIR = os.path.expanduser("~/PocketVibe/app")

# System prompt — 与训练时完全一致
SYSTEM = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# ══════════════════════════════════════════════
# 黄金模板：从种子数据中提取的已验证 HTML
# 当用户指令匹配到这些关键词时，直接返回验证过的代码
# ══════════════════════════════════════════════
GOLDEN_TEMPLATES = {}

def _load_golden_templates():
    """从种子数据文件加载黄金模板"""
    seed_path = os.path.expanduser("~/PocketVibe/data/seed/seed_examples.jsonl")
    if not os.path.exists(seed_path):
        print(f"⚠ 种子文件不存在: {seed_path}")
        return
    with open(seed_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                msgs = item["messages"]
                instruction = msgs[1]["content"]
                html = msgs[2]["content"]
                GOLDEN_TEMPLATES[instruction] = html
            except:
                continue
    print(f"✅ 加载了 {len(GOLDEN_TEMPLATES)} 条黄金模板")

# 模板匹配关键词 → 种子指令映射
TEMPLATE_KEYWORDS = {
    "番茄钟": "做一个番茄钟，25分钟工作5分钟休息，自动轮次切换",
    "画板": "做一个简易画板可以画画",
    "记账": "做一个简易记账本，可以记收入和支出",
    "喝水": "做个喝水提醒工具，记录每天喝了几杯水，目标8杯",
    "密码生成": "做一个随机密码生成器",
    "打字速度": "做一个打字速度测试工具",
    "待办事项": "做一个待办事项列表，可以添加勾选和删除",
    "节拍器": "做一个节拍器，可以调节BPM",
    "倒计时": "做一个倒计时器，可以设定分钟数，有开始暂停和重置按钮",
    "秒表": "帮我做个秒表，有开始停止和清零功能",
    "骰子": "做一个掷骰子工具，点击按钮随机出1到6的点数",
    "硬币": "弄个抛硬币的小工具",
    "吃什么": "做一个随机选择器，帮我决定今天吃什么",
    "BMI": "帮我做一个BMI计算器，输入身高体重就能算",
    "温度换算": "做一个摄氏和华氏温度的换算器",
    "记分板": "做一个两队记分板，可以加分减分",
    "猜数字": "做个猜数字游戏，1到100之间猜",
    "计算器": "做个简单的计算器，加减乘除",
    "年龄计算": "做个年龄计算器，输入生日算出年龄",
    "百分比": "帮我做个百分比计算器",
    "倒数日": "做一个倒数日工具，输入日期显示还有几天",
    "随机颜色": "做个随机颜色生成器",
    "石头剪刀布": "做一个石头剪刀布游戏",
    "计数器": "做一个简单计数器，可以加一减一",
    "字数统计": "做一个字数统计工具",
    "随机数": "做一个随机数生成器，可以设定范围",
    "进制转换": "帮我做一个进制转换器",
    "小费": "做一个小费计算器",
    "AA": "做一个AA制算账工具，输入总金额和人数",
    "面积": "做一个面积计算器，支持正方形长方形和圆形",
    "反应速度": "做个反应速度测试",
    "投票": "弄个简单投票工具，可以添加选项让大家投票",
    "折扣": "做一个折扣计算器",
    "汇率": "做个汇率换算器，人民币和美元互转",
    "打卡日历": "做一个打卡日历，点击日期标记已完成",
    "习惯打卡": "做一个习惯打卡工具，可以添加多个习惯每天打卡",
    "闪卡": "帮我做一个闪卡记忆工具，可以翻转查看答案",
    "购物清单": "做一个购物清单，可以添加物品和标记已购买",
    "心情": "做一个心情记录器，每天选一个心情保存",
    "备忘录": "做个简单的备忘录工具",
    "日程": "帮我做一个日程表，可以给每个小时添加事项",
    "评分": "做一个评分卡，可以给1到5星评分",
    "日记": "做一个简易日记本",
    "呼吸": "做一个呼吸练习引导工具",
    "颜色记忆": "做一个颜色记忆游戏",
    "白噪音": "做一个白噪音播放器，有几种不同的环境音效果",
}

def find_golden_template(instruction: str) -> str | None:
    """尝试从黄金模板中匹配"""
    inst_lower = instruction.lower()
    for keyword, seed_instruction in TEMPLATE_KEYWORDS.items():
        if keyword in inst_lower or keyword in instruction:
            if seed_instruction in GOLDEN_TEMPLATES:
                return GOLDEN_TEMPLATES[seed_instruction]
    # 精确匹配
    if instruction in GOLDEN_TEMPLATES:
        return GOLDEN_TEMPLATES[instruction]
    return None


# ══════════════════════════════════════════════
# 轻量级指令补全规则（扩展版）
# ══════════════════════════════════════════════
ENHANCE_RULES = [
    (r"番茄[钟锺]", "，25分钟工作5分钟休息，有开始暂停和重置按钮，显示当前倒计时和轮次切换"),
    (r"倒计时", "，可以设定分钟数，有开始暂停和重置按钮"),
    (r"计算器", "，支持加减乘除，有清除和退格按钮"),
    (r"记分板", "，可以加分减分和重置"),
    (r"待办|todo", "，可以添加勾选和删除任务"),
    (r"秒表", "，有开始停止和清零功能"),
    (r"骰子", "，点击按钮随机出1到6的点数"),
    (r"硬币", "，点击抛出显示正面或反面"),
    (r"密码生成", "，可以调节长度，包含大小写字母数字和特殊字符"),
    (r"BMI", "，输入身高体重就能算，显示BMI值和健康状态"),
    (r"猜数字", "，1到100之间猜，提示大了还是小了"),
    (r"石头剪刀布", "，和电脑对战，显示胜负统计"),
    (r"记账|账本", "，可以记收入和支出，显示余额"),
    (r"打卡|习惯", "，可以添加习惯每天打卡，显示连续天数"),
    (r"温度换算", "，摄氏和华氏互转"),
    (r"字数统计", "，输入文字显示字符数词数和不含空格数"),
    (r"喝水", "，记录每天喝了几杯水，目标8杯，显示进度"),
    (r"画板", "，可以选颜色调粗细，有橡皮擦和清除功能"),
    (r"日记", "，可以写日记保存到本地，显示历史记录"),
    (r"备忘录", "，输入文字保存到本地，下次打开还在"),
    (r"日程", "，可以给每个小时添加事项"),
    (r"评分", "，可以给1到5星评分，显示评分结果"),
    (r"闪卡", "，显示问题点击翻转查看答案，可以切换上下张"),
    (r"购物清单", "，可以添加物品和标记已购买"),
    (r"心情", "，选择表情记录心情，显示历史"),
    (r"节拍器", "，可以调节BPM，有声音和视觉反馈"),
    (r"呼吸练习", "，吸气屏住呼气的引导动画"),
    (r"打字[速度测试]", "，显示文字让用户打字，计算WPM"),
    (r"颜色记忆", "，记住颜色顺序然后按顺序点击"),
    (r"白噪音", "，有雨声海浪风声等不同环境音"),
    (r"食谱|冰箱|菜", "，可以添加食材，有肉类蔬菜类主食类分类，显示各分类数量"),
    (r"卡路里|热量", "，选择食物和份量算出热量"),
    (r"血压", "，输入收缩压舒张压和心率，显示记录"),
    (r"体重", "，记录每天体重显示变化趋势"),
    (r"睡眠", "，输入入睡和起床时间算出睡眠时长"),
    (r"房贷", "，支持等额本息和等额本金两种方式"),
    (r"复利", "，输入本金利率和年数算出最终金额"),
    (r"GPA|绩点", "，输入各科成绩和学分算出绩点"),
    (r"乘法口诀|口算", "，随机出题计时作答"),
    (r"课程表", "，可以填写每天每节课的内容"),
    (r"纪念日", "，输入日期显示天数和各种纪念日"),
    (r"人生进度", "，显示今年已过去百分之多少"),
    (r"真心话大冒险", "，随机出题目"),
    (r"抽签", "，输入名字列表随机抽取"),
    (r"抽奖", "，输入参与者名单随机抽出中奖者"),
    (r"分组", "，输入人名和组数自动分组"),
    (r"井字棋", "，可以两个人轮流下"),
    (r"演讲计时", "，分段提醒绿黄红灯"),
    (r"密码强度", "，输入密码实时显示强度"),
]

def enhance_instruction(raw: str) -> str:
    """对过短或模糊的指令进行轻量补全"""
    raw = raw.strip()
    if len(raw) > 25:
        return raw
    for pattern, suffix in ENHANCE_RULES:
        if re.search(pattern, raw):
            if suffix[:5] not in raw:
                return raw + suffix
    # 兜底：对任何短指令追加通用约束
    if len(raw) < 15:
        return raw + "，界面美观，功能完整，有清晰的操作按钮"
    return raw


# ══════════════════════════════════════════════
# 加载模型
# ══════════════════════════════════════════════
print(">>> 加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(">>> 加载基座模型 (bf16)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
print(">>> 挂载 LoRA 适配器...")
model = PeftModel.from_pretrained(base_model, ADAPT)
model.eval()
print(">>> 模型就绪")

# 加载黄金模板
_load_golden_templates()
print()


# ══════════════════════════════════════════════
# FastAPI
# ══════════════════════════════════════════════
app = FastAPI(title="PocketVibe", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    instruction: str
    max_tokens: int = 4000


class GenerateResponse(BaseModel):
    html: str
    char_count: int
    source: str = ""
    enhanced_instruction: str = ""


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """返回前端页面 (禁用缓存, 不设CSP让浏览器用默认策略)"""
    html_path = os.path.join(APP_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    instruction = req.instruction.strip()

    # 1. 尝试黄金模板匹配
    golden = find_golden_template(instruction)
    if golden:
        print(f"  [模板命中] {instruction[:40]}")
        return GenerateResponse(
            html=golden, char_count=len(golden),
            source="golden_template", enhanced_instruction=instruction,
        )

    # 2. 轻量级指令补全
    enhanced = enhance_instruction(instruction)
    print(f"  [模型生成] {enhanced[:50]}")

    # 3. 模型推理
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": enhanced},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inp = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=min(req.max_tokens, 4096),
            do_sample=False,
            repetition_penalty=1.1,
        )

    code = tokenizer.decode(
        out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()

    # 截断到 </html>
    end_idx = code.lower().rfind("</html>")
    if end_idx != -1:
        code = code[:end_idx + 7]

    return GenerateResponse(
        html=code, char_count=len(code),
        source="model", enhanced_instruction=enhanced,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Qwen2.5-Coder-1.5B + LoRA",
        "golden_templates": len(GOLDEN_TEMPLATES),
        "version": "3.0",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
