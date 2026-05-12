#!/usr/bin/env python3
"""
PocketVibe V2+ — 细粒度评分模块（被 04 / 05 脚本共同引用）
=================================================================
本模块定义：
  1. 5 维 100 分制评分函数 score_html_100()
  2. 5 条 V2 倾向的高难度测试用例 TEST_CASES
  3. 共享的推理工具函数（Best-of-3 + 贪心兜底 + 代码清洗）

【为什么不再用二值 6 维评分】
  旧版的 runnable/viewport/mobile/no_cdn/has_style/has_script 都是 0/1 判断，
  任何稍微能跑的 HTML 都能顶格 6/6（参考 logs/1443_eval.out），
  无法区分"能跑"和"功能完整"的差距。

【新评分维度（总 100 分）】
  | 维度 | 权重 | 评估要点                             | 参考                        |
  |------|------|--------------------------------------|-----------------------------|
  | J    | 30   | JS 深度：事件/状态/控制流/API/函数    | KRT2002 Python 指标 14 项   |
  | I    | 25   | 指令遵循：每个关键功能点是否落地      | Design2Code Text/Pos Match  |
  | F    | 20   | 功能可运行：DOM/script 闭合/无语法错  | Codex human-eval execution  |
  | C    | 15   | CSS 质量：viewport/适配/现代感        | Design2Code Color/Style     |
  | S    | 10   | 结构合规：DOCTYPE/lang/charset/title  | HTML5 规范                  |

【参考文献】
  - Si, C. et al. (2025). Design2Code: Benchmarking Multimodal Code Generation.
    NAACL 2025. https://aclanthology.org/2025.naacl-long.199/
  - Lu, Z. et al. (2025). WebDevJudge: Evaluating (M)LLMs as Critiques for
    Web Development Quality. arXiv:2510.18560 (IAA 89.7%).
  - KRT2002 (2024). qwen-python-finetuning 14-indicator evaluation framework.
    https://github.com/KRT2002/qwen-python-finetuning
  - Qwen official model card: temperature=0.7, top_p=0.8, top_k=20.
  - Li, Y. et al. (2022). Competition-Level Code Generation with AlphaCode.
    Science. (Best-of-N sampling)
=================================================================
"""
import os, re, torch

# ================================================================
# 1. 公共常量
# ================================================================
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# Qwen2.5 官方推荐 + 代码生成禁用 repetition_penalty
SAMPLE_KWARGS = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True,
    repetition_penalty=1.0,
    max_new_tokens=4096,   # 给复杂题足够空间，避免截断引入假象（参考 README §10.3）
)
GREEDY_KWARGS = dict(
    do_sample=False,
    max_new_tokens=4096,
    repetition_penalty=1.0,
)
BEST_OF_N = 3
MIN_ACCEPT_SCORE = 60   # Best-of-N 期间 100 分制下 >=60 算可接受，直接停止采样


# ================================================================
# 2. 5 条 V2 倾向的复杂测试用例（覆盖 Evol-Instruct 四方向 + 跨类组合）
# ================================================================
# 每条用例的 expected_features 格式：
#   ("功能描述", [keyword1, keyword2, ...])
# 评分时：代码中（HTML+JS 文本）包含任一 keyword 即算该功能点落地
TEST_CASES = [
    # ─── C1 DEPTH（深度进化）───
    {
        "tag":         "C1_depth_stopwatch_lap",
        "category":    "DEPTH",
        "difficulty":  "high",
        "instruction": ("做一个分段秒表，要求：支持开始/暂停/清零三个操作；"
                        "点击分段按钮可以记录当前圈速，最多保存10段；"
                        "每段显示与上一段的差值；自动高亮最快段（绿色）和最慢段（红色）。"),
        "expected_features": [
            ("开始/暂停控制",      ["start", "pause", "stop", "开始", "暂停", "停止"]),
            ("分段 Lap 功能",      ["lap", "split", "record", "分段", "记录", "圈"]),
            ("清零功能",          ["reset", "clear", "清零", "重置"]),
            ("10 段上限",          [">=10", "> 10", ">10", "length>=10", "length > 10", "length>10", "slice(-10", ".length === 10", "max.{0,10}10"]),
            ("与上段差值",         ["diff", "delta", "split", "prev", "last", "差", "-"]),
            ("最快段高亮",         ["fastest", "min", "best", "最快", "green", "#0f0", "#43e97b"]),
            ("最慢段高亮",         ["slowest", "max", "worst", "最慢", "red", "#f00", "#f5576c"]),
            ("时间格式化(分:秒)",  [":", "padStart", "toFixed", "%60"]),
        ],
    },
    # ─── C2 BREADTH（广度进化）───
    {
        "tag":         "C2_breadth_swim_timer",
        "category":    "BREADTH",
        "difficulty":  "high",
        "instruction": ("做一个游泳训练专用秒表，使用场景是湿手操作：字体要特别大（至少60px），"
                        "按钮要特别大（至少80px高度），配色用防水风格的深蓝/浅蓝渐变，"
                        "支持分段记录每一趟游泳的时间，显示总距离（按50米/趟估算）。"),
        "expected_features": [
            ("大字号时间显示",     ["font-size:.*60", "font-size:.*72", "font-size:.*80", "font-size:.*9", "font-size:.*1[0-9][0-9]", "3rem", "4rem", "5rem", "6rem"]),
            ("大按钮",             ["padding:.*2[0-9]", "padding:.*3[0-9]", "height:.*[6-9][0-9]", "min-height", "padding:16"]),
            ("深/浅蓝配色",        ["#0", "#1", "#2", "blue", "navy", "4facfe", "00f2fe", "667eea", "2196"]),
            ("渐变背景",           ["linear-gradient", "radial-gradient", "gradient"]),
            ("分段记录",           ["lap", "split", "trip", "分段", "趟", "lap"]),
            ("距离/米显示",         ["50", "米", "meter", "distance", "total", "distance"]),
            ("开始/暂停",          ["start", "pause", "stop", "开始", "暂停"]),
        ],
    },
    # ─── C3 REASONING（推理进化）───
    {
        "tag":         "C3_reasoning_calc_paren",
        "category":    "REASONING",
        "difficulty":  "high",
        "instruction": ("做一个科学计算器：支持 + - × ÷ 括号四则运算，"
                        "必须正确处理括号优先级和运算顺序；"
                        "当表达式有语法错误（如括号不匹配/连续运算符）时，"
                        "在结果区实时显示红色错误提示；支持回退键和清零键。"),
        "expected_features": [
            ("四则运算按钮",       ["+", "-", "*", "×", "/", "÷"]),
            ("括号按钮",           ["(", ")", "\\("]),
            ("括号优先级求值",     ["eval", "Function", "parse", "precedence", "stack", "priority"]),
            ("实时错误检测",       ["catch", "try", "error", "invalid", "错误", "NaN", "isNaN"]),
            ("错误红色提示",       ["red", "#f", "color:red", "color: red", "error", "错误"]),
            ("回退(backspace)",    ["backspace", "delete", "pop", "slice(0,-1)", "⌫", "退格"]),
            ("清零",              ["clear", "reset", "AC", "C", "清零"]),
            ("数字显示区",         ["display", "screen", "output", "result", "结果"]),
        ],
    },
    # ─── C4 COMBINATION（组合进化）───
    {
        "tag":         "C4_combination_todo_pomodoro",
        "category":    "COMBINATION",
        "difficulty":  "high",
        "instruction": ("做一个融合待办事项和番茄钟的工具："
                        "上半部分是待办列表，可以添加/勾选/删除任务；"
                        "点击任意待办项旁边的▶按钮，下半部分的番茄钟就自动以该任务名启动25分钟倒计时；"
                        "倒计时结束后自动把对应待办标记为完成。"),
        "expected_features": [
            ("添加待办",           ["add", "push", "新增", "添加", "onclick"]),
            ("勾选待办",           ["check", "done", "toggle", "completed", "勾选", "☑", "✅"]),
            ("删除待办",           ["remove", "splice", "delete", "删除", "×"]),
            ("启动按钮",           ["▶", "start", "play", "启动", "开始"]),
            ("25分钟倒计时",       ["25", "1500", "60*25", "25*60"]),
            ("计时器逻辑",         ["setInterval", "setTimeout", "Date", "--", "countdown"]),
            ("当前任务显示",       ["current", "active", "当前", "正在"]),
            ("结束自动标记",       ["complete", "done", "auto", "mark", "完成", "自动"]),
            ("显示时间",           ["padStart", ":", "toFixed", "format"]),
        ],
    },
    # ─── C5 CROSS_CATEGORY（跨类组合）───
    {
        "tag":         "C5_cross_rps_scoreboard",
        "category":    "CROSS_CATEGORY",
        "difficulty":  "high",
        "instruction": ("做一个石头剪刀布对战工具，但带实时记分板："
                        "用户选择石头/剪刀/布后，电脑随机出招，显示本轮胜负；"
                        "同时顶部记分板实时更新我方/电脑的累计胜场和平局次数；"
                        "支持一键重置分数；最近5局结果用小图标滚动显示。"),
        "expected_features": [
            ("三选一按钮",         ["石头", "剪刀", "布", "rock", "paper", "scissors", "✊", "✋", "✌"]),
            ("电脑随机出招",       ["Math.random", "random", "floor"]),
            ("胜负判定",           ["win", "lose", "draw", "胜", "负", "平"]),
            ("我方分数",           ["my", "player", "user", "我方", "player"]),
            ("电脑分数",           ["ai", "cpu", "computer", "电脑", "bot"]),
            ("平局统计",           ["draw", "tie", "平局", "平"]),
            ("重置分数",           ["reset", "clear", "重置", "清零"]),
            ("最近5局历史",        ["history", "recent", "last", "5", "slice(-5", "历史", "最近"]),
            ("动态更新 DOM",       ["textContent", "innerHTML", "innerText", "appendChild"]),
        ],
    },
]


# ================================================================
# 3. 代码清洗 & 提取
# ================================================================
def clean_code(raw: str) -> str:
    """去 markdown 包裹 + 截取真正的 HTML 段"""
    code = raw.strip()
    if code.startswith("```"):
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code, count=1)
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    code = code.strip()
    idx = code.upper().find("<!DOCTYPE")
    if idx > 0:
        code = code[idx:]
    end_idx = code.lower().rfind("</html>")
    if end_idx != -1:
        code = code[:end_idx + len("</html>")]
    return code.strip()


def extract_js(code: str) -> str:
    """抽取所有 <script> 块内的 JS 代码"""
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', code, re.DOTALL | re.IGNORECASE)
    return "\n".join(scripts)


def extract_css(code: str) -> str:
    """抽取所有 <style> 块内的 CSS 代码"""
    styles = re.findall(r'<style[^>]*>(.*?)</style>', code, re.DOTALL | re.IGNORECASE)
    return "\n".join(styles)


# ================================================================
# 4. 五维打分
# ================================================================
def score_js_depth(code: str) -> tuple[int, dict]:
    """
    J 维度 (30 分): JS 代码深度
    """
    js = extract_js(code)
    br = {}

    # 事件监听多样性 (8分): addEventListener / onclick / oninput / ...
    events = len(re.findall(
        r'addEventListener\s*\(|\bonclick\s*=|\boninput\s*=|\bonchange\s*=|\bonkeydown\s*=|\bonsubmit\s*=|\bonkeyup\s*=',
        js, re.IGNORECASE))
    br['events'] = min(events * 2, 8)

    # 状态变量 (6分): let/const/var 声明数量
    vars_cnt = len(re.findall(r'\b(?:let|const|var)\s+[A-Za-z_$]', js))
    br['state'] = min(vars_cnt, 6)

    # 控制结构 (6分): if/for/while/switch
    ctrl = len(re.findall(r'\b(?:if|for|while|switch)\s*\(', js))
    br['control'] = min(ctrl, 6)

    # 函数定义 (5分)
    funcs = (len(re.findall(r'\bfunction\s+[A-Za-z_$]\w*\s*\(', js))
             + len(re.findall(r'=>\s*[{(]', js))
             + len(re.findall(r'\b[A-Za-z_$]\w*\s*=\s*function\s*\(', js)))
    br['funcs'] = min(funcs, 5)

    # API / 存储 (5分): localStorage / setInterval / Date / Math.random / fetch
    apis = len(re.findall(
        r'localStorage|sessionStorage|setInterval|setTimeout|Date\s*\(|Math\.random|Math\.floor|Math\.max|Math\.min|JSON\.(?:parse|stringify)',
        js))
    br['apis'] = min(apis, 5)

    total = sum(br.values())
    return min(total, 30), br


def score_instruction_follow(code: str, test_case: dict) -> tuple[int, dict]:
    """
    I 维度 (25 分): 指令遵循度
    每个 expected_feature 关键词命中即得分，满分归一化到 25
    """
    features = test_case.get("expected_features", [])
    if not features:
        return 25, {"note": "no expected features"}

    haystack = code.lower()
    # 同时也加入 JS 内容作为检测范围（某些功能只在 JS 里体现）
    # 注意：haystack 已经是 code.lower()，包含了 script 内容
    hits = []
    missed = []
    for feat_name, keywords in features:
        hit = False
        for kw in keywords:
            try:
                # 支持正则形式的 keyword（如 "font-size:.*60"）
                if re.search(kw.lower(), haystack):
                    hit = True
                    break
            except re.error:
                if kw.lower() in haystack:
                    hit = True
                    break
        if hit:
            hits.append(feat_name)
        else:
            missed.append(feat_name)

    score = round(len(hits) / len(features) * 25)
    return score, {
        "hit_count":  len(hits),
        "total":      len(features),
        "hits":       hits,
        "missed":     missed,
    }


def score_functional(code: str) -> tuple[int, dict]:
    """
    F 维度 (20 分): 功能可运行性（静态 DOM/脚本完整性）
    """
    br = {}

    # (a) DOCTYPE 声明 (3)
    br['doctype'] = 3 if re.search(r'<!doctype\s+html', code, re.IGNORECASE) else 0

    # (b) </html> 闭合 (3)
    br['html_close'] = 3 if "</html>" in code.lower() else 0

    # (c) <script> 块闭合 (3)
    open_scripts  = len(re.findall(r'<script[^>]*>', code, re.IGNORECASE))
    close_scripts = len(re.findall(r'</script\s*>',   code, re.IGNORECASE))
    br['script_balance'] = 3 if open_scripts > 0 and open_scripts == close_scripts else 0

    # (d) <style> 块闭合 (2)
    open_styles  = len(re.findall(r'<style[^>]*>', code, re.IGNORECASE))
    close_styles = len(re.findall(r'</style\s*>',   code, re.IGNORECASE))
    br['style_balance'] = 2 if open_styles > 0 and open_styles == close_styles else 0

    # (e) 大括号平衡 (4): JS 代码的 { 数量 == } 数量
    js = extract_js(code)
    open_braces  = js.count('{')
    close_braces = js.count('}')
    br['brace_balance'] = 4 if open_braces > 0 and abs(open_braces - close_braces) <= 1 else 0

    # (f) 小括号平衡 (3)
    open_paren  = js.count('(')
    close_paren = js.count(')')
    br['paren_balance'] = 3 if open_paren > 0 and abs(open_paren - close_paren) <= 1 else 0

    # (g) 没有明显的未完成标记 (2): 代码末尾没有挂着 `function` / `const x =` 不完整符号
    tail = code[-200:].lower() if len(code) > 200 else code.lower()
    dangling = any(t in tail[-50:] for t in ['function(', 'const ', 'let ', '= function'])
    br['no_dangling_tail'] = 2 if not dangling else 0

    total = sum(br.values())
    return min(total, 20), br


def score_css_quality(code: str) -> tuple[int, dict]:
    """
    C 维度 (15 分): CSS 质量和移动端适配
    """
    css = extract_css(code)
    full = code.lower()
    br = {}

    # viewport meta (3)
    br['viewport'] = 3 if 'viewport' in full else 0

    # 移动端适配 (3): max-width / 100% / 100vw / @media
    mobile_hits = sum([
        1 if 'max-width' in full else 0,
        1 if '100%' in full or '100vw' in full or '100vh' in full else 0,
        1 if '@media' in css else 0,
    ])
    br['mobile'] = min(mobile_hits, 3)

    # 现代 UI (3): 圆角 / 阴影 / 渐变
    modern_hits = sum([
        1 if 'border-radius' in css else 0,
        1 if 'box-shadow' in css else 0,
        1 if 'linear-gradient' in css or 'radial-gradient' in css else 0,
    ])
    br['modern_ui'] = modern_hits

    # 颜色系统 (3): 至少 3 个不同颜色（hex/rgb/命名色）
    colors = set(re.findall(r'#[0-9a-fA-F]{3,8}\b', css))
    colors |= set(re.findall(r'rgb\s*\([^)]+\)', css))
    br['colors'] = min(len(colors), 3)

    # 无外部 CDN (3)
    has_cdn = any(k in full for k in ['cdn.', 'unpkg', 'jsdelivr', 'googleapis', 'cdnjs'])
    br['no_cdn'] = 3 if not has_cdn else 0

    total = sum(br.values())
    return min(total, 15), br


def score_structure(code: str) -> tuple[int, dict]:
    """
    S 维度 (10 分): HTML 结构合规性
    """
    br = {}
    low = code.lower()

    br['html_lang']    = 2 if re.search(r'<html[^>]*\blang=', code, re.IGNORECASE) else 0
    br['meta_charset'] = 2 if 'charset' in low else 0
    br['has_title']    = 2 if '<title' in low else 0
    br['head_body']    = 2 if '<head' in low and '<body' in low else 0
    br['length_ok']    = 2 if 500 <= len(code) <= 20000 else (1 if 300 <= len(code) else 0)

    total = sum(br.values())
    return min(total, 10), br


def score_html_100(code: str, test_case: dict) -> dict:
    """
    综合 5 维打分，返回 100 分制详细结果。
    """
    j_score, j_br = score_js_depth(code)
    i_score, i_br = score_instruction_follow(code, test_case)
    f_score, f_br = score_functional(code)
    c_score, c_br = score_css_quality(code)
    s_score, s_br = score_structure(code)
    total = j_score + i_score + f_score + c_score + s_score

    return {
        "total":        total,
        "J_js_depth":   j_score,
        "I_instruct":   i_score,
        "F_functional": f_score,
        "C_css":        c_score,
        "S_structure":  s_score,
        "breakdown": {
            "J": j_br,
            "I": i_br,
            "F": f_br,
            "C": c_br,
            "S": s_br,
        },
        "length": len(code),
    }


# ================================================================
# 5. 推理封装：Best-of-N + 贪心兜底
# ================================================================
def generate_one(tokenizer, model, instruction: str, gen_kwargs: dict) -> str:
    """单次生成，返回清洗后的 HTML 代码"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **gen_kwargs,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_code(raw)


def generate_best_of_n(tokenizer, model, test_case: dict, label: str = "") -> tuple[str, list[dict]]:
    """
    Best-of-N 采样 + 贪心兜底。
    返回 (最佳 HTML, [每次采样的评分字典])
    """
    candidates = []
    for i in range(BEST_OF_N):
        code  = generate_one(tokenizer, model, test_case["instruction"], SAMPLE_KWARGS)
        score = score_html_100(code, test_case)
        candidates.append({"code": code, "score": score, "mode": f"sample_{i+1}"})
        print(f"      [{label} 采样 {i+1}/{BEST_OF_N}] total={score['total']}/100 "
              f"(J{score['J_js_depth']} I{score['I_instruct']} F{score['F_functional']} "
              f"C{score['C_css']} S{score['S_structure']}) len={score['length']}")
        if score["total"] >= MIN_ACCEPT_SCORE + 20:   # 80 分以上直接停止
            print(f"      → 优秀! 跳过剩余采样")
            break

    # 选最佳
    best = max(candidates, key=lambda x: x["score"]["total"])

    # 贪心兜底：如果最佳 <40，再用贪心解码试一次
    if best["score"]["total"] < 40:
        print(f"      [{label} 贪心兜底] 最佳 {best['score']['total']}/100 < 40, 尝试 greedy ...")
        greedy_code  = generate_one(tokenizer, model, test_case["instruction"], GREEDY_KWARGS)
        greedy_score = score_html_100(greedy_code, test_case)
        candidates.append({"code": greedy_code, "score": greedy_score, "mode": "greedy"})
        print(f"      [{label} 贪心结果] total={greedy_score['total']}/100")
        if greedy_score["total"] > best["score"]["total"]:
            best = candidates[-1]

    return best["code"], candidates


# ================================================================
# 6. 工具函数：模型加载
# ================================================================
def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model_with_adapter(adapter_dir: str):
    """加载基座 + 挂载 LoRA 适配器"""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    print(f"  >>> 加载基础模型 (bf16) ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  >>> 挂载 LoRA 适配器: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model
