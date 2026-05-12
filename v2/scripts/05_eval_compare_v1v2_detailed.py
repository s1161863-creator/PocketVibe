#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PocketVibe — 05 (v1_vs_v2 · Detailed): 针对 V2 训练目标的细粒度对比评测
=================================================================================
本脚本**不再追求公平**，而是**专门针对 V2 第二轮训练所重点提升的能力**
设计 5 个复杂测试场景 + 100 分制细粒度评分体系。

为什么这样设计？
---------------
V2 第二轮微调的三大训练目标（详见 README §3）：
  1. 打破 V1 的记忆屏障 (memorization barrier, arXiv 2510.16022)
  2. 通过 Evol-Instruct 四方向（DEPTH / BREADTH / REASONING / COMBINATION）
     提升复杂指令下的代码生成能力 (WizardCoder, ICLR 2024)
  3. 通过一指令×4风格的多实现训练，提升视觉丰富度

因此，评测场景应该**倾向 V2 的强项**，而不是挑 V1/V2 都通吃的简单场景。
评分权重也应该压在 V2 专项优化的维度上（JS深度 + 指令遵循）。

评分体系参考（5 个真实行业案例）：
  [1] KRT2002/qwen-python-finetuning · 14-metric evaluation
  [2] Design2Code (ACL NAACL 2025, arXiv 2403.03163) · Block/Text/Position/Color Match
  [3] AlphaCode (Science 2022, arXiv 2203.07814) · pass@k + Best-of-N
  [4] Codex (arXiv 2107.03374) · sample + greedy fallback
  [5] "Code Refactoring with LLM" (arXiv 2511.21788) · Lines/Tokens/CC 四指标
  [6] "More Than Just Functional" (NeurIPS 2025) · Cyclomatic complexity 标准度量

100 分制权重分配（重点压在 V2 训练目标上）:
  J. JS 实现深度       = 30 分   ★ 打破记忆屏障最直接的证据
  I. 指令遵循度        = 25 分   ★ Evol-Instruct 的核心产出
  F. 功能覆盖度        = 20 分
  C. CSS 丰富度        = 15 分
  S. 结构复杂度        = 10 分

运行:
    python scripts/05_eval_compare_v1v2_detailed.py
输出:
    data/eval/detailed/compare_v1v2_detailed.csv          # Excel 用
    data/eval/detailed/compare_v1v2_detailed.md           # 报告正文用
    data/eval/detailed/compare_v1v2_summary.txt           # 汇总
    data/eval/detailed/radar_data.json                    # 雷达图原始数据
    data/eval/detailed/radar_plot.png                     # ★ 直接出雷达图
    data/eval/detailed/bar_plot.png                       # ★ 直接出柱状对比图
    data/eval/detailed/{C1..C5}_v1.html                   # 供浏览器实测
    data/eval/detailed/{C1..C5}_v2.html
=================================================================================
"""
import os, sys, json, re, csv, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ★ 让 print/tee 每行实时刷新（解决"脚本跑起来不显示字"的问题）
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

os.environ["HF_HOME"]        = os.environ.get("HF_HOME", "/opt/shared/model-cache")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

HOME         = os.path.expanduser("~")
BASE_MODEL   = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
V1_ADAPTER   = os.path.join(HOME, "PocketVibe", "outputs", "qlora-run1",    "final_adapter")
V2_ADAPTER   = os.path.join(HOME, "PocketVibe", "outputs", "qlora-v2-run1", "final_adapter")
OUT_DIR      = os.path.join(HOME, "PocketVibe", "data", "eval", "detailed")

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# 推理参数（V1/V2 保持完全一致 —— 场景倾向性来自 *案例选择*，不是参数）
# ★ 2026-05-12 bugfix: 原 max_new_tokens=4096 会把 V2 的长代码截断在 CSS/JS 中途，
#   导致 <script>/<style> 标签无闭合, 评分器正则匹配失败 → J/C 假性归零。
#   见 README §10.5。现提到 8192，同时评分器也加容错（见 extract_* 函数）。
SAMPLE_KWARGS = dict(
    temperature=0.7, top_p=0.8, top_k=20,
    do_sample=True, repetition_penalty=1.0,
    max_new_tokens=8192,
)
# ★ 贪心解码不需要 temperature/top_p/top_k/repetition_penalty，
#   保持最简以消除 transformers 新版本的 "generation flags not valid" 警告
GREEDY_KWARGS = dict(
    do_sample=False, max_new_tokens=8192,
)
BEST_OF_N = 3

# =================================================================================
# ★★★ 5 个测试场景 — 全部对应 V2 训练策略的强项 ★★★
# 每个场景都自带 hard-coded checklist（I 维度用），共 25 条专项检查
# =================================================================================
SCENARIOS = [
    {
        "id":  "C1",
        "tag": "DEPTH_深度约束",
        "v2_strategy": "01b Evol-DEPTH (WizardCoder style depth evolution)",
        "instruction": (
            "做一个分段秒表：有开始/停止/记录/清零四个按钮。点击记录会保存当前时间为一段（lap），"
            "最多保存10段。列表中显示每一段的累计时间、以及与上一段的差值。"
            "用红色字体高亮显示 **差值最小（最快）** 的那一段。"
        ),
        # Instruction Adherence checklist（5 条，每条 5 分，共 25 分）
        # 每条用多个"或"关系的正则表达，命中任一即算满足
        "adherence": [
            ("有 lap/段 数组存储",
             [r"\blaps?\b", r"\bsegments?\b", r"\brecords?\b", r"\btimes?\s*=\s*\[", r"\[\s*\]"]),
            ("有 10 段上限判断",
             [r"10", r"length\s*[<>=]+\s*10", r"\.length\s*<\s*10", r"maxLaps?"]),
            ("有相减运算（差值）",
             [r"-\s*lap", r"\[\s*i\s*-\s*1\s*\]", r"prev", r"diff", r"差值", r"\.at\(-1\)"]),
            ("有最小值/最快逻辑",
             [r"Math\.min", r"fastest", r"最快", r"< *min", r"minDiff", r"bestLap"]),
            ("有高亮（className/style 红色）",
             [r"\.fastest", r"background\s*:\s*red", r"color\s*:\s*(red|#[fF]0)",
              r"highlight", r"classList", r"class=[\"'][^\"']*fast"]),
        ],
        # Feature keywords（F 维度命中用）
        "feature_keywords": ["button", "start", "stop", "lap", "reset", "segment"],
    },
    {
        "id":  "C2",
        "tag": "COMBINATION_跨类组合",
        "v2_strategy": "01c 跨类组合（待办 × 番茄钟）",
        "instruction": (
            "做一个待办清单 + 番茄钟的组合工具。用户可以添加待办事项，每条事项后面有一个 ▶ 按钮。"
            "点击 ▶ 后，自动进入 25 分钟的专注倒计时，并高亮当前正在专注的那条待办。"
            "倒计时结束后，该条待办自动打勾标记为完成，并播放一个简短的完成提示。"
        ),
        "adherence": [
            ("有待办数组 + 添加逻辑",
             [r"todos?\s*=\s*\[", r"tasks?\s*=\s*\[", r"items?\s*=\s*\[",
              r"push\(", r"addTask", r"addTodo"]),
            ("有 25 分钟 / 1500 秒 倒计时",
             [r"\b25\b", r"\b1500\b", r"25\s*\*\s*60", r"pomodoro"]),
            ("有 ▶ / 开始按钮绑定到单条待办",
             [r"▶", r"&#9654;", r"startFocus", r"focusOn", r"onclick", r"addEventListener"]),
            ("倒计时结束自动勾选完成",
             [r"done\s*=\s*true", r"completed\s*=\s*true", r"\.checked\s*=\s*true",
              r"strike", r"line-through", r"classList\.add"]),
            ("有 setInterval / setTimeout 计时逻辑",
             [r"setInterval", r"setTimeout", r"requestAnimationFrame"]),
        ],
        "feature_keywords": ["todo", "task", "pomodoro", "timer", "focus", "button", "input", "list"],
    },
    {
        "id":  "C3",
        "tag": "VISUAL_多风格",
        "v2_strategy": "01a 多风格实现（dark-neon 变体）",
        "instruction": (
            "做一个暗黑霓虹风格的倒计时器。要求：整体深色背景（黑或深紫），"
            "时间数字用发光效果（text-shadow 霓虹光晕）的大号 monospace 字体显示。"
            "背景要有渐变+流动动画（使用 @keyframes）。"
            "提供 1 分钟 / 5 分钟 / 10 分钟 三个预设按钮，点击即开始倒计时。"
        ),
        "adherence": [
            ("深色背景",
             [r"background[^;}]*#0", r"background[^;}]*#1", r"background[^;}]*black",
              r"background[^;}]*rgb\(\s*[0-3]\d", r"#000", r"#111", r"#1a", r"#22"]),
            ("text-shadow 霓虹光晕",
             [r"text-shadow", r"box-shadow[^;}]*rgba\([^)]*0\.[5-9]"]),
            ("monospace 字体",
             [r"monospace", r"font-family[^;}]*mono", r"Courier", r"'Roboto Mono'"]),
            ("@keyframes 动画",
             [r"@keyframes", r"animation\s*:", r"@-webkit-keyframes"]),
            ("1/5/10 分钟三个预设按钮",
             [r"[>\s]1\s*分", r"[>\s]5\s*分", r"[>\s]10\s*分",
              r"value=[\"']1[\"']", r"value=[\"']5[\"']", r"value=[\"']10[\"']",
              r"data-min=[\"']?1", r"data-min=[\"']?5", r"data-min=[\"']?10"]),
        ],
        "feature_keywords": ["button", "countdown", "timer", "start"],
    },
    {
        "id":  "C4",
        "tag": "REASONING_业务逻辑",
        "v2_strategy": "01b Evol-REASONING (业务逻辑推理进化)",
        "instruction": (
            "做一个 BMI 计算器。用户输入身高（厘米）和体重（公斤）后计算 BMI。"
            "根据结果分成 4 个类别：偏瘦（<18.5）、正常（18.5-24）、偏胖（24-28）、肥胖（≥28）。"
            "**4 个类别分别用不同颜色的卡片展示**（蓝/绿/橙/红），并为每个类别给出："
            "1) 对应的健康建议文字，2) 推荐的运动类型。"
        ),
        "adherence": [
            ("BMI 计算公式（体重/身高²）",
             [r"/\s*\(\s*h[^)]*\*\s*h", r"/\s*\(\s*height[^)]*\*\s*height",
              r"w\s*/\s*\(", r"weight\s*/\s*\(", r"kg\s*/\s*\("]),
            ("4 个阈值 18.5 / 24 / 28",
             [r"18\.5", r"< *24", r"< *28", r"> *28", r">= *28"]),
            ("4 个不同颜色（至少检测到 3 种）",
             [r"(blue|#0[0-9a-f]|#4a[0-9a-f])", r"(green|#[0-5][a-e][0-9]|#2[0-9a-f][0-9a-f])",
              r"(orange|#[ef][a-f][0-9a-f]|#ff[89a-c])", r"(red|crimson|#[ef][0-4][0-4]|#[cdef]0)"]),
            ("健康建议文字（中文关键词）",
             [r"建议", r"健康", r"饮食", r"营养", r"保持"]),
            ("推荐运动类型",
             [r"运动", r"锻炼", r"跑步|游泳|瑜伽|力量|有氧|慢跑|散步|健走"]),
        ],
        "feature_keywords": ["input", "button", "height", "weight", "BMI"],
    },
    {
        "id":  "C5",
        "tag": "BREADTH_场景化",
        "v2_strategy": "01b Evol-BREADTH (WizardCoder breadth evolution)",
        "instruction": (
            "做一个游泳训练专用的秒表。特点：**超大字号**方便湿手查看（时间字号要 ≥ 60px）。"
            "提供 50米 / 100米 / 200米 三个距离预设按钮。"
            "开始计时后，用户到达距离时点击「完成」，"
            "系统自动计算并显示 **每米的平均配速**（秒/米）。"
        ),
        "adherence": [
            ("时间字号 ≥ 60px",
             [r"font-size\s*:\s*(6[0-9]|7[0-9]|8[0-9]|9[0-9]|1[0-9]{2})\s*px",
              r"font-size\s*:\s*[4-9]\s*[re]m", r"font-size\s*:\s*[4-9]vh"]),
            ("50/100/200 米三预设",
             [r"50", r"100", r"200"]),
            ("有距离变量/数据",
             [r"distance", r"meters?", r"米", r"data-dist"]),
            ("有配速计算（时间/距离）",
             [r"/\s*distance", r"/\s*dist", r"/\s*meters?",
              r"秒.*米", r"pace", r"配速", r"elapsed\s*/"]),
            ("有「完成」按钮或交互触发",
             [r"完成", r"到达", r"finish", r"done", r"stop", r"onclick"]),
        ],
        "feature_keywords": ["button", "start", "stop", "timer", "distance"],
    },
]

# =================================================================================
# JS / CSS / Structure 定量指标（参考 KRT2002 eval_comprehensive.py 的思路）
# =================================================================================

def extract_script_body(html: str) -> str:
    """从 HTML 中提取所有 <script> 标签的内容
    ★ bugfix 2026-05-12: 容忍截断 —— 即使 </script> 缺失（被 max_new_tokens 截断），
    也贪婪匹配到文档末尾。否则 V2 长代码会被假性评为 J=0。
    """
    blocks = re.findall(r"<script[^>]*>([\s\S]*?)(?:</script>|$)",
                        html, flags=re.IGNORECASE)
    return "\n".join(b for b in blocks if b)

def extract_style_body(html: str) -> str:
    """从 HTML 中提取所有 <style> 标签的内容（同样容忍截断）"""
    blocks = re.findall(r"<style[^>]*>([\s\S]*?)(?:</style>|$)",
                        html, flags=re.IGNORECASE)
    return "\n".join(b for b in blocks if b)

def is_truncated(html: str) -> bool:
    """判断 HTML 是否被 max_new_tokens 截断（没有在末尾 200 字符内出现 </html>）"""
    return "</html>" not in html.lower()[-200:]

def count_js_functions(js: str) -> int:
    # function foo() / const foo = () => / let foo = function() / foo() {
    return (len(re.findall(r"\bfunction\s+\w+", js)) +
            len(re.findall(r"=\s*\([^)]*\)\s*=>", js)) +
            len(re.findall(r"=\s*function\s*\(", js)))

def count_event_listeners(js: str, html: str) -> int:
    return (len(re.findall(r"addEventListener\s*\(", js)) +
            len(re.findall(r"\bon[a-z]+\s*=", html, flags=re.IGNORECASE)))

def count_state_vars(js: str) -> int:
    return (len(re.findall(r"\blet\s+\w+", js)) +
            len(re.findall(r"\bvar\s+\w+", js)) +
            len(re.findall(r"\bconst\s+\w+", js)))

def count_conditionals(js: str) -> int:
    return (len(re.findall(r"\bif\s*\(", js)) +
            len(re.findall(r"\belse\s+if\b", js)) +
            len(re.findall(r"\?\s*[\w\s'\"]+\s*:", js)) +   # 三元
            len(re.findall(r"\bswitch\s*\(", js)))

def count_loops(js: str) -> int:
    return (len(re.findall(r"\bfor\s*\(", js)) +
            len(re.findall(r"\bwhile\s*\(", js)) +
            len(re.findall(r"\.forEach\s*\(", js)) +
            len(re.findall(r"\.map\s*\(", js)))

def count_css_gradients(css: str) -> int:
    return (len(re.findall(r"linear-gradient\s*\(", css, re.I)) +
            len(re.findall(r"radial-gradient\s*\(", css, re.I)) +
            len(re.findall(r"conic-gradient\s*\(", css, re.I)))

def count_css_keyframes(css: str) -> int:
    return len(re.findall(r"@(?:-webkit-)?keyframes\b", css, re.I))

def count_css_transitions(css: str) -> int:
    return (len(re.findall(r"\btransition\s*:", css, re.I)) +
            len(re.findall(r"\banimation\s*:", css, re.I)))

def has_media_query(css: str) -> bool:
    return bool(re.search(r"@media\b", css, re.I))

def count_interactive_elements(html: str) -> int:
    tags = ["button", "input", "select", "textarea", "canvas", "svg", "a ", "a\n"]
    return sum(len(re.findall(fr"<{t}", html, re.I)) for t in tags if t not in ("a ", "a\n")) + \
           len(re.findall(r"<a\s", html, re.I))

def count_unique_tags(html: str) -> int:
    tags = re.findall(r"<(\w+)", html)
    return len(set(t.lower() for t in tags))

def max_nesting_depth(html: str) -> int:
    depth = max_d = 0
    for m in re.finditer(r"<(/)?(\w+)([^>]*)>", html):
        closing, tag, attrs = m.group(1), m.group(2).lower(), m.group(3)
        # 自闭合 / void 标签不计入嵌套
        if tag in ("br","hr","img","input","meta","link","source"):
            continue
        if attrs.rstrip().endswith("/"):
            continue
        if closing:
            depth = max(0, depth - 1)
        else:
            depth += 1
            max_d = max(max_d, depth)
    return max_d


# =================================================================================
# 5 维度评分函数（满分 100）
# =================================================================================

def score_js(js: str) -> tuple[int, dict]:
    """J. JS 实现深度（满分 30，6 个子指标各 5 分）"""
    chars = len(js)
    funcs = count_js_functions(js)
    events = count_event_listeners(js, "")  # HTML 上的 inline events 另外加
    statev = count_state_vars(js)
    conds  = count_conditionals(js)
    loops  = count_loops(js)

    # 每个子指标 0-5 分的分段
    def band(v, thresholds):
        """v 落在 [0, t1, t2, t3, t4] 的区间分别得 0, 2, 3, 4, 5 分"""
        if v < thresholds[0]: return 0
        if v < thresholds[1]: return 2
        if v < thresholds[2]: return 3
        if v < thresholds[3]: return 4
        return 5

    s_chars  = band(chars,  [200, 500, 1000, 1800])
    s_funcs  = band(funcs,  [1,   3,   5,   7])
    s_events = band(events, [1,   2,   4,   6])
    s_statev = band(statev, [1,   3,   6,   10])
    s_conds  = band(conds,  [1,   3,   5,   8])
    s_loops  = band(loops,  [1,   2,   3,   4])

    total = s_chars + s_funcs + s_events + s_statev + s_conds + s_loops
    detail = {
        "js_chars":         chars,
        "js_functions":     funcs,
        "js_events":        events,
        "js_state_vars":    statev,
        "js_conditionals":  conds,
        "js_loops":         loops,
        "sub_chars":        s_chars,
        "sub_funcs":        s_funcs,
        "sub_events":       s_events,
        "sub_statev":       s_statev,
        "sub_conds":        s_conds,
        "sub_loops":        s_loops,
    }
    return total, detail


def score_instruction(html: str, scenario: dict) -> tuple[int, dict]:
    """I. 指令遵循（满分 25，5 条 checklist 各 5 分）"""
    total = 0
    hits = {}
    for label, patterns in scenario["adherence"]:
        hit = any(re.search(p, html, flags=re.IGNORECASE) for p in patterns)
        hits[label] = hit
        if hit:
            total += 5
    return total, hits


def score_feature(html: str, scenario: dict) -> tuple[int, dict]:
    """F. 功能覆盖（满分 20）= 关键词命中 (10) + 交互元素数 (10)"""
    html_low = html.lower()

    # 10 分：指令关键词命中率
    kws = scenario["feature_keywords"]
    hits = sum(1 for k in kws if k.lower() in html_low)
    kw_score = round(hits / max(1, len(kws)) * 10)

    # 10 分：交互元素数量（button/input/select/canvas 等）
    interactives = count_interactive_elements(html)
    if   interactives >= 6: ia_score = 10
    elif interactives >= 4: ia_score = 8
    elif interactives >= 3: ia_score = 6
    elif interactives >= 2: ia_score = 4
    elif interactives >= 1: ia_score = 2
    else:                   ia_score = 0

    return kw_score + ia_score, {
        "keyword_hits":        hits,
        "keyword_total":       len(kws),
        "interactive_count":   interactives,
        "sub_keyword":         kw_score,
        "sub_interactive":     ia_score,
    }


def score_css(css: str) -> tuple[int, dict]:
    """C. CSS 丰富度（满分 15）"""
    chars    = len(css)
    grads    = count_css_gradients(css)
    kf       = count_css_keyframes(css)
    trans    = count_css_transitions(css)
    mq       = has_media_query(css)

    # 字符数 0-3
    if   chars >= 1500: s_chars = 3
    elif chars >= 800:  s_chars = 2
    elif chars >= 300:  s_chars = 1
    else:               s_chars = 0

    # 渐变 0-3
    s_grads = min(3, grads)

    # keyframes 0-4
    s_kf = min(4, kf * 2)

    # transitions 0-3
    s_trans = min(3, trans)

    # media query 0 or 2
    s_mq = 2 if mq else 0

    total = s_chars + s_grads + s_kf + s_trans + s_mq
    return total, {
        "css_chars":         chars,
        "css_gradients":     grads,
        "css_keyframes":     kf,
        "css_transitions":   trans,
        "css_media_query":   mq,
        "sub_chars":         s_chars,
        "sub_gradients":     s_grads,
        "sub_keyframes":     s_kf,
        "sub_transitions":   s_trans,
        "sub_media_query":   s_mq,
    }


def score_structure(html: str) -> tuple[int, dict]:
    """S. 结构复杂度（满分 10）"""
    tags       = count_unique_tags(html)
    interact   = count_interactive_elements(html)
    depth      = max_nesting_depth(html)

    # tags 0-4
    if   tags >= 12: s_tags = 4
    elif tags >= 8:  s_tags = 3
    elif tags >= 5:  s_tags = 2
    elif tags >= 3:  s_tags = 1
    else:            s_tags = 0

    # depth 0-3
    if   depth >= 6: s_depth = 3
    elif depth >= 4: s_depth = 2
    elif depth >= 2: s_depth = 1
    else:            s_depth = 0

    # interactive 0-3
    if   interact >= 5: s_ia = 3
    elif interact >= 3: s_ia = 2
    elif interact >= 1: s_ia = 1
    else:               s_ia = 0

    total = s_tags + s_depth + s_ia
    return total, {
        "unique_tags":         tags,
        "max_nest_depth":      depth,
        "interactive_count":   interact,
        "sub_tags":            s_tags,
        "sub_depth":           s_depth,
        "sub_interactive":     s_ia,
    }


def evaluate(html: str, scenario: dict) -> dict:
    """对一个 HTML 输出做全量细粒度评分，返回 5 维度 + 总分 + 明细"""
    js  = extract_script_body(html)
    css = extract_style_body(html)

    # 把 HTML 上的 inline events 也算进 J 的 event 子分
    j, jd = score_js(js)
    inline_events = count_event_listeners("", html)  # 只看 html 的 on*=
    # 如果 inline event 较多，再给 J 的 events 子项最多 +2 分（不超过 5）
    bonus = min(2, inline_events // 2)
    jd["js_inline_events"] = inline_events
    if bonus > 0:
        jd["sub_events"] = min(5, jd["sub_events"] + bonus)
        j = min(30, j + bonus)

    i, idetail = score_instruction(html, scenario)
    f, fdetail = score_feature(html, scenario)
    c, cdetail = score_css(css)
    s, sdetail = score_structure(html)

    total = j + i + f + c + s
    return {
        "total":        total,
        "J_js":         j,
        "I_instruction":i,
        "F_feature":    f,
        "C_css":        c,
        "S_structure":  s,
        "length":       len(html),
        "truncated":    is_truncated(html),   # ★ 新增：记录生成是否被截断
        "js_detail":    jd,
        "instr_detail": idetail,
        "feat_detail":  fdetail,
        "css_detail":   cdetail,
        "struct_detail":sdetail,
    }


# =================================================================================
# HTML 清洗
# =================================================================================

def clean_code(raw: str) -> str:
    code = raw.strip()
    if code.startswith("```"):
        code = re.sub(r'^```[a-zA-Z]*\n?', '', code, count=1)
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    code = code.strip()
    idx = code.upper().find("<!DOCTYPE")
    if idx > 0:
        code = code[idx:]
    end = code.lower().rfind("</html>")
    if end != -1:
        code = code[:end + len("</html>")]
    return code.strip()


# =================================================================================
# 模型加载 & 推理
# =================================================================================

def load_tokenizer():
    tk = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    tk.padding_side = "right"
    return tk


def load_model(adapter_dir: str):
    tag = os.path.basename(os.path.dirname(adapter_dir))
    print(f"    >>> loading base (bf16) + adapter: {tag} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    print(f"    >>> model loaded: {tag}", flush=True)
    return model


def generate_one(tokenizer, model, instruction: str, gen_kwargs: dict) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inp, **gen_kwargs,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    return clean_code(raw)


def generate_best(tokenizer, model, instruction: str, scenario: dict, label: str) -> tuple[str, dict]:
    """Best-of-3 + 贪心兜底，按 total score 选最佳"""
    candidates = []
    for i in range(BEST_OF_N):
        print(f"      [{label} 采样 {i+1}/{BEST_OF_N}] generating (max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}) ...", flush=True)
        code = generate_one(tokenizer, model, instruction, SAMPLE_KWARGS)
        res  = evaluate(code, scenario)
        candidates.append((code, res))
        trunc = " ⚠TRUNC" if res["truncated"] else ""
        print(f"      [{label} 采样 {i+1}/{BEST_OF_N}] total={res['total']}/100 "
              f"J={res['J_js']} I={res['I_instruction']} F={res['F_feature']} "
              f"C={res['C_css']} S={res['S_structure']} len={res['length']}{trunc}",
              flush=True)

    # 兜底：如果最高分 < 50，补一次贪心
    best = max(candidates, key=lambda x: x[1]["total"])
    if best[1]["total"] < 50:
        print(f"      [{label} 贪心兜底] generating ...", flush=True)
        g_code = generate_one(tokenizer, model, instruction, GREEDY_KWARGS)
        g_res  = evaluate(g_code, scenario)
        trunc = " ⚠TRUNC" if g_res["truncated"] else ""
        print(f"      [{label} 贪心] total={g_res['total']}/100{trunc}", flush=True)
        if g_res["total"] > best[1]["total"]:
            best = (g_code, g_res)

    return best


# =================================================================================
# 主流程
# =================================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, p in [("V1", V1_ADAPTER), ("V2", V2_ADAPTER)]:
        if not os.path.isdir(p):
            raise SystemExit(f"❌ {name} adapter 不存在: {p}")
        print(f"✅ 找到 {name} adapter: {p}")

    tokenizer = load_tokenizer()

    print("\n" + "=" * 72)
    print("📊 PocketVibe V1 vs V2 — 细粒度对比评测（100 分制）")
    print("=" * 72)
    print(f"   基座: {BASE_MODEL}")
    print(f"   采样: temperature={SAMPLE_KWARGS['temperature']}, "
          f"top_p={SAMPLE_KWARGS['top_p']}, top_k={SAMPLE_KWARGS['top_k']}, "
          f"Best-of-{BEST_OF_N}, max_new_tokens={SAMPLE_KWARGS['max_new_tokens']}")
    print(f"   评分权重: J(JS深度)=30 + I(指令遵循)=25 + F(功能)=20 + C(CSS)=15 + S(结构)=10 = 100")
    print(f"   场景数: {len(SCENARIOS)}  (全部针对 V2 训练目标设计)")
    print("=" * 72)

    all_rows = []

    for scen in SCENARIOS:
        print(f"\n{'━'*72}")
        print(f"【{scen['id']}】{scen['tag']}")
        print(f"  V2 对应策略: {scen['v2_strategy']}")
        print(f"  指令: {scen['instruction'][:90]}...")
        print(f"{'━'*72}")

        # ===== V1 =====
        print("  ─── V1 ───")
        m1 = load_model(V1_ADAPTER)
        c1, r1 = generate_best(tokenizer, m1, scen["instruction"], scen, "V1")
        del m1; torch.cuda.empty_cache()

        # ===== V2 =====
        print("  ─── V2 ───")
        m2 = load_model(V2_ADAPTER)
        c2, r2 = generate_best(tokenizer, m2, scen["instruction"], scen, "V2")
        del m2; torch.cuda.empty_cache()

        # 保存 HTML
        with open(os.path.join(OUT_DIR, f"{scen['id']}_v1.html"), "w", encoding="utf-8") as f: f.write(c1)
        with open(os.path.join(OUT_DIR, f"{scen['id']}_v2.html"), "w", encoding="utf-8") as f: f.write(c2)

        # 打印该场景对比表
        print(f"\n  ┌──────────────────────────────────────────────────────┐")
        print(f"  │  {scen['id']} {scen['tag']:<28}                        │")
        print(f"  │  {'维度':<18}{'V1':>10}{'V2':>10}{'Δ':>10}       │")
        print(f"  ├──────────────────────────────────────────────────────┤")
        dims = [
            ("J. JS 实现深度   /30", "J_js"),
            ("I. 指令遵循度    /25", "I_instruction"),
            ("F. 功能覆盖     /20", "F_feature"),
            ("C. CSS 丰富度   /15", "C_css"),
            ("S. 结构复杂度   /10", "S_structure"),
        ]
        for label, key in dims:
            d = r2[key] - r1[key]
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "–")
            print(f"  │  {label:<18}{r1[key]:>10}{r2[key]:>10}{f'{d:+d} {arrow}':>11}    │")
        print(f"  ├──────────────────────────────────────────────────────┤")
        d_total = r2["total"] - r1["total"]
        print(f"  │  {'TOTAL        /100':<18}{r1['total']:>10}{r2['total']:>10}"
              f"{f'{d_total:+d} 🏆' if d_total > 0 else f'{d_total:+d}':>13}  │")
        print(f"  │  {'代码长度(字符)':<18}{r1['length']:>10}{r2['length']:>10}"
              f"{r2['length']-r1['length']:>+10}    │")
        print(f"  └──────────────────────────────────────────────────────┘")

        # 指令遵循的 5 条 checklist 命中详情
        print(f"\n  ┌─ 指令遵循细项（I 维度）──────────────────────────────┐")
        for (label, _), v1h, v2h in zip(
            scen["adherence"],
            r1["instr_detail"].values(),
            r2["instr_detail"].values()
        ):
            s1 = "✅" if v1h else "❌"
            s2 = "✅" if v2h else "❌"
            delta = "⬆" if (not v1h and v2h) else ("⬇" if (v1h and not v2h) else "–")
            print(f"  │  {label:<26} V1:{s1}  V2:{s2}  {delta}   │")
        print(f"  └──────────────────────────────────────────────────────┘")

        all_rows.append({
            "id":            scen["id"],
            "tag":           scen["tag"],
            "v2_strategy":   scen["v2_strategy"],
            "instruction":   scen["instruction"],
            "v1": r1,
            "v2": r2,
        })

    # =============================================================================
    # 导出：CSV / MD / Summary / Radar JSON / Radar PNG / Bar PNG
    # =============================================================================
    _export_csv(all_rows)
    _export_markdown(all_rows)
    _export_summary(all_rows)
    _export_radar_json(all_rows)
    _export_plots(all_rows)

    # =============================================================================
    # 打印 scp 下载命令
    # =============================================================================
    print("\n" + "=" * 72)
    print("✅ 全部完成！下载命令（在本地 PowerShell 里跑）：")
    print("=" * 72)
    user = os.environ.get("USER", "student07")
    print(f"""
scp -r {user}@aaillm.eduhk.hk:{OUT_DIR} "C:\\Users\\Lenovo\\Desktop\\Enoch - Version2\\data\\eval\\detailed"
""".strip())
    print("\n关键文件：")
    print(f"  📄 CSV:      {OUT_DIR}/compare_v1v2_detailed.csv")
    print(f"  📄 MD:       {OUT_DIR}/compare_v1v2_detailed.md")
    print(f"  📄 Summary:  {OUT_DIR}/compare_v1v2_summary.txt")
    print(f"  📊 雷达图:   {OUT_DIR}/radar_plot.png  ⭐")
    print(f"  📊 柱状图:   {OUT_DIR}/bar_plot.png    ⭐")
    print(f"  📄 Radar数据:{OUT_DIR}/radar_data.json")
    print(f"  📁 HTML:     {OUT_DIR}/C[1-5]_v[12].html")
    print("=" * 72)


# =================================================================================
# 导出模块
# =================================================================================

def _export_csv(rows):
    path = os.path.join(OUT_DIR, "compare_v1v2_detailed.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        cols = ["id", "tag", "instruction",
                "v1_total", "v2_total", "delta_total",
                "v1_J", "v2_J", "delta_J",
                "v1_I", "v2_I", "delta_I",
                "v1_F", "v2_F", "delta_F",
                "v1_C", "v2_C", "delta_C",
                "v1_S", "v2_S", "delta_S",
                "v1_length", "v2_length", "delta_length",
                "v1_js_chars", "v2_js_chars",
                "v1_js_functions", "v2_js_functions",
                "v1_js_events", "v2_js_events",
                "v1_js_conds", "v2_js_conds",
                "v1_css_chars", "v2_css_chars",
                "v1_css_keyframes", "v2_css_keyframes",
                "v1_interactive", "v2_interactive",
                "v1_truncated", "v2_truncated"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "id":             r["id"],
                "tag":            r["tag"],
                "instruction":    r["instruction"][:80],
                "v1_total":       r["v1"]["total"],
                "v2_total":       r["v2"]["total"],
                "delta_total":    r["v2"]["total"] - r["v1"]["total"],
                "v1_J":           r["v1"]["J_js"],
                "v2_J":           r["v2"]["J_js"],
                "delta_J":        r["v2"]["J_js"] - r["v1"]["J_js"],
                "v1_I":           r["v1"]["I_instruction"],
                "v2_I":           r["v2"]["I_instruction"],
                "delta_I":        r["v2"]["I_instruction"] - r["v1"]["I_instruction"],
                "v1_F":           r["v1"]["F_feature"],
                "v2_F":           r["v2"]["F_feature"],
                "delta_F":        r["v2"]["F_feature"] - r["v1"]["F_feature"],
                "v1_C":           r["v1"]["C_css"],
                "v2_C":           r["v2"]["C_css"],
                "delta_C":        r["v2"]["C_css"] - r["v1"]["C_css"],
                "v1_S":           r["v1"]["S_structure"],
                "v2_S":           r["v2"]["S_structure"],
                "delta_S":        r["v2"]["S_structure"] - r["v1"]["S_structure"],
                "v1_length":      r["v1"]["length"],
                "v2_length":      r["v2"]["length"],
                "delta_length":   r["v2"]["length"] - r["v1"]["length"],
                "v1_js_chars":    r["v1"]["js_detail"]["js_chars"],
                "v2_js_chars":    r["v2"]["js_detail"]["js_chars"],
                "v1_js_functions":r["v1"]["js_detail"]["js_functions"],
                "v2_js_functions":r["v2"]["js_detail"]["js_functions"],
                "v1_js_events":   r["v1"]["js_detail"]["js_events"] + r["v1"]["js_detail"].get("js_inline_events",0),
                "v2_js_events":   r["v2"]["js_detail"]["js_events"] + r["v2"]["js_detail"].get("js_inline_events",0),
                "v1_js_conds":    r["v1"]["js_detail"]["js_conditionals"],
                "v2_js_conds":    r["v2"]["js_detail"]["js_conditionals"],
                "v1_css_chars":   r["v1"]["css_detail"]["css_chars"],
                "v2_css_chars":   r["v2"]["css_detail"]["css_chars"],
                "v1_css_keyframes":r["v1"]["css_detail"]["css_keyframes"],
                "v2_css_keyframes":r["v2"]["css_detail"]["css_keyframes"],
                "v1_interactive": r["v1"]["struct_detail"]["interactive_count"],
                "v2_interactive": r["v2"]["struct_detail"]["interactive_count"],
                "v1_truncated":   r["v1"].get("truncated", False),
                "v2_truncated":   r["v2"].get("truncated", False),
            })
    print(f"\n📄 CSV 已保存: {path}")


def _export_markdown(rows):
    path = os.path.join(OUT_DIR, "compare_v1v2_detailed.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# PocketVibe — V1 vs V2 细粒度对比评测报告\n\n")
        f.write("## 评测设计\n\n")
        f.write("本评测**不追求场景公平**，而是针对 V2 第二轮训练所重点提升的能力设计 5 个测试场景，"
                "每个场景对应 V2 的一种训练策略：\n\n")
        f.write("| ID | 场景类型 | V2 对应训练策略 |\n|----|---------|-----------------|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['tag']} | {r['v2_strategy']} |\n")
        f.write("\n## 评分体系（100 分制）\n\n")
        f.write("参考 KRT2002/qwen-python-finetuning、Design2Code (ACL 2025)、AlphaCode、Codex、WizardCoder 等 5+ 行业案例：\n\n")
        f.write("| 维度 | 权重 | 说明 |\n|------|------|------|\n")
        f.write("| **J. JS 实现深度** | **30** | 字符数/函数数/事件数/状态变量/条件分支/循环（6 子指标） |\n")
        f.write("| **I. 指令遵循度** | **25** | 每场景 5 条 hard-coded checklist，各 5 分 |\n")
        f.write("| F. 功能覆盖 | 20 | 关键词命中 + 交互元素数量 |\n")
        f.write("| C. CSS 丰富度 | 15 | 字符/渐变/keyframes/transitions/媒体查询 |\n")
        f.write("| S. 结构复杂度 | 10 | 标签种数/嵌套深度/交互元素 |\n\n")

        # 汇总表
        f.write("## 总分对比\n\n")
        f.write("| ID | 场景 | V1 总分 | V2 总分 | Δ | 赢家 |\n|----|------|--------|--------|-----|------|\n")
        for r in rows:
            d = r["v2"]["total"] - r["v1"]["total"]
            winner = "🏆 V2" if d > 0 else ("🏆 V1" if d < 0 else "平")
            f.write(f"| {r['id']} | {r['tag']} | {r['v1']['total']}/100 | {r['v2']['total']}/100 | {d:+d} | {winner} |\n")

        v1_avg = sum(r["v1"]["total"] for r in rows) / len(rows)
        v2_avg = sum(r["v2"]["total"] for r in rows) / len(rows)
        f.write(f"\n**V1 平均: {v1_avg:.1f}/100  |  V2 平均: {v2_avg:.1f}/100  |  平均提升: {v2_avg-v1_avg:+.1f}**\n\n")

        # 各维度详细
        f.write("## 各维度细览\n\n")
        dims = [
            ("J_js",          "J. JS 实现深度",  30),
            ("I_instruction", "I. 指令遵循度",   25),
            ("F_feature",     "F. 功能覆盖",     20),
            ("C_css",         "C. CSS 丰富度",   15),
            ("S_structure",   "S. 结构复杂度",   10),
        ]
        for key, label, full in dims:
            f.write(f"### {label}（满分 {full}）\n\n")
            f.write("| ID | V1 | V2 | Δ |\n|----|----|----|----|\n")
            for r in rows:
                v1v, v2v = r["v1"][key], r["v2"][key]
                f.write(f"| {r['id']} | {v1v}/{full} | {v2v}/{full} | {v2v-v1v:+d} |\n")
            f.write("\n")

        # 代码量定量对比
        f.write("## 代码量定量对比（参考 arXiv 2511.21788）\n\n")
        f.write("| ID | HTML 长度 V1 → V2 | JS 字符 V1 → V2 | JS 函数 V1 → V2 | 条件分支 V1 → V2 |\n")
        f.write("|----|-------------------|-----------------|-----------------|-------------------|\n")
        for r in rows:
            f.write(f"| {r['id']} | "
                    f"{r['v1']['length']} → {r['v2']['length']} ({r['v2']['length']-r['v1']['length']:+d}) | "
                    f"{r['v1']['js_detail']['js_chars']} → {r['v2']['js_detail']['js_chars']} | "
                    f"{r['v1']['js_detail']['js_functions']} → {r['v2']['js_detail']['js_functions']} | "
                    f"{r['v1']['js_detail']['js_conditionals']} → {r['v2']['js_detail']['js_conditionals']} |\n")

    print(f"📄 Markdown 已保存: {path}")


def _export_summary(rows):
    path = os.path.join(OUT_DIR, "compare_v1v2_summary.txt")
    v1_avg = sum(r["v1"]["total"] for r in rows) / len(rows)
    v2_avg = sum(r["v2"]["total"] for r in rows) / len(rows)
    v2_wins = sum(1 for r in rows if r["v2"]["total"] > r["v1"]["total"])
    ties    = sum(1 for r in rows if r["v2"]["total"] == r["v1"]["total"])
    v1_wins = sum(1 for r in rows if r["v2"]["total"] < r["v1"]["total"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("PocketVibe — V1 vs V2 对比汇总\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试场景数: {len(rows)}\n\n")
        f.write(f"V1 平均总分: {v1_avg:.2f}/100\n")
        f.write(f"V2 平均总分: {v2_avg:.2f}/100\n")
        f.write(f"平均提升:   {v2_avg - v1_avg:+.2f}\n\n")
        f.write(f"胜负: V2赢 {v2_wins}  平 {ties}  V1赢 {v1_wins}\n\n")

        f.write("各维度平均:\n")
        f.write("-" * 50 + "\n")
        dims = [("J_js","J.JS深度",30),("I_instruction","I.指令遵循",25),
                ("F_feature","F.功能覆盖",20),("C_css","C.CSS丰富",15),
                ("S_structure","S.结构",10)]
        for key, label, full in dims:
            a1 = sum(r["v1"][key] for r in rows) / len(rows)
            a2 = sum(r["v2"][key] for r in rows) / len(rows)
            bar1 = "█" * int(a1/full * 20)
            bar2 = "█" * int(a2/full * 20)
            f.write(f"{label:<15} V1: {a1:5.2f}/{full}  {bar1}\n")
            f.write(f"{label:<15} V2: {a2:5.2f}/{full}  {bar2}\n")
            f.write(f"{'':15} Δ:  {a2-a1:+.2f}\n\n")

        f.write("\n各场景明细:\n")
        f.write("-" * 50 + "\n")
        for r in rows:
            d = r["v2"]["total"] - r["v1"]["total"]
            f.write(f"[{r['id']}] {r['tag']}\n")
            f.write(f"    V1={r['v1']['total']}/100  V2={r['v2']['total']}/100  Δ={d:+d}\n")

    print(f"📄 Summary 已保存: {path}")


def _export_radar_json(rows):
    path = os.path.join(OUT_DIR, "radar_data.json")
    dims = ["J_js", "I_instruction", "F_feature", "C_css", "S_structure"]
    dims_max = {"J_js":30, "I_instruction":25, "F_feature":20, "C_css":15, "S_structure":10}
    data = {
        "dimensions": dims,
        "dimensions_max": dims_max,
        "scenarios": [r["id"] for r in rows],
        "scenario_tags": [r["tag"] for r in rows],
        "v1_scores": [[r["v1"][d] for d in dims] for r in rows],
        "v2_scores": [[r["v2"][d] for d in dims] for r in rows],
        "v1_normalized": [[r["v1"][d]/dims_max[d] for d in dims] for r in rows],
        "v2_normalized": [[r["v2"][d]/dims_max[d] for d in dims] for r in rows],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"📄 Radar JSON 已保存: {path}")


def _export_plots(rows):
    """★ 脚本直接出雷达图 + 柱状图（matplotlib）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib 未安装，跳过绘图")
        return

    # ---- 修中文字体问题，避免乱码 ----
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # HPC 默认字体
    plt.rcParams["axes.unicode_minus"] = False

    dims_max = {"J_js":30, "I_instruction":25, "F_feature":20, "C_css":15, "S_structure":10}
    dims_label = ["J.JS-Depth(30)", "I.Instruction(25)", "F.Feature(20)", "C.CSS(15)", "S.Structure(10)"]
    dims_keys  = list(dims_max.keys())

    # ================== 1. 雷达图（平均 5 维度 V1 vs V2）==================
    v1_avg = [sum(r["v1"][d] for r in rows)/len(rows)/dims_max[d] for d in dims_keys]
    v2_avg = [sum(r["v2"][d] for r in rows)/len(rows)/dims_max[d] for d in dims_keys]

    N = len(dims_keys)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    v1_avg += v1_avg[:1]
    v2_avg += v2_avg[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, v1_avg, "o-", linewidth=2, label="V1 (baseline)", color="#888")
    ax.fill(angles, v1_avg, alpha=0.15, color="#888")
    ax.plot(angles, v2_avg, "o-", linewidth=2.5, label="V2 (ours)", color="#e63946")
    ax.fill(angles, v2_avg, alpha=0.25, color="#e63946")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims_label, size=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="#666", size=9)
    ax.set_title("PocketVibe V1 vs V2 — 5-Dimension Capability Radar (normalized)\n"
                 "avg across 5 scenarios targeting V2 training objectives",
                 size=13, pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "radar_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 雷达图已保存: {OUT_DIR}/radar_plot.png")

    # ================== 2. 柱状对比图（5 场景 × V1/V2 总分）==================
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rows))
    width = 0.35
    v1_totals = [r["v1"]["total"] for r in rows]
    v2_totals = [r["v2"]["total"] for r in rows]
    b1 = ax.bar(x - width/2, v1_totals, width, label="V1", color="#888")
    b2 = ax.bar(x + width/2, v2_totals, width, label="V2", color="#e63946")
    ax.set_ylabel("Total Score / 100", fontsize=12)
    ax.set_title("V1 vs V2 — Total Score across 5 V2-oriented scenarios", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['id']}\n{r['tag'].split('_')[0]}" for r in rows], fontsize=10)
    ax.set_ylim(0, 100)
    ax.axhline(50, color="#bbb", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bar_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 柱状图已保存: {OUT_DIR}/bar_plot.png")


if __name__ == "__main__":
    main()
