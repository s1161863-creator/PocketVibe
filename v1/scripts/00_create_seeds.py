#!/usr/bin/env python3
"""PocketVibe seed-data generator.

Generates 50 seed chat examples for the mobile micro-app task and writes them
to data/seed/seed_examples.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "seed" / "seed_examples.jsonl"


def shell(
    *,
    title: str,
    emoji: str,
    color_a: str,
    color_b: str,
    body: str,
    script: str,
    extra_css: str = "",
    dark: bool = False,
) -> str:
    card_bg = "#18202f" if dark else "#ffffff"
    text_color = "#f4f7fb" if dark else "#243147"
    input_bg = "#202a3a" if dark else "#ffffff"
    input_border = "#44526c" if dark else "#dde4ee"
    sub_color = "#a9b6c8" if dark else "#6b7280"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: linear-gradient(135deg, {color_a}, {color_b});
    }}
    .card {{
      width: 100%;
      max-width: 380px;
      background: {card_bg};
      color: {text_color};
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28);
    }}
    h1 {{
      font-size: 24px;
      margin-bottom: 14px;
      text-align: center;
    }}
    p.sub {{
      text-align: center;
      color: {sub_color};
      margin-bottom: 14px;
      font-size: 14px;
    }}
    input, select, textarea, button {{
      width: 100%;
      font-size: 16px;
      border: none;
      border-radius: 14px;
      margin-top: 10px;
    }}
    input, select, textarea {{
      padding: 13px 14px;
      background: {input_bg};
      color: {text_color};
      border: 2px solid {input_border};
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
    }}
    button {{
      padding: 13px 14px;
      background: linear-gradient(135deg, {color_a}, {color_b});
      color: white;
      font-weight: 700;
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
    }}
    .row {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 10px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chip {{
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.16);
      border-radius: 999px;
      font-size: 13px;
    }}
    .result {{
      font-size: 38px;
      font-weight: 800;
      text-align: center;
      margin-top: 16px;
      line-height: 1.2;
    }}
    .panel {{
      margin-top: 14px;
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.12);
    }}
    .small {{
      font-size: 14px;
      color: {sub_color};
      margin-top: 8px;
      text-align: center;
    }}
    ul {{
      list-style: none;
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }}
    li {{
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.12);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    canvas {{
      width: 100%;
      height: 240px;
      border-radius: 18px;
      background: #fff;
      margin-top: 12px;
      touch-action: none;
    }}
    {extra_css}
  </style>
</head>
<body>
  <main class="card">
    <h1>{emoji} {title}</h1>
    {body}
  </main>
  <script>
    {script}
  </script>
</body>
</html>"""


def timer_html() -> str:
    body = """
    <p class="sub">设定分钟数，开始后自动倒计时。</p>
    <input id="minutes" type="number" min="1" max="99" value="5" placeholder="分钟数">
    <div class="result" id="display">05:00</div>
    <div class="row">
      <button onclick="startTimer()">开始</button>
      <button onclick="pauseTimer()">暂停</button>
    </div>
    <button onclick="resetTimer()">重置</button>
    """
    script = """
    let remain = 300;
    let timer = null;
    function fmt(sec){ return String(Math.floor(sec/60)).padStart(2,'0') + ':' + String(sec%60).padStart(2,'0'); }
    function sync(){ document.getElementById('display').textContent = fmt(remain); }
    function startTimer(){
      if(timer) return;
      if(remain <= 0){ remain = parseInt(document.getElementById('minutes').value || '5', 10) * 60; }
      timer = setInterval(() => {
        remain -= 1;
        if(remain <= 0){
          remain = 0;
          clearInterval(timer);
          timer = null;
          document.getElementById('display').textContent = '时间到';
          return;
        }
        sync();
      }, 1000);
      sync();
    }
    function pauseTimer(){ clearInterval(timer); timer = null; }
    function resetTimer(){
      pauseTimer();
      remain = parseInt(document.getElementById('minutes').value || '5', 10) * 60;
      sync();
    }
    sync();
    """
    return shell(title="倒计时器", emoji="⏱", color_a="#667eea", color_b="#764ba2", body=body, script=script)


def stopwatch_html() -> str:
    body = """
    <p class="sub">适合训练、实验和专注记录。</p>
    <div class="result" id="display">00:00.00</div>
    <div class="row">
      <button onclick="toggle()">开始 / 停止</button>
      <button onclick="resetWatch()">清零</button>
    </div>
    """
    script = """
    let startedAt = 0;
    let elapsed = 0;
    let handle = null;
    function render(){
      const ms = elapsed;
      const m = String(Math.floor(ms/60000)).padStart(2, '0');
      const s = String(Math.floor(ms%60000/1000)).padStart(2, '0');
      const cs = String(Math.floor(ms%1000/10)).padStart(2, '0');
      document.getElementById('display').textContent = `${m}:${s}.${cs}`;
    }
    function toggle(){
      if(handle){
        clearInterval(handle);
        handle = null;
        return;
      }
      startedAt = Date.now() - elapsed;
      handle = setInterval(() => {
        elapsed = Date.now() - startedAt;
        render();
      }, 30);
    }
    function resetWatch(){
      clearInterval(handle);
      handle = null;
      elapsed = 0;
      render();
    }
    render();
    """
    return shell(title="秒表", emoji="🏁", color_a="#43cea2", color_b="#185a9d", body=body, script=script)


def dice_html() -> str:
    body = """
    <p class="sub">点击按钮随机掷出 1 到 6。</p>
    <div class="result" id="display">🎲</div>
    <button onclick="rollDice()">掷一下</button>
    """
    script = """
    const faces = ['⚀','⚁','⚂','⚃','⚄','⚅'];
    function rollDice(){
      const index = Math.floor(Math.random() * faces.length);
      document.getElementById('display').textContent = faces[index];
    }
    """
    return shell(title="掷骰子", emoji="🎲", color_a="#ff758c", color_b="#ff7eb3", body=body, script=script)


def coin_html() -> str:
    body = """
    <p class="sub">随机得到正面或反面。</p>
    <div class="result" id="display">🪙</div>
    <button onclick="flipCoin()">抛硬币</button>
    """
    script = """
    function flipCoin(){
      document.getElementById('display').textContent = Math.random() > 0.5 ? '正面' : '反面';
    }
    """
    return shell(title="抛硬币", emoji="🪙", color_a="#f6d365", color_b="#fda085", body=body, script=script)


def picker_html() -> str:
    body = """
    <p class="sub">输入选项后随机帮你做决定。</p>
    <input id="item" placeholder="比如：火锅、面条、烧烤">
    <div class="row">
      <button onclick="addItem()">添加</button>
      <button onclick="pickOne()">随机选择</button>
    </div>
    <div class="chips" id="chips"></div>
    <div class="result" id="display">?</div>
    """
    script = """
    const items = ['火锅','寿司','面条','汉堡'];
    function render(){
      document.getElementById('chips').innerHTML = items.map(x => `<span class="chip">${x}</span>`).join('');
    }
    function addItem(){
      const input = document.getElementById('item');
      const value = input.value.trim();
      if(!value) return;
      items.push(value);
      input.value = '';
      render();
    }
    function pickOne(){
      if(!items.length) return;
      document.getElementById('display').textContent = items[Math.floor(Math.random() * items.length)];
    }
    render();
    """
    return shell(title="随机选择器", emoji="🍜", color_a="#fa709a", color_b="#fee140", body=body, script=script)


def bmi_html() -> str:
    body = """
    <p class="sub">输入身高体重，快速计算 BMI。</p>
    <input id="height" type="number" placeholder="身高 cm">
    <input id="weight" type="number" placeholder="体重 kg">
    <button onclick="calcBmi()">计算 BMI</button>
    <div class="result" id="display">-</div>
    <p class="small" id="note"></p>
    """
    script = """
    function calcBmi(){
      const h = parseFloat(document.getElementById('height').value) / 100;
      const w = parseFloat(document.getElementById('weight').value);
      if(!h || !w) return;
      const bmi = (w / (h*h)).toFixed(1);
      let label = '正常';
      if(bmi < 18.5) label = '偏瘦';
      else if(bmi >= 24 && bmi < 28) label = '偏胖';
      else if(bmi >= 28) label = '肥胖';
      document.getElementById('display').textContent = bmi;
      document.getElementById('note').textContent = `状态：${label}`;
    }
    """
    return shell(title="BMI 计算器", emoji="📊", color_a="#4facfe", color_b="#00f2fe", body=body, script=script)


def dual_converter_html(title: str, emoji: str, a_label: str, b_label: str, a_id: str, b_id: str, a_to_b: str, b_to_a: str, color_a: str, color_b: str, dark: bool = False) -> str:
    body = f"""
    <p class="sub">双向实时换算。</p>
    <input id="{a_id}" type="number" placeholder="{a_label}" oninput="convertA()">
    <input id="{b_id}" type="number" placeholder="{b_label}" oninput="convertB()">
    """
    script = f"""
    function convertA(){{
      const value = parseFloat(document.getElementById('{a_id}').value);
      if(Number.isNaN(value)) return;
      document.getElementById('{b_id}').value = ({a_to_b}).toFixed(2);
    }}
    function convertB(){{
      const value = parseFloat(document.getElementById('{b_id}').value);
      if(Number.isNaN(value)) return;
      document.getElementById('{a_id}').value = ({b_to_a}).toFixed(2);
    }}
    """
    return shell(title=title, emoji=emoji, color_a=color_a, color_b=color_b, body=body, script=script, dark=dark)


def scoreboard_html() -> str:
    body = """
    <p class="sub">两队记分，支持加减分。</p>
    <div class="row">
      <div class="panel">
        <div style="text-align:center;font-weight:700;">A 队</div>
        <div class="result" id="a">0</div>
        <div class="row">
          <button onclick="change('a',1)">+1</button>
          <button onclick="change('a',-1)">-1</button>
        </div>
      </div>
      <div class="panel">
        <div style="text-align:center;font-weight:700;">B 队</div>
        <div class="result" id="b">0</div>
        <div class="row">
          <button onclick="change('b',1)">+1</button>
          <button onclick="change('b',-1)">-1</button>
        </div>
      </div>
    </div>
    <button onclick="resetBoard()">重置比分</button>
    """
    script = """
    function change(id, delta){
      const el = document.getElementById(id);
      el.textContent = Math.max(0, parseInt(el.textContent, 10) + delta);
    }
    function resetBoard(){
      document.getElementById('a').textContent = '0';
      document.getElementById('b').textContent = '0';
    }
    """
    return shell(title="记分板", emoji="🏆", color_a="#f6a623", color_b="#f76b1c", body=body, script=script)


def guess_html() -> str:
    body = """
    <p class="sub">猜一个 1 到 100 的数字。</p>
    <input id="guess" type="number" min="1" max="100" placeholder="输入你的猜测">
    <div class="row">
      <button onclick="submitGuess()">提交</button>
      <button onclick="resetGame()">重开</button>
    </div>
    <div class="result" id="display">?</div>
    <p class="small" id="note">我已经想好数字了。</p>
    """
    script = """
    let answer = Math.floor(Math.random() * 100) + 1;
    let count = 0;
    function submitGuess(){
      const value = parseInt(document.getElementById('guess').value, 10);
      if(Number.isNaN(value)) return;
      count += 1;
      if(value === answer){
        document.getElementById('display').textContent = '🎉';
        document.getElementById('note').textContent = `答对了，共猜了 ${count} 次`;
      } else if(value < answer){
        document.getElementById('display').textContent = '更大';
        document.getElementById('note').textContent = '再往上猜一点';
      } else {
        document.getElementById('display').textContent = '更小';
        document.getElementById('note').textContent = '再往下猜一点';
      }
    }
    function resetGame(){
      answer = Math.floor(Math.random() * 100) + 1;
      count = 0;
      document.getElementById('display').textContent = '?';
      document.getElementById('note').textContent = '新的数字已经生成';
      document.getElementById('guess').value = '';
    }
    """
    return shell(title="猜数字", emoji="🔢", color_a="#8e2de2", color_b="#4a00e0", body=body, script=script)


def calculator_html() -> str:
    body = """
    <p class="sub">基础四则运算。</p>
    <input id="screen" value="0" readonly>
    <div class="row" id="keys"></div>
    """
    extra_css = """
    #keys { grid-template-columns: repeat(4, 1fr); }
    #keys button { min-height: 54px; }
    """
    script = """
    const keys = ['7','8','9','/','4','5','6','*','1','2','3','-','0','.','=','+','C','DEL'];
    let expr = '';
    const container = document.getElementById('keys');
    keys.forEach(key => {
      const btn = document.createElement('button');
      btn.textContent = key;
      btn.onclick = () => tap(key);
      container.appendChild(btn);
    });
    function tap(key){
      if(key === 'C'){ expr = ''; }
      else if(key === 'DEL'){ expr = expr.slice(0, -1); }
      else if(key === '='){
        try { expr = String(Function('"use strict";return (' + expr + ')')()); }
        catch (_err) { expr = '错误'; }
      } else {
        expr += key;
      }
      document.getElementById('screen').value = expr || '0';
    }
    """
    return shell(title="计算器", emoji="🧮", color_a="#141e30", color_b="#243b55", body=body, script=script, extra_css=extra_css, dark=True)


def date_diff_html() -> str:
    body = """
    <p class="sub">输入生日，自动计算年龄和天数。</p>
    <input id="birthday" type="date" onchange="calcAge()">
    <div class="result" id="display">-</div>
    <p class="small" id="note"></p>
    """
    script = """
    function calcAge(){
      const value = document.getElementById('birthday').value;
      if(!value) return;
      const birth = new Date(value);
      const now = new Date();
      const days = Math.floor((now - birth) / 86400000);
      const years = Math.floor(days / 365.25);
      document.getElementById('display').textContent = years + ' 岁';
      document.getElementById('note').textContent = '已经度过 ' + days.toLocaleString() + ' 天';
    }
    """
    return shell(title="年龄计算器", emoji="🎂", color_a="#e0c3fc", color_b="#8ec5fc", body=body, script=script)


def percentage_html() -> str:
    body = """
    <p class="sub">计算某个数值占总数的比例。</p>
    <input id="part" type="number" placeholder="部分数值">
    <input id="whole" type="number" placeholder="总数">
    <button onclick="calcPercent()">立即计算</button>
    <div class="result" id="display">0%</div>
    """
    script = """
    function calcPercent(){
      const part = parseFloat(document.getElementById('part').value);
      const whole = parseFloat(document.getElementById('whole').value);
      if(!whole) return;
      document.getElementById('display').textContent = (part / whole * 100).toFixed(1) + '%';
    }
    """
    return shell(title="百分比计算器", emoji="📈", color_a="#56ab2f", color_b="#a8e063", body=body, script=script)


def day_countdown_html() -> str:
    body = """
    <p class="sub">看看目标日期还有多少天。</p>
    <input id="target" type="date" onchange="calcCountdown()">
    <div class="result" id="display">-</div>
    <p class="small" id="note"></p>
    """
    script = """
    function calcCountdown(){
      const value = document.getElementById('target').value;
      if(!value) return;
      const target = new Date(value);
      const diff = Math.ceil((target - new Date()) / 86400000);
      document.getElementById('display').textContent = Math.abs(diff) + ' 天';
      document.getElementById('note').textContent = diff >= 0 ? `距离目标还有 ${diff} 天` : `已经过去 ${Math.abs(diff)} 天`;
    }
    """
    return shell(title="倒数日", emoji="📅", color_a="#ff9a9e", color_b="#fad0c4", body=body, script=script)


def password_html() -> str:
    body = """
    <p class="sub">一键生成随机密码。</p>
    <input id="length" type="range" min="6" max="32" value="14" oninput="document.getElementById('lenText').textContent=this.value">
    <p class="small">长度：<span id="lenText">14</span></p>
    <div class="result" id="display" style="font-size:24px;word-break:break-all;">点击下方生成</div>
    <div class="row">
      <button onclick="generatePassword()">生成</button>
      <button onclick="copyPassword()">复制</button>
    </div>
    """
    script = """
    function generatePassword(){
      const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%';
      const len = parseInt(document.getElementById('length').value, 10);
      let out = '';
      for(let i=0;i<len;i++) out += chars[Math.floor(Math.random() * chars.length)];
      document.getElementById('display').textContent = out;
    }
    function copyPassword(){
      navigator.clipboard.writeText(document.getElementById('display').textContent);
    }
    """
    return shell(title="密码生成器", emoji="🔐", color_a="#0f2027", color_b="#2c5364", body=body, script=script, dark=True)


def color_html() -> str:
    body = """
    <p class="sub">随机生成一组颜色。</p>
    <div class="panel" id="swatch" style="height:120px;"></div>
    <div class="result" id="display" style="font-size:28px;">#667EEA</div>
    <div class="row">
      <button onclick="generateColor()">换一个</button>
      <button onclick="copyHex()">复制</button>
    </div>
    """
    script = """
    function generateColor(){
      const color = '#' + Math.floor(Math.random() * 16777215).toString(16).padStart(6, '0').toUpperCase();
      document.getElementById('swatch').style.background = color;
      document.getElementById('display').textContent = color;
    }
    function copyHex(){ navigator.clipboard.writeText(document.getElementById('display').textContent); }
    generateColor();
    """
    return shell(title="随机颜色生成器", emoji="🎨", color_a="#667eea", color_b="#764ba2", body=body, script=script)


def rps_html() -> str:
    body = """
    <p class="sub">石头、剪刀、布，和手机对战。</p>
    <div class="row">
      <button onclick="play(0)">✊</button>
      <button onclick="play(1)">✋</button>
      <button onclick="play(2)">✌️</button>
    </div>
    <div class="result" id="display">准备出招</div>
    <p class="small" id="note"></p>
    """
    script = """
    const signs = ['✊','✋','✌️'];
    const words = ['平局','你赢了','你输了'];
    function play(me){
      const ai = Math.floor(Math.random() * 3);
      const result = (3 + me - ai) % 3;
      document.getElementById('display').textContent = `${signs[me]} vs ${signs[ai]}`;
      document.getElementById('note').textContent = words[result];
    }
    """
    return shell(title="石头剪刀布", emoji="✊", color_a="#fc6076", color_b="#ff9a44", body=body, script=script)


def counter_html() -> str:
    body = """
    <p class="sub">简单计数，加一减一。</p>
    <div class="result" id="display">0</div>
    <div class="row">
      <button onclick="change(-1)">-1</button>
      <button onclick="change(1)">+1</button>
    </div>
    <button onclick="resetCounter()">重置</button>
    """
    script = """
    function change(step){
      const el = document.getElementById('display');
      el.textContent = String(parseInt(el.textContent, 10) + step);
    }
    function resetCounter(){ document.getElementById('display').textContent = '0'; }
    """
    return shell(title="计数器", emoji="🔢", color_a="#11998e", color_b="#38ef7d", body=body, script=script)


def word_count_html() -> str:
    body = """
    <p class="sub">输入文字后实时统计。</p>
    <textarea id="text" placeholder="在这里输入内容..." oninput="countText()"></textarea>
    <div class="row">
      <div class="panel"><strong id="chars">0</strong><div>字符数</div></div>
      <div class="panel"><strong id="words">0</strong><div>词数</div></div>
    </div>
    """
    script = """
    function countText(){
      const value = document.getElementById('text').value;
      document.getElementById('chars').textContent = String(value.length);
      const words = value.trim() ? value.trim().split(/\\s+/).length : 0;
      document.getElementById('words').textContent = String(words);
    }
    """
    return shell(title="字数统计", emoji="📝", color_a="#b06ab3", color_b="#4568dc", body=body, script=script)


def random_number_html() -> str:
    body = """
    <p class="sub">生成指定范围内的随机整数。</p>
    <input id="min" type="number" value="1" placeholder="最小值">
    <input id="max" type="number" value="100" placeholder="最大值">
    <button onclick="generateNumber()">生成随机数</button>
    <div class="result" id="display">?</div>
    """
    script = """
    function generateNumber(){
      const min = parseInt(document.getElementById('min').value, 10);
      const max = parseInt(document.getElementById('max').value, 10);
      if(Number.isNaN(min) || Number.isNaN(max) || max < min) return;
      document.getElementById('display').textContent = String(Math.floor(Math.random() * (max - min + 1)) + min);
    }
    """
    return shell(title="随机数生成器", emoji="🎯", color_a="#4b6cb7", color_b="#182848", body=body, script=script)


def base_convert_html() -> str:
    body = """
    <p class="sub">十进制、二进制、八进制、十六进制互看。</p>
    <input id="decimal" type="number" placeholder="输入十进制数值" oninput="convertBase()">
    <div class="panel">二进制：<strong id="b2">-</strong></div>
    <div class="panel">八进制：<strong id="b8">-</strong></div>
    <div class="panel">十六进制：<strong id="b16">-</strong></div>
    """
    script = """
    function convertBase(){
      const value = parseInt(document.getElementById('decimal').value, 10);
      if(Number.isNaN(value)) return;
      document.getElementById('b2').textContent = value.toString(2);
      document.getElementById('b8').textContent = value.toString(8);
      document.getElementById('b16').textContent = value.toString(16).toUpperCase();
    }
    """
    return shell(title="进制转换器", emoji="🔄", color_a="#232526", color_b="#414345", body=body, script=script, dark=True)


def tip_html() -> str:
    body = """
    <p class="sub">输入账单和小费比例。</p>
    <input id="bill" type="number" placeholder="账单金额">
    <input id="tip" type="number" value="15" placeholder="小费比例 %">
    <button onclick="calcTip()">计算</button>
    <div class="result" id="display">¥0.00</div>
    <p class="small" id="note"></p>
    """
    script = """
    function calcTip(){
      const bill = parseFloat(document.getElementById('bill').value);
      const rate = parseFloat(document.getElementById('tip').value);
      if(Number.isNaN(bill) || Number.isNaN(rate)) return;
      const tip = bill * rate / 100;
      document.getElementById('display').textContent = '¥' + tip.toFixed(2);
      document.getElementById('note').textContent = '总金额：¥' + (bill + tip).toFixed(2);
    }
    """
    return shell(title="小费计算器", emoji="💸", color_a="#f7971e", color_b="#ffd200", body=body, script=script)


def split_bill_html() -> str:
    body = """
    <p class="sub">聚餐后快速 AA。</p>
    <input id="bill" type="number" placeholder="总金额">
    <input id="people" type="number" value="2" placeholder="人数">
    <button onclick="splitBill()">平均分摊</button>
    <div class="result" id="display">¥0.00</div>
    """
    script = """
    function splitBill(){
      const bill = parseFloat(document.getElementById('bill').value);
      const people = parseFloat(document.getElementById('people').value);
      if(!bill || !people) return;
      document.getElementById('display').textContent = '¥' + (bill / people).toFixed(2);
    }
    """
    return shell(title="AA 分账", emoji="🍽️", color_a="#ff9966", color_b="#ff5e62", body=body, script=script)


def area_html() -> str:
    body = """
    <p class="sub">矩形面积计算。</p>
    <input id="width" type="number" placeholder="宽">
    <input id="height" type="number" placeholder="高">
    <button onclick="calcArea()">计算面积</button>
    <div class="result" id="display">0</div>
    """
    script = """
    function calcArea(){
      const w = parseFloat(document.getElementById('width').value);
      const h = parseFloat(document.getElementById('height').value);
      if(Number.isNaN(w) || Number.isNaN(h)) return;
      document.getElementById('display').textContent = (w * h).toFixed(2);
    }
    """
    return shell(title="面积计算器", emoji="📐", color_a="#3a7bd5", color_b="#3a6073", body=body, script=script)


def time_diff_html() -> str:
    body = """
    <p class="sub">计算两个时间点之间的差值。</p>
    <input id="start" type="time">
    <input id="end" type="time">
    <button onclick="calcGap()">计算时间差</button>
    <div class="result" id="display">0 分钟</div>
    """
    script = """
    function calcGap(){
      const start = document.getElementById('start').value;
      const end = document.getElementById('end').value;
      if(!start || !end) return;
      const [sh, sm] = start.split(':').map(Number);
      const [eh, em] = end.split(':').map(Number);
      let diff = (eh * 60 + em) - (sh * 60 + sm);
      if(diff < 0) diff += 1440;
      document.getElementById('display').textContent = diff + ' 分钟';
    }
    """
    return shell(title="时间间隔计算器", emoji="🕒", color_a="#396afc", color_b="#2948ff", body=body, script=script)


def vote_html() -> str:
    body = """
    <p class="sub">两项投票实时计数。</p>
    <div class="row">
      <button onclick="vote('a')">方案 A</button>
      <button onclick="vote('b')">方案 B</button>
    </div>
    <div class="row">
      <div class="panel">A：<strong id="a">0</strong></div>
      <div class="panel">B：<strong id="b">0</strong></div>
    </div>
    <button onclick="resetVote()">清空投票</button>
    """
    script = """
    function vote(key){
      const el = document.getElementById(key);
      el.textContent = String(parseInt(el.textContent, 10) + 1);
    }
    function resetVote(){
      document.getElementById('a').textContent = '0';
      document.getElementById('b').textContent = '0';
    }
    """
    return shell(title="投票器", emoji="🗳️", color_a="#7f7fd5", color_b="#86a8e7", body=body, script=script)


def wheel_html() -> str:
    body = """
    <p class="sub">随机轮盘，帮你做选择。</p>
    <div class="chips" id="chips"></div>
    <button onclick="spin()">开始抽选</button>
    <div class="result" id="display">等你开始</div>
    """
    script = """
    const options = ['学习','运动','散步','看书','打扫'];
    document.getElementById('chips').innerHTML = options.map(x => `<span class="chip">${x}</span>`).join('');
    function spin(){
      document.getElementById('display').textContent = options[Math.floor(Math.random() * options.length)];
    }
    """
    return shell(title="幸运转盘", emoji="🎡", color_a="#c471ed", color_b="#f64f59", body=body, script=script)


def water_html() -> str:
    body = """
    <p class="sub">记录今天喝了几杯水。</p>
    <div class="result" id="display">0 杯</div>
    <div class="row">
      <button onclick="change(1)">喝了一杯</button>
      <button onclick="change(-1)">撤回一杯</button>
    </div>
    <button onclick="resetWater()">今日清零</button>
    """
    script = """
    let cups = 0;
    function render(){ document.getElementById('display').textContent = cups + ' 杯'; }
    function change(step){ cups = Math.max(0, cups + step); render(); }
    function resetWater(){ cups = 0; render(); }
    render();
    """
    return shell(title="喝水记录器", emoji="💧", color_a="#00c6ff", color_b="#0072ff", body=body, script=script)


def todo_html() -> str:
    body = """
    <p class="sub">添加待办并支持勾选。</p>
    <input id="todoInput" placeholder="输入待办事项">
    <button onclick="addTodo()">添加待办</button>
    <ul id="todoList"></ul>
    """
    script = """
    function addTodo(){
      const input = document.getElementById('todoInput');
      const text = input.value.trim();
      if(!text) return;
      const li = document.createElement('li');
      li.innerHTML = `<label style="display:flex;align-items:center;gap:10px;flex:1;"><input type="checkbox"> <span>${text}</span></label><button style="width:auto;padding:8px 12px;margin-top:0;" onclick="this.parentElement.remove()">删除</button>`;
      document.getElementById('todoList').appendChild(li);
      input.value = '';
    }
    """
    return shell(title="待办清单", emoji="✅", color_a="#11998e", color_b="#38ef7d", body=body, script=script)


def pomodoro_html() -> str:
    body = """
    <p class="sub">25 分钟专注，5 分钟休息。</p>
    <div class="result" id="display">25:00</div>
    <p class="small" id="note">当前模式：专注</p>
    <div class="row">
      <button onclick="startPomodoro()">开始</button>
      <button onclick="switchMode()">切换模式</button>
    </div>
    """
    script = """
    let focus = true;
    let remain = 1500;
    let handle = null;
    function paint(){
      const m = String(Math.floor(remain / 60)).padStart(2,'0');
      const s = String(remain % 60).padStart(2,'0');
      document.getElementById('display').textContent = `${m}:${s}`;
      document.getElementById('note').textContent = '当前模式：' + (focus ? '专注' : '休息');
    }
    function switchMode(){
      focus = !focus;
      remain = focus ? 1500 : 300;
      clearInterval(handle);
      handle = null;
      paint();
    }
    function startPomodoro(){
      if(handle) return;
      handle = setInterval(() => {
        remain -= 1;
        if(remain <= 0){
          switchMode();
        } else {
          paint();
        }
      }, 1000);
    }
    paint();
    """
    return shell(title="番茄钟", emoji="🍅", color_a="#f857a6", color_b="#ff5858", body=body, script=script)


def notes_html() -> str:
    body = """
    <p class="sub">记录便签内容。</p>
    <textarea id="memo" placeholder="写点什么..."></textarea>
    <button onclick="saveMemo()">保存便签</button>
    <div class="panel" id="saved">还没有保存内容</div>
    """
    script = """
    function saveMemo(){
      const text = document.getElementById('memo').value.trim();
      document.getElementById('saved').textContent = text || '内容为空';
    }
    """
    return shell(title="便签本", emoji="🗒️", color_a="#f7971e", color_b="#ffd200", body=body, script=script)


def mood_html() -> str:
    body = """
    <p class="sub">记录今天的心情。</p>
    <div class="row">
      <button onclick="setMood('😀 很开心')">😀</button>
      <button onclick="setMood('😐 一般般')">😐</button>
      <button onclick="setMood('😴 有点累')">😴</button>
    </div>
    <div class="result" id="display">请选择</div>
    """
    script = """
    function setMood(text){ document.getElementById('display').textContent = text; }
    """
    return shell(title="心情记录", emoji="🌤️", color_a="#fbc2eb", color_b="#a6c1ee", body=body, script=script)


def habit_html() -> str:
    body = """
    <p class="sub">连续打卡，保持习惯。</p>
    <div class="result" id="display">0 天</div>
    <div class="row">
      <button onclick="checkIn()">今日打卡</button>
      <button onclick="clearHabit()">重置</button>
    </div>
    """
    script = """
    let days = 0;
    function draw(){ document.getElementById('display').textContent = days + ' 天'; }
    function checkIn(){ days += 1; draw(); }
    function clearHabit(){ days = 0; draw(); }
    draw();
    """
    return shell(title="习惯打卡", emoji="📌", color_a="#4e54c8", color_b="#8f94fb", body=body, script=script)


def expense_html() -> str:
    body = """
    <p class="sub">简单记账，累计支出。</p>
    <input id="amount" type="number" placeholder="输入金额">
    <input id="name" placeholder="消费项目">
    <button onclick="addExpense()">记录一笔</button>
    <ul id="list"></ul>
    <div class="result" id="display">¥0.00</div>
    """
    script = """
    let total = 0;
    function addExpense(){
      const amount = parseFloat(document.getElementById('amount').value);
      const name = document.getElementById('name').value.trim() || '未命名';
      if(Number.isNaN(amount)) return;
      total += amount;
      const li = document.createElement('li');
      li.innerHTML = `<span>${name}</span><strong>¥${amount.toFixed(2)}</strong>`;
      document.getElementById('list').appendChild(li);
      document.getElementById('display').textContent = '¥' + total.toFixed(2);
      document.getElementById('amount').value = '';
      document.getElementById('name').value = '';
    }
    """
    return shell(title="记账本", emoji="💰", color_a="#1d976c", color_b="#93f9b9", body=body, script=script)


def metronome_html() -> str:
    body = """
    <p class="sub">输入 BPM，跟着节拍点头。</p>
    <input id="bpm" type="number" value="90" placeholder="节拍 BPM">
    <div class="result" id="display">●</div>
    <div class="row">
      <button onclick="startMetro()">开始</button>
      <button onclick="stopMetro()">停止</button>
    </div>
    """
    script = """
    let metro = null;
    let on = false;
    function flash(){
      on = !on;
      document.getElementById('display').textContent = on ? '●' : '○';
    }
    function startMetro(){
      stopMetro();
      const bpm = parseFloat(document.getElementById('bpm').value) || 90;
      metro = setInterval(flash, 60000 / bpm);
    }
    function stopMetro(){ clearInterval(metro); metro = null; document.getElementById('display').textContent = '●'; }
    """
    return shell(title="节拍器", emoji="🎵", color_a="#614385", color_b="#516395", body=body, script=script, dark=True)


def drawing_html() -> str:
    body = """
    <p class="sub">触摸画板，随手涂鸦。</p>
    <input id="color" type="color" value="#ff5e62">
    <button onclick="clearCanvas()">清空画板</button>
    <canvas id="board" width="340" height="240"></canvas>
    """
    script = """
    const canvas = document.getElementById('board');
    const ctx = canvas.getContext('2d');
    let drawing = false;
    function point(event){
      const rect = canvas.getBoundingClientRect();
      const source = event.touches ? event.touches[0] : event;
      return { x: (source.clientX - rect.left) * (canvas.width / rect.width), y: (source.clientY - rect.top) * (canvas.height / rect.height) };
    }
    function begin(event){
      drawing = true;
      const p = point(event);
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
    }
    function move(event){
      if(!drawing) return;
      event.preventDefault();
      const p = point(event);
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.strokeStyle = document.getElementById('color').value;
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    function end(){ drawing = false; }
    function clearCanvas(){ ctx.clearRect(0, 0, canvas.width, canvas.height); }
    canvas.addEventListener('mousedown', begin);
    canvas.addEventListener('mousemove', move);
    canvas.addEventListener('mouseup', end);
    canvas.addEventListener('mouseleave', end);
    canvas.addEventListener('touchstart', begin, {passive:false});
    canvas.addEventListener('touchmove', move, {passive:false});
    canvas.addEventListener('touchend', end);
    """
    return shell(title="简易画板", emoji="🖍️", color_a="#ff9966", color_b="#ff5e62", body=body, script=script)


def gpa_html() -> str:
    body = """
    <p class="sub">输入三门课成绩，快速估算 GPA。</p>
    <input id="s1" type="number" placeholder="课程 1 成绩">
    <input id="s2" type="number" placeholder="课程 2 成绩">
    <input id="s3" type="number" placeholder="课程 3 成绩">
    <button onclick="calcGpa()">计算 GPA</button>
    <div class="result" id="display">0.00</div>
    """
    script = """
    function point(score){
      if(score >= 90) return 4.0;
      if(score >= 85) return 3.7;
      if(score >= 82) return 3.3;
      if(score >= 78) return 3.0;
      if(score >= 75) return 2.7;
      if(score >= 72) return 2.3;
      if(score >= 68) return 2.0;
      if(score >= 64) return 1.5;
      if(score >= 60) return 1.0;
      return 0;
    }
    function calcGpa(){
      const scores = ['s1','s2','s3'].map(id => parseFloat(document.getElementById(id).value)).filter(x => !Number.isNaN(x));
      if(!scores.length) return;
      const gpa = scores.reduce((sum, score) => sum + point(score), 0) / scores.length;
      document.getElementById('display').textContent = gpa.toFixed(2);
    }
    """
    return shell(title="GPA 计算器", emoji="🎓", color_a="#36d1dc", color_b="#5b86e5", body=body, script=script)


def flashcard_html() -> str:
    body = """
    <p class="sub">点击切换卡片内容。</p>
    <div class="panel" id="card" style="text-align:center;font-size:28px;font-weight:700;">apple</div>
    <div class="row">
      <button onclick="flipCard()">翻面</button>
      <button onclick="nextCard()">下一张</button>
    </div>
    """
    script = """
    const cards = [
      { front: 'apple', back: '苹果' },
      { front: 'book', back: '书' },
      { front: 'water', back: '水' }
    ];
    let index = 0;
    let showFront = true;
    function draw(){
      document.getElementById('card').textContent = showFront ? cards[index].front : cards[index].back;
    }
    function flipCard(){ showFront = !showFront; draw(); }
    function nextCard(){ index = (index + 1) % cards.length; showFront = true; draw(); }
    draw();
    """
    return shell(title="单词卡片", emoji="📚", color_a="#8EC5FC", color_b="#E0C3FC", body=body, script=script)


def breathing_html() -> str:
    body = """
    <p class="sub">4 秒吸气，4 秒呼气。</p>
    <div class="result" id="display">准备开始</div>
    <button onclick="startBreathing()">开始呼吸练习</button>
    """
    script = """
    let breath = null;
    function startBreathing(){
      clearInterval(breath);
      let phase = 0;
      const phases = ['吸气 4 秒', '屏住 4 秒', '呼气 4 秒', '停留 4 秒'];
      document.getElementById('display').textContent = phases[0];
      breath = setInterval(() => {
        phase = (phase + 1) % phases.length;
        document.getElementById('display').textContent = phases[phase];
      }, 4000);
    }
    """
    return shell(title="呼吸训练器", emoji="🌬️", color_a="#56ccf2", color_b="#2f80ed", body=body, script=script)


def pace_html() -> str:
    body = """
    <p class="sub">输入总距离和总耗时，算出平均配速。</p>
    <input id="km" type="number" step="0.1" placeholder="距离 km">
    <input id="minutes" type="number" placeholder="总时长 分钟">
    <button onclick="calcPace()">计算配速</button>
    <div class="result" id="display">0:00 /km</div>
    """
    script = """
    function calcPace(){
      const km = parseFloat(document.getElementById('km').value);
      const mins = parseFloat(document.getElementById('minutes').value);
      if(!km || !mins) return;
      const pace = mins / km;
      const m = Math.floor(pace);
      const s = String(Math.round((pace - m) * 60)).padStart(2, '0');
      document.getElementById('display').textContent = `${m}:${s} /km`;
    }
    """
    return shell(title="跑步配速计算器", emoji="🏃", color_a="#00b09b", color_b="#96c93d", body=body, script=script)


def sleep_html() -> str:
    body = """
    <p class="sub">计算昨晚睡了多久。</p>
    <input id="bed" type="time">
    <input id="wake" type="time">
    <button onclick="calcSleep()">计算睡眠时长</button>
    <div class="result" id="display">0 小时</div>
    """
    script = """
    function calcSleep(){
      const bed = document.getElementById('bed').value;
      const wake = document.getElementById('wake').value;
      if(!bed || !wake) return;
      const [bh, bm] = bed.split(':').map(Number);
      const [wh, wm] = wake.split(':').map(Number);
      let mins = (wh * 60 + wm) - (bh * 60 + bm);
      if(mins < 0) mins += 1440;
      document.getElementById('display').textContent = (mins / 60).toFixed(1) + ' 小时';
    }
    """
    return shell(title="睡眠时长计算器", emoji="😴", color_a="#4b79a1", color_b="#283e51", body=body, script=script, dark=True)


def quote_html() -> str:
    body = """
    <p class="sub">随机来一句鼓励的话。</p>
    <div class="panel" id="quote" style="text-align:center;font-size:22px;">今天也值得认真过。</div>
    <button onclick="nextQuote()">换一句</button>
    """
    script = """
    const quotes = ['今天也值得认真过。', '别急，进步本来就不喧哗。', '把复杂的事做简单，就是能力。', '先开始，剩下的路会慢慢清楚。'];
    function nextQuote(){
      document.getElementById('quote').textContent = quotes[Math.floor(Math.random() * quotes.length)];
    }
    """
    return shell(title="随机语录", emoji="💡", color_a="#f7971e", color_b="#ffd200", body=body, script=script)


def speed_test_html() -> str:
    body = """
    <p class="sub">30 秒内看看你能点多少次。</p>
    <div class="result" id="time">30</div>
    <div class="result" id="score">0 次</div>
    <div class="row">
      <button onclick="startTest()">开始测试</button>
      <button onclick="tap()">点击</button>
    </div>
    """
    script = """
    let time = 30;
    let score = 0;
    let handle = null;
    function render(){
      document.getElementById('time').textContent = String(time);
      document.getElementById('score').textContent = score + ' 次';
    }
    function startTest(){
      clearInterval(handle);
      time = 30;
      score = 0;
      render();
      handle = setInterval(() => {
        time -= 1;
        if(time <= 0){ clearInterval(handle); handle = null; }
        render();
      }, 1000);
    }
    function tap(){ if(handle) { score += 1; render(); } }
    render();
    """
    return shell(title="点击速度测试", emoji="⚡", color_a="#fc4a1a", color_b="#f7b733", body=body, script=script)


def packing_html() -> str:
    body = """
    <p class="sub">旅行前列好打包清单。</p>
    <input id="pack" placeholder="输入物品">
    <button onclick="addPack()">加入清单</button>
    <ul id="packs"></ul>
    """
    script = """
    function addPack(){
      const input = document.getElementById('pack');
      const value = input.value.trim();
      if(!value) return;
      const li = document.createElement('li');
      li.innerHTML = `<span>${value}</span><input type="checkbox">`;
      document.getElementById('packs').appendChild(li);
      input.value = '';
    }
    """
    return shell(title="行李清单", emoji="🧳", color_a="#7b4397", color_b="#dc2430", body=body, script=script)


def budget_html() -> str:
    body = """
    <p class="sub">按 50/30/20 粗略分配预算。</p>
    <input id="income" type="number" placeholder="月收入">
    <button onclick="calcBudget()">开始分配</button>
    <div class="panel">必要支出：<strong id="need">0</strong></div>
    <div class="panel">想要支出：<strong id="want">0</strong></div>
    <div class="panel">储蓄投资：<strong id="save">0</strong></div>
    """
    script = """
    function calcBudget(){
      const income = parseFloat(document.getElementById('income').value);
      if(Number.isNaN(income)) return;
      document.getElementById('need').textContent = '¥' + (income * 0.5).toFixed(2);
      document.getElementById('want').textContent = '¥' + (income * 0.3).toFixed(2);
      document.getElementById('save').textContent = '¥' + (income * 0.2).toFixed(2);
    }
    """
    return shell(title="预算分配器", emoji="📦", color_a="#1e3c72", color_b="#2a5298", body=body, script=script, dark=True)


def savings_html() -> str:
    body = """
    <p class="sub">看看离存款目标还有多远。</p>
    <input id="goal" type="number" placeholder="目标金额">
    <input id="saved" type="number" placeholder="已存金额">
    <button onclick="calcSavings()">计算进度</button>
    <div class="result" id="display">0%</div>
    """
    script = """
    function calcSavings(){
      const goal = parseFloat(document.getElementById('goal').value);
      const saved = parseFloat(document.getElementById('saved').value);
      if(!goal) return;
      document.getElementById('display').textContent = ((saved / goal) * 100).toFixed(1) + '%';
    }
    """
    return shell(title="存款目标追踪器", emoji="🎯", color_a="#2b5876", color_b="#4e4376", body=body, script=script)


def calories_html() -> str:
    body = """
    <p class="sub">选择食物估算热量。</p>
    <select id="food">
      <option value="116">米饭 100g</option>
      <option value="250">炸鸡 100g</option>
      <option value="89">香蕉 100g</option>
      <option value="52">苹果 100g</option>
    </select>
    <input id="grams" type="number" value="100" placeholder="克数">
    <button onclick="calcCalories()">计算热量</button>
    <div class="result" id="display">0 kcal</div>
    """
    script = """
    function calcCalories(){
      const unit = parseFloat(document.getElementById('food').value);
      const grams = parseFloat(document.getElementById('grams').value);
      if(Number.isNaN(unit) || Number.isNaN(grams)) return;
      document.getElementById('display').textContent = (unit * grams / 100).toFixed(0) + ' kcal';
    }
    """
    return shell(title="卡路里计算器", emoji="🥗", color_a="#56ab2f", color_b="#a8e063", body=body, script=script)


def heart_rate_html() -> str:
    body = """
    <p class="sub">输入年龄，估算训练心率区间。</p>
    <input id="age" type="number" placeholder="年龄">
    <button onclick="calcHeart()">计算区间</button>
    <div class="panel">最大心率：<strong id="maxHr">-</strong></div>
    <div class="panel">燃脂区间：<strong id="zone">-</strong></div>
    """
    script = """
    function calcHeart(){
      const age = parseFloat(document.getElementById('age').value);
      if(Number.isNaN(age)) return;
      const max = 220 - age;
      document.getElementById('maxHr').textContent = max.toFixed(0) + ' bpm';
      document.getElementById('zone').textContent = Math.round(max * 0.6) + ' - ' + Math.round(max * 0.75) + ' bpm';
    }
    """
    return shell(title="心率区间计算器", emoji="❤️", color_a="#ff416c", color_b="#ff4b2b", body=body, script=script)


def unit_price_html() -> str:
    body = """
    <p class="sub">比较两个商品谁更划算。</p>
    <input id="price1" type="number" placeholder="商品 A 价格">
    <input id="size1" type="number" placeholder="商品 A 规格">
    <input id="price2" type="number" placeholder="商品 B 价格">
    <input id="size2" type="number" placeholder="商品 B 规格">
    <button onclick="compareUnit()">比较性价比</button>
    <div class="panel" id="display">等待输入</div>
    """
    script = """
    function compareUnit(){
      const price1 = parseFloat(document.getElementById('price1').value);
      const size1 = parseFloat(document.getElementById('size1').value);
      const price2 = parseFloat(document.getElementById('price2').value);
      const size2 = parseFloat(document.getElementById('size2').value);
      if(!price1 || !size1 || !price2 || !size2) return;
      const u1 = price1 / size1;
      const u2 = price2 / size2;
      const better = u1 < u2 ? '商品 A 更划算' : '商品 B 更划算';
      document.getElementById('display').textContent = `${better}｜A 单价 ${u1.toFixed(2)}，B 单价 ${u2.toFixed(2)}`;
    }
    """
    return shell(title="比价器", emoji="🛒", color_a="#614385", color_b="#516395", body=body, script=script, dark=True)


SEEDS = [
    ("做一个倒计时器，可以设定分钟数，有开始暂停和重置按钮", timer_html()),
    ("帮我做个秒表，有开始停止和清零功能", stopwatch_html()),
    ("做一个掷骰子工具，点击按钮随机出1到6的点数", dice_html()),
    ("弄个抛硬币的小工具", coin_html()),
    ("做一个随机选择器，帮我决定今天吃什么", picker_html()),
    ("帮我做一个BMI计算器，输入身高体重就能算", bmi_html()),
    ("做一个摄氏和华氏温度的换算器", dual_converter_html("温度换算", "🌡️", "摄氏 °C", "华氏 °F", "c", "f", "value * 9 / 5 + 32", "(value - 32) * 5 / 9", "#667eea", "#764ba2")),
    ("做个厘米和英寸互转的工具", dual_converter_html("长度换算", "📏", "厘米 cm", "英寸 inch", "cm", "inch", "value / 2.54", "value * 2.54", "#43e97b", "#38f9d7")),
    ("做个公斤和磅的换算", dual_converter_html("重量换算", "⚖️", "公斤 kg", "磅 lb", "kg", "lb", "value * 2.20462", "value / 2.20462", "#f6d365", "#fda085")),
    ("做一个两队记分板，可以加分减分", scoreboard_html()),
    ("做个猜数字游戏，1到100之间猜", guess_html()),
    ("做个简单的计算器，加减乘除", calculator_html()),
    ("做个年龄计算器，输入生日算出年龄", date_diff_html()),
    ("帮我做个百分比计算器", percentage_html()),
    ("做一个倒数日工具，输入日期显示还有几天", day_countdown_html()),
    ("做一个随机密码生成器", password_html()),
    ("做个随机颜色生成器", color_html()),
    ("做一个石头剪刀布游戏", rps_html()),
    ("做一个简单计数器，可以加一减一", counter_html()),
    ("做一个字数统计工具", word_count_html()),
    ("做一个随机数生成器，可以设定范围", random_number_html()),
    ("帮我做一个进制转换器", base_convert_html()),
    ("做一个小费计算器", tip_html()),
    ("做一个AA分账工具，输入总金额和人数就能平摊", split_bill_html()),
    ("做一个矩形面积计算器", area_html()),
    ("做个时间间隔计算器，输入开始和结束时间", time_diff_html()),
    ("做一个双选项投票器", vote_html()),
    ("帮我做个幸运转盘选择器", wheel_html()),
    ("做一个喝水记录器，今天喝了几杯水", water_html()),
    ("做一个待办事项清单，能添加和勾选完成", todo_html()),
    ("做一个番茄钟，25分钟专注5分钟休息", pomodoro_html()),
    ("做一个手机便签本，输入内容后可以保存", notes_html()),
    ("做一个心情记录小工具", mood_html()),
    ("做一个习惯打卡器，统计连续打卡天数", habit_html()),
    ("做一个简单记账本，录入消费金额和项目", expense_html()),
    ("做个节拍器，输入BPM之后按节拍闪动", metronome_html()),
    ("做一个可以涂鸦的画板", drawing_html()),
    ("做一个GPA计算器，输入三门课成绩算绩点", gpa_html()),
    ("做一个英语单词卡片翻翻看", flashcard_html()),
    ("做一个呼吸训练器，提醒吸气呼气节奏", breathing_html()),
    ("做一个跑步配速计算器，输入距离和时间", pace_html()),
    ("做一个睡眠时长计算器，输入入睡和起床时间", sleep_html()),
    ("做一个随机语录生成器", quote_html()),
    ("做一个点击速度测试工具", speed_test_html()),
    ("做一个旅行打包清单", packing_html()),
    ("做一个预算分配器，按收入给出50 30 20预算", budget_html()),
    ("做一个存款目标追踪器，输入目标和已存金额", savings_html()),
    ("做一个卡路里计算器，选食物和分量算热量", calories_html()),
    ("做一个心率区间计算工具，输入年龄算训练区间", heart_rate_html()),
    ("做一个比价器，比较两个商品谁更划算", unit_price_html()),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for instruction, html in SEEDS:
            item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": html},
                ]
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"已生成 {len(SEEDS)} 条种子数据")
    print(f"输出文件: {OUTPUT_PATH}")
    for index, (instruction, html) in enumerate(SEEDS[:5], start=1):
        print(f"[{index:>2}] {instruction} | {len(html)} 字符")


if __name__ == "__main__":
    main()
