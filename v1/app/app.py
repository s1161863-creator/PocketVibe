#!/usr/bin/env python3
"""
PocketVibe — Gradio 前端 v3
功能: 输入自然语言指令 → 生成手机 HTML 微应用 → iPhone 模拟器实时预览
改进: iframe sandbox 修复 / 历史记录 / 重新生成 / 优化示例
运行: python app/app.py
依赖: pip install gradio requests
"""
import gradio as gr
import requests
import json
import os
import time

# ── 后端 API 地址 ──
API_URL = os.environ.get("PV_API_URL", "http://localhost:8000/generate")

# ── 精选示例（从种子数据中挑选的高质量演示指令）──
EXAMPLES = [
    ["做一个倒计时器，可以设定分钟数，有开始暂停和重置按钮"],
    ["做一个番茄钟，25分钟工作5分钟休息，自动轮次切换"],
    ["做一个简单计算器，加减乘除"],
    ["做一个两队记分板，可以加分减分"],
    ["做一个石头剪刀布游戏"],
    ["做一个待办事项列表，可以添加勾选和删除"],
    ["做一个随机密码生成器，可以调节长度"],
    ["帮我做一个BMI计算器，输入身高体重就能算"],
    ["做一个掷骰子工具，点击按钮随机出1到6的点数"],
    ["做一个猜数字游戏，1到100之间猜"],
]

# ── 历史记录（内存中，会话期间保持）──
history_store = []


def _wrap_in_phone(html_code: str) -> str:
    """把 HTML 代码放进 iPhone 模拟器 iframe"""
    safe = html_code.replace("&", "&amp;").replace('"', "&quot;")
    return f'''
<div style="display:flex;justify-content:center;padding:20px 10px">
  <div style="
    width:375px; height:720px;
    border-radius:44px;
    border:6px solid #1a1a2e;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35), inset 0 0 0 2px #444;
    overflow:hidden; background:#000; position:relative;
  ">
    <div style="
      position:absolute;top:0;left:50%;transform:translateX(-50%);
      width:126px;height:28px;background:#1a1a2e;
      border-radius:0 0 18px 18px;z-index:10;
    "></div>
    <div style="
      position:absolute;bottom:6px;left:50%;transform:translateX(-50%);
      width:120px;height:4px;background:#666;border-radius:2px;z-index:10;
    "></div>
    <iframe
      srcdoc="{safe}"
      style="width:100%;height:100%;border:none;background:#fff;"
      sandbox="allow-scripts allow-same-origin allow-forms allow-modals allow-popups"
      loading="lazy"
    ></iframe>
  </div>
</div>
'''


def _history_html() -> str:
    """生成历史记录面板的 HTML"""
    if not history_store:
        return "<p style='color:#aaa;text-align:center;padding:30px'>暂无历史记录</p>"
    items = []
    for i, h in enumerate(reversed(history_store)):
        idx = len(history_store) - 1 - i
        t = h.get("time", "")
        inst = h.get("instruction", "")[:50]
        chars = h.get("chars", 0)
        items.append(
            f'<div style="padding:10px 12px;margin:4px 0;background:#f8f8ff;'
            f'border-radius:10px;border-left:4px solid #667eea;cursor:default">'
            f'<div style="font-size:14px;font-weight:600;color:#333">#{idx+1} {inst}</div>'
            f'<div style="font-size:12px;color:#888;margin-top:4px">{t} | {chars:,} 字符</div>'
            f'</div>'
        )
    return "".join(items)


def generate_app(instruction: str) -> tuple:
    """调用后端 API 生成 HTML"""
    if not instruction.strip():
        return (
            "<p style='color:gray;text-align:center;padding:60px'>📝 请输入需求描述</p>",
            "",
            "⚠️ 请输入指令",
            _history_html(),
        )

    try:
        resp = requests.post(
            API_URL,
            json={"instruction": instruction, "max_tokens": 4000},
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        html_code = data.get("html", "")
        char_count = data.get("char_count", len(html_code))

        if not html_code:
            return (
                "<p style='color:red;text-align:center;padding:40px'>生成失败，返回为空</p>",
                "", "❌ 生成失败", _history_html(),
            )

        # 保存到历史记录
        history_store.append({
            "instruction": instruction,
            "html": html_code,
            "chars": char_count,
            "time": time.strftime("%H:%M:%S"),
        })

        phone = _wrap_in_phone(html_code)
        status = f"✅ 生成成功 | {char_count:,} 字符 | 历史 #{len(history_store)}"
        return phone, html_code, status, _history_html()

    except requests.exceptions.ConnectionError:
        msg = f"❌ 无法连接 API ({API_URL})\n请确认后端已启动"
        return f"<pre style='color:red;padding:20px'>{msg}</pre>", "", msg, _history_html()
    except requests.exceptions.Timeout:
        return (
            "<p style='color:orange;text-align:center;padding:40px'>⏱️ 生成超时（>180s）</p>",
            "", "⏱️ 超时，请重试", _history_html(),
        )
    except Exception as e:
        return f"<p style='color:red'>错误: {e}</p>", "", f"❌ {e}", _history_html()


def load_history_item(evt: gr.SelectData) -> tuple:
    """从历史记录中加载一条"""
    idx = len(history_store) - 1 - evt.index
    if 0 <= idx < len(history_store):
        h = history_store[idx]
        return h["instruction"], _wrap_in_phone(h["html"]), h["html"]
    return "", "", ""


# ── Gradio 界面 ──
with gr.Blocks(
    title="PocketVibe — 手机微应用生成器",
    theme=gr.themes.Soft(primary_hue="violet"),
    css="""
        .phone-col { min-width: 420px; }
        #status-box textarea { font-size: 14px !important; }
        .example-btn { font-size: 13px !important; }
        .gr-button-primary { 
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            font-size: 18px !important; padding: 14px !important;
        }
    """,
) as demo:

    gr.Markdown("""
# 📱 PocketVibe — 手机微应用生成器
**输入自然语言描述，一键生成可运行的手机 HTML 微应用**
> Qwen2.5-Coder-1.5B + QLoRA 微调 | LoRA r=32 α=64 | ~550 条训练数据
    """)

    with gr.Row():
        # ── 左栏：输入 + 控制 ──
        with gr.Column(scale=2):
            instruction_box = gr.Textbox(
                label="📝 需求描述",
                placeholder="例如：做一个倒计时器，可以设定分钟数，有开始暂停和重置按钮",
                lines=3, max_lines=6,
            )

            with gr.Row():
                generate_btn = gr.Button("🚀 生成微应用", variant="primary", scale=3)
                regen_btn = gr.Button("🔄 重新生成", variant="secondary", scale=1)

            status_box = gr.Textbox(
                label="状态", interactive=False,
                elem_id="status-box", lines=1,
            )

            with gr.Accordion("📋 示例指令（点击填入）", open=False):
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=instruction_box,
                    label="",
                    examples_per_page=10,
                )

            with gr.Accordion("📜 历史记录", open=False):
                history_panel = gr.HTML(
                    value="<p style='color:#aaa;text-align:center;padding:20px'>暂无记录</p>",
                    label="生成历史",
                )

        # ── 右栏：预览 + 代码 ──
        with gr.Column(scale=2, elem_classes=["phone-col"]):
            with gr.Tabs():
                with gr.Tab("📱 预览"):
                    preview = gr.HTML(
                        value="""<div style='text-align:center;color:#aaa;padding:80px 20px'>
                            <div style='font-size:48px;margin-bottom:16px'>📱</div>
                            <div style='font-size:16px'>输入需求描述，点击生成</div>
                            <div style='font-size:13px;margin-top:8px;color:#ccc'>
                                推荐先试：番茄钟 / 计算器 / 待办清单
                            </div>
                        </div>""",
                    )
                with gr.Tab("💻 源码"):
                    code_box = gr.Code(
                        label="HTML 代码", language="html",
                        lines=30, interactive=False,
                    )

    # ── 事件绑定 ──
    gen_outputs = [preview, code_box, status_box, history_panel]

    generate_btn.click(
        fn=generate_app, inputs=instruction_box, outputs=gen_outputs,
    )
    regen_btn.click(
        fn=generate_app, inputs=instruction_box, outputs=gen_outputs,
    )
    instruction_box.submit(
        fn=generate_app, inputs=instruction_box, outputs=gen_outputs,
    )

    gr.Markdown("""
---
**使用提示：** 指令越具体，生成质量越高。例如「做一个倒计时器，可以设定分钟数，有开始暂停和重置按钮」比「做个倒计时」效果好得多。
    """)


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    print(f">>> PocketVibe Gradio v3 启动")
    print(f">>> 后端 API: {API_URL}")
    print(f">>> 端口: {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
