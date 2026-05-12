#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PocketVibe — Gradio Web Demo (纯后端, 用于 Demo 25% 得分)
=================================================================
功能:
  1. 输入自然语言描述 → 调用 V2 LoRA 适配器 → 返回完整 HTML
  2. 右侧 iframe 实时渲染 → 手机浏览器可扫码访问
  3. 内置 V1/V2 切换开关, 便于现场演示两代对比

部署 (HPC):
  cd ~/PocketVibe
  source ~/venvs/pv-train/bin/activate
  pip install gradio
  sbatch slurm/serve_app.slurm          # 见底部 SLURM 模板

本地测试 (CPU 慢但可跑通):
  python app/app.py --device cpu --share

依赖:
  pip install gradio transformers peft torch
"""
import os, argparse, html
from pathlib import Path
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =================================================================
# 路径 & 默认配置
# =================================================================
BASE_MODEL  = os.environ.get("PV_BASE_MODEL",
                             "Qwen/Qwen2.5-Coder-1.5B-Instruct")
V1_ADAPTER  = os.environ.get("PV_V1_ADAPTER",
                             os.path.expanduser("~/PocketVibe/outputs/qlora-run1/final_adapter"))
V2_ADAPTER  = os.environ.get("PV_V2_ADAPTER",
                             os.path.expanduser("~/PocketVibe/outputs/qlora-v2-run1/final_adapter"))

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的 HTML 文件。"
    "要求：所有 CSS 用 <style> 标签内联在 <head> 中，所有 JavaScript 用 <script> 标签内联在 <body> 末尾。"
    "界面必须适配手机屏幕（使用 viewport meta 标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用 Markdown，只输出纯 HTML 代码。"
)

EXAMPLES = [
    "做一个分段秒表，支持开始/暂停/清零，最多 10 段，最快段绿色最慢段红色高亮",
    "做一个番茄钟，25 分钟工作 5 分钟休息，轮次自动切换",
    "做一个 BMI 计算器，输入身高体重显示指数和健康等级",
    "做一个掷骰子工具，点击按钮随机显示 1-6 点",
    "做一个石头剪刀布游戏，带实时计分板和最近 5 局历史",
]

# =================================================================
# 模型加载 (懒加载 + 缓存)
# =================================================================
_cache = {"tokenizer": None, "base": None, "v1": None, "v2": None, "device": None}

def load_models(device: str = "cuda"):
    """首次调用时加载, 之后复用"""
    if _cache["tokenizer"] is not None:
        return
    print(f">>> 加载分词器: {BASE_MODEL}")
    _cache["tokenizer"] = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f">>> 加载基座模型 (device={device}, dtype={dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        base = base.to("cpu")
    _cache["base"] = base
    _cache["device"] = device

    # V2 优先加载（主力模型）
    if Path(V2_ADAPTER).exists():
        print(f">>> 挂载 V2 LoRA: {V2_ADAPTER}")
        _cache["v2"] = PeftModel.from_pretrained(base, V2_ADAPTER, adapter_name="v2")
    else:
        print(f"!!! V2 adapter 未找到: {V2_ADAPTER}")

    # V1 可选（对比用）
    if Path(V1_ADAPTER).exists() and _cache["v2"] is not None:
        print(f">>> 追加 V1 LoRA: {V1_ADAPTER}")
        _cache["v2"].load_adapter(V1_ADAPTER, adapter_name="v1")
        _cache["v1"] = _cache["v2"]   # 共用 PEFT 对象, 通过 set_adapter 切换

def generate(instruction: str, version: str, temperature: float, max_new_tokens: int) -> str:
    """返回纯 HTML 字符串"""
    if not instruction or not instruction.strip():
        return "<!-- 请输入需求 -->"
    load_models(_cache.get("device") or "cuda")

    tok = _cache["tokenizer"]
    model = _cache["v2"]
    if model is None:
        return "<!-- 模型未加载, 请检查 adapter 路径 -->"

    # 切换 adapter
    try:
        model.set_adapter("v2" if version == "V2 (推荐)" else "v1")
    except Exception as e:
        print(f"[warn] set_adapter 失败, 使用默认: {e}")

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": instruction.strip()},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.0,
            pad_token_id=tok.eos_token_id,
        )
    code = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 清洗可能的 markdown 代码块标记
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[-1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()

def run_pipeline(instruction, version, temperature, max_new_tokens):
    """Gradio 回调: 返回 (html 源码, 预览 iframe html)"""
    code = generate(instruction, version, temperature, int(max_new_tokens))
    # 用 srcdoc 嵌入 iframe 安全渲染
    preview = (
        f'<iframe srcdoc="{html.escape(code, quote=True)}" '
        f'style="width:100%;height:640px;border:1px solid #ddd;border-radius:12px;background:#fff"></iframe>'
    )
    return code, preview

# =================================================================
# Gradio UI
# =================================================================
def build_ui():
    with gr.Blocks(title="PocketVibe Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🪄 PocketVibe — 中文自然语言 → 移动端 HTML 微应用\n"
            "**INT6138 Project II 部署 Demo**  |  基座 `Qwen2.5-Coder-1.5B-Instruct` + LoRA 微调  |  "
            "输入一句话描述 → 右侧即时预览"
        )
        with gr.Row():
            with gr.Column(scale=1):
                inst = gr.Textbox(
                    label="需求描述 (中文自然语言)",
                    placeholder="例如: 做一个分段秒表, 最多10段, 最快段绿色最慢段红色",
                    lines=3,
                )
                version = gr.Radio(
                    choices=["V2 (推荐)", "V1 (对比)"],
                    value="V2 (推荐)",
                    label="模型版本",
                )
                with gr.Accordion("高级参数", open=False):
                    temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature")
                    max_tokens  = gr.Slider(512, 4096, value=3072, step=256, label="最大生成长度")
                btn = gr.Button("🚀 生成 HTML", variant="primary")
                gr.Examples(
                    examples=[[e] for e in EXAMPLES],
                    inputs=[inst],
                    label="示例 (点击一键填充)",
                )
            with gr.Column(scale=2):
                preview = gr.HTML(label="📱 实时预览 (iframe 沙盒渲染)")
                code_box = gr.Code(language="html", label="生成的 HTML 源码", lines=18)

        btn.click(
            run_pipeline,
            inputs=[inst, version, temperature, max_tokens],
            outputs=[code_box, preview],
        )

        gr.Markdown(
            "---\n"
            "**部署信息**:  `V2 adapter` = 830 条高信息熵数据 · NEFTune · CompletionOnlyLoss · eval_loss=0.1094  \n"
            "**硬件**: EdUHK AAILLM HPC · NVIDIA A16 16GB · bf16 LoRA  \n"
            "**数据**: Evol-Instruct 四方向 + 一指令×4 风格 + 跨类组合 (详见 README §5)"
        )
    return demo

# =================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--host",   default="0.0.0.0")
    ap.add_argument("--port",   type=int, default=7860)
    ap.add_argument("--share",  action="store_true", help="生成 gradio.live 公网链接 (本地演示用)")
    args = ap.parse_args()

    _cache["device"] = args.device
    load_models(args.device)                      # 预加载
    ui = build_ui()
    ui.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
