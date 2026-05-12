#!/usr/bin/env python3
"""
PocketVibe V2 — 06: FastAPI 推理服务
=================================================================
把微调后的模型封装成 HTTP API，供 Gradio 前端调用
端口：8000（默认）

接口：
  POST /generate   → 生成 HTML 代码
  GET  /health     → 健康检查
  GET  /info       → 模型信息

运行：sbatch slurm/serve.slurm
本地测试：
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/generate \
       -H "Content-Type: application/json" \
       -d '{"instruction":"做一个倒计时器"}'
=================================================================
"""
import os, json, time, torch, logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 环境变量 ──
os.environ["HF_HOME"]        = os.environ.get("HF_HOME", "/opt/shared/model-cache")
os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")

HOME        = os.path.expanduser("~")
BASE_MODEL  = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_DIR = os.path.join(HOME, "PocketVibe_v2", "outputs", "qlora-v2-run1", "final_adapter")
PORT        = int(os.environ.get("PV_PORT", "8000"))

SYSTEM_PROMPT = (
    "你是一个移动端微应用生成器。用户会用自然语言描述一个小工具的需求，"
    "你需要直接输出一个完整的、可独立运行的HTML文件。"
    "要求：所有CSS用<style>标签内联在<head>中，所有JavaScript用<script>标签内联在<body>末尾。"
    "界面必须适配手机屏幕（使用viewport meta标签和响应式设计），风格现代简洁，"
    "使用圆角、阴影、渐变配色。不要输出任何解释文字，不要使用Markdown，只输出纯HTML代码。"
)

# ── 全局模型对象（启动时加载一次）──
tokenizer = None
model     = None
_load_time = None


def load_model():
    global tokenizer, model, _load_time
    t0 = time.time()

    logger.info(f"加载分词器：{BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("加载基础模型（bf16）...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"加载 LoRA 适配器：{ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    _load_time = round(time.time() - t0, 1)
    logger.info(f"✅ 模型加载完成，耗时 {_load_time}s")


# ── FastAPI 应用 ──
app = FastAPI(
    title="PocketVibe V2 API",
    description="Qwen2.5-Coder-1.5B + QLoRA → 移动端 HTML 生成",
    version="2.0.0",
)

# 允许跨域（Gradio 前端从不同端口调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ──
class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=2, max_length=500, description="自然语言工具描述")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens:  int   = Field(default=2048, ge=64, le=4096)
    top_p:       float = Field(default=0.9, ge=0.1, le=1.0)

class GenerateResponse(BaseModel):
    html:         str
    instruction:  str
    length:       int
    elapsed_ms:   int
    model_version: str = "v2"

class HealthResponse(BaseModel):
    status:      str
    model_ready: bool
    load_time_s: Optional[float] = None

class InfoResponse(BaseModel):
    base_model:  str
    adapter_dir: str
    system_prompt_len: int
    port:        int


# ── 路由 ──
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_ready=model is not None,
        load_time_s=_load_time,
    )


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        base_model=BASE_MODEL,
        adapter_dir=ADAPTER_DIR,
        system_prompt_len=len(SYSTEM_PROMPT),
        port=PORT,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型尚未加载，请稍后重试")

    t0 = time.time()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": req.instruction},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            repetition_penalty=1.1,
        )

    html = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # 清洗可能的 markdown 代码块包裹
    import re
    if html.startswith("```"):
        html = re.sub(r'^```[a-zA-Z]*\n?', '', html)
    if html.endswith("```"):
        html = html.rsplit("```", 1)[0]
    html = html.strip()

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"生成完成 | 指令: {req.instruction[:30]} | 长度: {len(html)} | 耗时: {elapsed_ms}ms")

    return GenerateResponse(
        html=html,
        instruction=req.instruction,
        length=len(html),
        elapsed_ms=elapsed_ms,
    )


# ── 启动事件：加载模型 ──
@app.on_event("startup")
def startup_event():
    load_model()


if __name__ == "__main__":
    uvicorn.run(
        "06_serve_api:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        reload=False,
    )
