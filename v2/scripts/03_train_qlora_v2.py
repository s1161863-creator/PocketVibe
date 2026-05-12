#!/usr/bin/env python3
"""
=================================================================
PocketVibe V2 — QLoRA 微调 Qwen2.5-Coder-1.5B-Instruct
（第二轮训练，系统性修复 V1 六个问题）
=================================================================
核心改进（对应 README.md 第 3 节）：
  1. LoRA target 扩展到 7 层（含 MLP gate/up/down_proj）
  2. rank 从 32 降到 16（~800 条数据更合适）
  3. NEFTune noise_alpha=5（零成本泛化提升，ICLR 2024）
  4. CompletionOnly Loss（只让 assistant HTML 代码贡献梯度）
  5. EarlyStoppingCallback patience=2（防止 late-stage 过拟合）
  6. eval_steps=50（更频繁评估，更及时感知过拟合）
  7. 验证集为真正 held-out 类别（02_category_split.py 保证）

运行方式: sbatch slurm/train.slurm
=================================================================
"""
import os, json, torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ======================= 环境变量配置 =======================
# HPC 上的模型缓存路径
os.environ["HF_HOME"]         = os.environ.get("HF_HOME", "/opt/shared/model-cache")
# online-first：缓存没命中时自动拉取（V1 教训：离线模式 + 缓存缺配置 = 启动失败）
os.environ["HF_HUB_OFFLINE"]  = os.environ.get("HF_HUB_OFFLINE", "0")

# 允许通过环境变量覆盖关键参数（便于 SLURM 脚本传参）
USE_4BIT     = os.environ.get("PV_USE_4BIT", "0") == "1"   # A16 bitsandbytes 后端缺失，默认 bf16
# ★ V2 重做：max_seq_length 从 1024 提升到 4096（留安全余量）
#   背景：V2 首轮训练配置了 max_seq_length=1024，但实测 train.jsonl 100% 样本
#         > 1024 tokens（平均 1816, 最大 4335 估算值），整个训练集被截尾，
#         模型从未见过完整的 </html> 闭合 → 必须重做训练。
#   为什么选 4096 而不是 3072：
#     - 字符→token 估算系数 2.5 对 HTML/JS 代码偏乐观（BPE 对英文标签压缩率低）
#     - 实际最长样本 token 数可能到 5000+，3072 会截掉 ~20% 样本
#     - 4096 是 2 的整数幂，对齐 Flash Attention，速度最快
#     - 配合 02c_filter_by_length.py 预过滤超长样本，可保证训练 0 截断
MAX_SEQ_LEN  = int(os.environ.get("PV_MAX_SEQ_LENGTH", "4096"))
# ★ V2 重做新增：梯度累积步数从 16 提升到 32，
#   保持有效 batch 的同时为 4x 序列长度腾出显存余量
GRAD_ACC     = int(os.environ.get("PV_GRAD_ACC_STEPS", "32"))
# ★ V2 重做新增：gradient checkpointing 开关，4096 序列下必开
USE_GRAD_CKPT = os.environ.get("PV_USE_GRAD_CHECKPOINTING", "1") == "1"
# 输出目录 tag：V2 重做直接覆盖 V2 首轮产物目录（qlora-v2-run1），
# 旧产物已在 HPC 侧手动备份为 qlora-v2-run1_old_1024trunc_*
OUTPUT_TAG   = os.environ.get("PV_RUN_TAG", "qlora-v2-run1")

# ======================= 路径配置 =======================
BASE_MODEL  = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
HOME        = os.path.expanduser("~")
# 复用 V1 目录（~/PocketVibe）的 venv 和模型缓存，避免重装依赖
PROJECT_DIR = os.environ.get("PV_PROJECT_DIR", os.path.join(HOME, "PocketVibe"))
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "outputs", OUTPUT_TAG)
LOG_DIR     = os.path.join(PROJECT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# =====================================================
# 1. 量化配置（可选，默认关闭）
# =====================================================
# V1 教训：bitsandbytes 在 HPC 缺少 GPU 后端，用 bf16 LoRA 反而更稳
# 若确认 bitsandbytes 可用则设 PV_USE_4BIT=1
if USE_4BIT:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",              # NF4 优于 INT4（信息论最优量化）
        bnb_4bit_compute_dtype=torch.bfloat16,  # 推理时反量化为 bf16 计算
        bnb_4bit_use_double_quant=True,          # 双重量化，再省 ~0.4 GB 显存
    )
    model_kwargs = {"quantization_config": bnb_config}
    print(">>> 使用 4-bit NF4 量化加载")
else:
    model_kwargs = {"torch_dtype": torch.bfloat16}
    print(">>> 使用 bf16 精度加载（不量化）")

# =====================================================
# 2. 加载分词器
# =====================================================
print(f">>> 加载分词器：{BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # SFTTrainer 要求右 padding

# =====================================================
# 3. 加载模型
# =====================================================
print(f">>> 加载模型（USE_4BIT={USE_4BIT}）...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    **model_kwargs,
)

if USE_4BIT:
    model = prepare_model_for_kbit_training(model)  # 冻结原始权重，启用梯度检查点
else:
    model.enable_input_require_grads()

# ★ V2 重做新增：gradient checkpointing 兼容性配置
#   4096 序列长度下必须启用 checkpointing，否则 A16 16GB 必 OOM
#   LoRA + checkpointing 组合要求：关闭 KV cache + 强制输入需要梯度
if USE_GRAD_CKPT:
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

# =====================================================
# 4. LoRA 配置（V2 核心改进：扩展到 MLP 层）
# =====================================================
# 报告得分点：解释每个改动的理由
#
# V1 问题：只挂 Attention 4 层，遗漏了 MLP
# V2 修复：挂满所有 7 个 linear 层
#   - q/k/v/o_proj：Attention 自注意力权重
#   - gate_proj/up_proj/down_proj：FFN（MLP）层
#     → 负责"语义→代码结构"的映射，对代码生成更关键
#
# rank 从 32 降到 16：
#   - 数据量 ~800 条，r=32 的 LoRA 参数容量远超训练信号
#   - r=16 在 1~2K 数据量下经验最优（Databricks 2025 指南）
#
# dropout 从 0.05 提高到 0.1：小数据集需要更强正则
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,          # scale = alpha/r = 2.0（与 V1 保持一致）
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 层（V1 已有）
        "gate_proj", "up_proj", "down_proj",      # ★ MLP 层（V2 新增）
    ],
    lora_dropout=0.1,       # V1=0.05 → V2=0.1（更强正则）
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 预期输出示例：
#   trainable params: ~12M || all params: ~1.5B || trainable%: ~0.8%
# （7 层比 V1 的 4 层多约 50% LoRA 参数，但仍远低于全量微调）

# =====================================================
# 5. 加载数据集
# =====================================================
print(">>> 加载训练数据...")
# ★ V2 重做新增：如果存在 02c_filter_by_length.py 产出的过滤文件，优先使用
#   该文件已剔除 token 数超过 MAX_SEQ_LEN 的样本，保证训练 0 截断
train_filtered = os.path.join(DATA_DIR, "train_filtered.jsonl")
train_default  = os.path.join(DATA_DIR, "train.jsonl")
if os.path.exists(train_filtered):
    train_file = train_filtered
    print(f"   ✓ 使用过滤后的训练集：train_filtered.jsonl（0 截断保证）")
else:
    train_file = train_default
    print(f"   ⚠ 未找到 train_filtered.jsonl，使用原始 train.jsonl")
    print(f"     建议先运行 scripts/02c_filter_by_length.py 预过滤超长样本")

dataset = load_dataset(
    "json",
    data_files={
        "train": train_file,
        "val":   os.path.join(DATA_DIR, "val.jsonl"),
    },
)
print(f"   训练集：{len(dataset['train'])} 条")
print(f"   验证集：{len(dataset['val'])} 条（held-out 类别，真实泛化评估）")

# =====================================================
# 6. 格式化 + 预 tokenize（V2 重做修复 DataCollator bug）
# =====================================================
# 背景：trl 0.13+ 里 SFTConfig(dataset_text_field=...) 与
# DataCollatorForCompletionOnlyLM 不兼容——trl 不会帮你 tokenize，
# collator 拿到的就是原始字符串列表，报错
# "too many dimensions 'str'"。
# 正确做法：我们在 map 阶段就把 messages 直接 tokenize 成 input_ids，
# 然后不再用 dataset_text_field，SFTTrainer 拿到已 tokenize 的数据就直接走。
_ORIG_COLUMNS = dataset["train"].column_names  # ['messages', '_category', ...]

def tokenize_fn(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    # 训练时做硬截断到 MAX_SEQ_LEN（尽管 02c 已过滤，双保险）
    out = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,           # DataCollator 负责动态 padding
        add_special_tokens=False, # chat_template 已经包好特殊 token
    )
    return out

# 只保留 input_ids / attention_mask 两列，其余 metadata 全部移除
dataset = dataset.map(
    tokenize_fn,
    remove_columns=_ORIG_COLUMNS,
    desc="Tokenizing",
)

# =====================================================
# 7. CompletionOnly Loss（V2 新增）
# =====================================================
# 原理：只让 assistant 输出的 HTML 代码部分贡献 loss，
#       system prompt 和 user 指令的 token 被 mask 为 -100
#       → 梯度不浪费在背诵固定系统提示上
#
# Qwen2.5 的 assistant 开始标记：<|im_start|>assistant\n
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"

data_collator = DataCollatorForCompletionOnlyLM(
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tokenizer,
)

# =====================================================
# 8. 训练参数（V2 改动说明见注释）
# =====================================================
# 注意：trl 0.13+ 要求把 SFT 专用参数（max_seq_length / dataset_text_field /
#       neftune_noise_alpha）从 SFTTrainer() 移到 SFTConfig 里。
#       SFTConfig 是 TrainingArguments 的子类，兼容所有 TrainingArguments 参数。
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # 轮次（V2 重做）：2 → 3（配合 early stopping + 低 LR）
    # V1（旧）=3 轮 + LR=2e-4，早期深度过拟合
    # V2 首轮 =2 轮 + LR=1e-4（但被 1024 截断 bug 毁掉）
    # V2 重做 =3 轮 + LR=5e-5：低 LR 收敛更慢，多给 1 轮缓冲；
    #   若 eval_loss 在 2 轮就反弹，EarlyStopping(patience=2) 自动兜底
    num_train_epochs=3,

    # Batch size：物理 1，梯度累积 32 → 有效 batch=32（V2 重做：从 16 提到 32）
    # 原因：序列长度从 1024→4096（4x）导致单步显存占用增加，
    #       累积步数加倍可以在保持收敛稳定性的同时降低峰值显存
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACC,
    per_device_eval_batch_size=1,

    # 学习率（V2 重做）：1e-4 → 5e-5
    # 原因（对应报告"灾难性遗忘风险"讨论）：
    #   1. 基座是 Qwen2.5-Coder-1.5B-Instruct，已做完整 SFT+DPO
    #      → 高 LR 二次 SFT 容易覆盖已有通用代码能力（灾难性遗忘）
    #   2. LoRA 7 层 + r=16，可训练参数 ~18M，数据仅 ~800 条
    #      → 参数/样本比 22500:1，低 LR 更安全
    #   3. Alpaca-LoRA / DeepSeek-Coder-Instruct 等主流 Instruct 二次 SFT
    #      配方均在 2e-5 ~ 5e-5 区间
    #   4. V2 已叠加 NEFTune α=5 / dropout=0.1 / EarlyStopping，正则充足
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    # 评估策略：改为 steps（每 50 步评估一次）
    # V1 = epoch（3 个 epoch 只评估 3 次，感知过拟合太滞后）
    # V2 = steps，更频繁，配合 early stopping 更及时
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,

    # 加载验证集 loss 最低的 checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # 精度
    fp16=False,
    bf16=True,              # A16 Ampere 架构原生支持 bf16

    # ★ V2 重做新增：gradient checkpointing（4096 序列下必开）
    gradient_checkpointing=USE_GRAD_CKPT,

    # 其他
    max_grad_norm=1.0,      # 梯度裁剪，防止 MLP 层新增参数引起的梯度爆炸
    logging_steps=10,
    report_to="none",       # 不上传 wandb
    dataloader_num_workers=4,
    remove_unused_columns=False,  # 保留 _category 等 metadata 字段

    # ====== SFT 专用参数（trl 0.13+ 必须放在 SFTConfig 里） ======
    # 注意：我们已经手动预 tokenize 了 dataset（见第 6 节），
    # 所以不设 dataset_text_field，trl 会直接用 input_ids。
    max_seq_length=MAX_SEQ_LEN,
    # ★ NEFTune（Noisy Embedding Fine-Tuning, ICLR 2024, arXiv:2310.05914）
    # 在 embedding 层添加均匀分布随机噪声（α=5），零计算开销提升泛化
    neftune_noise_alpha=5,
    packing=False,
)

# =====================================================
# 9. 创建 SFTTrainer（V2 新增 NEFTune + CompletionOnly）
# =====================================================
# 注：trl 0.13+ 把 max_seq_length / dataset_text_field / neftune_noise_alpha
#     从 SFTTrainer 的入参挪到了 SFTConfig 里（见上方 training_args）。
trainer = SFTTrainer(
    model=model,
    # transformers 4.46+ 把 Trainer 的 `tokenizer=` 改名为 `processing_class=`
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    args=training_args,
    data_collator=data_collator,   # ★ V2 新增：CompletionOnly Loss
    callbacks=[
        # ★ V2 新增：Early Stopping
        # patience=2：验证集 loss 连续 2 次评估不降则停止
        # 防止 V1 的 late-stage memorization 问题
        EarlyStoppingCallback(early_stopping_patience=2),
    ],
)

# =====================================================
# 10. 打印训练配置摘要
# =====================================================
print(f"\n{'='*60}")
print(f"🚀 PocketVibe V2 训练配置摘要（重做，修复 1024 截断 bug）")
print(f"{'='*60}")
print(f"   基座模型：{BASE_MODEL}")
print(f"   量化方式：{'4-bit NF4' if USE_4BIT else 'bf16（不量化）'}")
print(f"   LoRA rank：{lora_config.r}  alpha：{lora_config.lora_alpha}")
print(f"   LoRA target：{lora_config.target_modules}")
print(f"   LoRA dropout：{lora_config.lora_dropout}")
print(f"   有效 batch size：{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   学习率：{training_args.learning_rate}  调度：{training_args.lr_scheduler_type}  ★ V2 重做：从 1e-4 降到 5e-5（防灾难性遗忘）")
print(f"   最大 epoch：{training_args.num_train_epochs}  + Early Stopping(patience=2)  ★ V2 重做：从 2 提到 3")
print(f"   max_seq_length：{MAX_SEQ_LEN}  ★ V2 重做：从 1024 提升（首轮截断率 100%）")
print(f"   gradient_accumulation_steps：{GRAD_ACC}  ★ V2 重做：从 16 提升")
print(f"   gradient_checkpointing：{USE_GRAD_CKPT}  ★ V2 重做：必开")
print(f"   NEFTune alpha：5")
print(f"   CompletionOnly Loss：启用")
print(f"   输出目录：{OUTPUT_DIR}")
print(f"{'='*60}\n")

# =====================================================
# 11. 开始训练
# =====================================================
trainer.train()

# =====================================================
# 12. 保存最终适配器
# =====================================================
FINAL_DIR = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"\n✅ 训练完成！LoRA 适配器已保存到：{FINAL_DIR}")

# 保存训练日志（用于 07_plot_loss.py 画图）
# 文件名随 RUN_TAG 走，避免覆盖 V2 旧日志
log_path = os.path.join(LOG_DIR, f"train_log_{OUTPUT_TAG}.json")
with open(log_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"📊 训练日志：{log_path}")

# 打印最终 loss
log_history = trainer.state.log_history
train_losses = [x["loss"]      for x in log_history if "loss"      in x]
eval_losses  = [x["eval_loss"] for x in log_history if "eval_loss" in x]
if train_losses:
    print(f"   最终 train_loss：{train_losses[-1]:.4f}")
if eval_losses:
    print(f"   最佳 eval_loss： {min(eval_losses):.4f}")

print(f"\n⏭️  下一步：python scripts/04_inference_test.py")
