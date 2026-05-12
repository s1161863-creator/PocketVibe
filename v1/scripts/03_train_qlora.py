#!/usr/bin/env python3
"""QLoRA fine-tuning for PocketVibe on Qwen2.5-Coder-1.5B-Instruct."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
try:
    from trl import SFTTrainer
except ModuleNotFoundError as exc:
    if exc.name == "trl":
        raise SystemExit(
            "Missing dependency: trl. Install training dependencies with: "
            "pip install -r requirements-train.txt"
        ) from exc
    raise


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "qlora-run2"
LOG_DIR = ROOT / "logs"

BASE_MODEL = os.getenv("PV_BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
HF_HOME = os.getenv("HF_HOME", "/opt/shared/model-cache")
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0")
PV_USE_4BIT = os.getenv("PV_USE_4BIT", "1") == "1"
PV_MAX_SEQ_LENGTH = int(os.getenv("PV_MAX_SEQ_LENGTH", "2048"))
PV_GRAD_ACC_STEPS = int(os.getenv("PV_GRAD_ACC_STEPS", "8"))
PV_USE_GRAD_CHECKPOINTING = os.getenv("PV_USE_GRAD_CHECKPOINTING", "1") == "1"

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_OFFLINE"] = HF_HUB_OFFLINE


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model():
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }

    if PV_USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

    if PV_USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # For LoRA + gradient checkpointing, disable KV cache and ensure input grads are tracked.
    if PV_USE_GRAD_CHECKPOINTING:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model, lora_config


def load_and_format_dataset(tokenizer):
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "val": str(DATA_DIR / "val.jsonl"),
        },
    )

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(format_chat)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 加载分词器")
    tokenizer = build_tokenizer()

    print(">>> 加载量化模型并挂载 LoRA")
    model, lora_config = build_model()

    print(">>> 加载训练数据")
    dataset = load_and_format_dataset(tokenizer)
    print(f"训练集: {len(dataset['train'])} 条 | 验证集: {len(dataset['val'])} 条")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=PV_GRAD_ACC_STEPS,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,
        bf16=True,
        gradient_checkpointing=PV_USE_GRAD_CHECKPOINTING,
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=training_args,
        dataset_text_field="text",
        max_seq_length=PV_MAX_SEQ_LENGTH,
        packing=False,
    )

    print(">>> 开始训练")
    print(f"4bit QLoRA: {'ON' if PV_USE_4BIT else 'OFF (LoRA bf16)'}")
    print(f"基座模型: {BASE_MODEL}")
    print(f"max_seq_length: {PV_MAX_SEQ_LENGTH}")
    print(f"gradient_accumulation_steps: {PV_GRAD_ACC_STEPS}")
    print(f"gradient_checkpointing: {PV_USE_GRAD_CHECKPOINTING}")
    print(f"LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    print(
        "有效 batch size:",
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    )
    trainer.train()

    final_dir = OUTPUT_DIR / "final_adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"适配器已保存: {final_dir}")

    log_path = LOG_DIR / "train_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(trainer.state.log_history, fh, ensure_ascii=False, indent=2)
    print(f"训练日志已保存: {log_path}")


if __name__ == "__main__":
    main()
