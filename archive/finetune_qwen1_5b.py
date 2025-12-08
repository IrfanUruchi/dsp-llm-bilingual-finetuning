#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
DATA_PATH  = os.environ.get("DATA_PATH", "data/train_ready/fft_bin_medium_instruct.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/qwen2_5-1_5b_dsp_v5_lowmem")
EPOCHS     = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1")) 
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "16"))  
LR         = float(os.environ.get("LR", "2e-4"))
MAX_LEN    = int(os.environ.get("MAX_LEN", "512"))   
SEED       = int(os.environ.get("SEED", "2025"))

print(f"Base model : {MODEL_NAME}")
print(f"Data path  : {DATA_PATH}")
print(f"Output dir : {OUTPUT_DIR}")
set_seed(SEED)

ds = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(ex):
    text = f"### Instruction:\n{ex['instruction']}\n"
    if ex.get("input"):
        text += f"\n### Input:\n{ex['input']}\n"
    text += f"\n### Response:\n{ex['output']}"
    return {"text": text}

ds = ds.map(format_example)
print(f"Loaded {len(ds)} formatted examples")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_fn(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out

ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading 4-bit base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print("Model loaded.")

model.gradient_checkpointing_enable()
model.config.use_cache = False  
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=8,                 
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
print("PEFT adapters ready.")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    fp16=True,
    optim="paged_adamw_8bit", 
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    report_to="none",
    remove_unused_columns=False,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=ds_tok,
)

print("Training starts!")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f" Model saved at {OUTPUT_DIR}")
