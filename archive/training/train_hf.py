# training/train_hf.py
import os, json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
DATA_FILE  = os.environ.get("DATA_FILE", "data/dsp_sqa_master_final.jsonl")
OUT_DIR    = os.environ.get("OUT_DIR", "training/out-qwen15_v3")

print(f"Starting on {BASE_MODEL}")
print(f"Dataset: {DATA_FILE}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("json", data_files=DATA_FILE, split="train")

def format_text(example):
    text = f"### Question:\n{example['instruction']}\n"
    if example.get('input'):
        text += f"\n### Input:\n{example['input']}\n"
    text += f"\n### Answer:\n{example['output']}"
    return {"text": text}

ds = ds.map(format_text)
print(f"Dataset loaded: {len(ds)} examples")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    offload_buffers=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
print("Loaded.")

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)
print("PEFT adapters ready.")

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_ratio=0.05,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_total_limit=1,
    bf16=False,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none",
    save_strategy="epoch",
    max_grad_norm=0.3,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
print("Training starts!")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )

tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    args=training_args,
)


trainer.train()
print(f"QLoRA training done -> {OUT_DIR}")
