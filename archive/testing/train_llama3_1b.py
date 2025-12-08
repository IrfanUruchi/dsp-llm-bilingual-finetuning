import os
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

model_name = "./Llama-3.2-1B-Instruct"

print(f"Loading model from: {model_name}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0},
    quantization_config=bnb_config,
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


lora_config = LoraConfig(
    r=8,               
    lora_alpha=16,     
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


dataset_path = "dsp_qa_llama3.jsonl"

ds = load_dataset("json", data_files=dataset_path, split="train")

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds = ds.map(format_chat)

tokenized = ds.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        max_length=1024,  
    ),
    batched=True,
    remove_columns=ds.column_names,
)

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


training_args = TrainingArguments(
    output_dir="./llama3_dsp_lora_6GB",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  
    learning_rate=2e-4,
    warmup_steps=10,
    max_steps=600,
    fp16=True,
    bf16=False,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=200,
    lr_scheduler_type="cosine",
)


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    train_dataset=tokenized,
)

trainer.train()
trainer.save_model("./llama3_dsp_lora")

print("\nModel saved \n")
