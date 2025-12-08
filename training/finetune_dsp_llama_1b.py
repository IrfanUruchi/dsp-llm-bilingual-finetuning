#!/usr/bin/env python
import os
from dataclasses import dataclass

from datasets import load_dataset, DatasetDict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

@dataclass
class TrainConfig:
    base_model: str = "llm-dsp/datasets/Stackoverflow/Llama-3.2-1B-Instruct"

    dataset_path: str = "../datasets/generated/dsp_albanian_v3_hybrid.jsonl"

    output_dir: str = "models/dsp_llama_1b_lora"

    max_seq_len: int = 1024
    train_epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    grad_acc: int = 8

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    seed: int = 42
    logging_steps: int = 20
    save_steps: int = 200


cfg = TrainConfig()

def load_dsp_dataset() -> DatasetDict:
    """
    Load the Albanian DSP hybrid dataset:
      {"messages": [ {role, content}, ... ]}
    and split train/validation.
    """
    raw = load_dataset(
        "json",
        data_files={"train": cfg.dataset_path},
    )["train"]

    raw = raw.shuffle(seed=cfg.seed)
    split = raw.train_test_split(test_size=0.1, seed=cfg.seed)

    ds = DatasetDict(
        train=split["train"],
        validation=split["test"],
    )
    print(f"[DATA] Train: {len(ds['train'])} | Validation: {len(ds['validation'])}")
    return ds


def get_model_and_tokenizer():
    print(f"[MODEL] Loading base model from: {cfg.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16, 
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return tokenizer, model

def formatting_func(tokenizer):
    """
    Each row has:
      {"messages": [ {role, content}, ... ]}
    We convert that into a chat-formatted 'text' string
    using Llama's chat template.
    """
    def _format(batch):
        texts = []
        msgs_list = batch["messages"] 
        for msgs in msgs_list:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    return _format

def tokenize_func(tokenizer):
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
            padding=False,
        )
    return _tok

def main():
    os.makedirs(cfg.output_dir, exist_ok=True)

    dataset = load_dsp_dataset()
    tokenizer, model = get_model_and_tokenizer()

    fmt = formatting_func(tokenizer)
    tok = tokenize_func(tokenizer)

    train_text = dataset["train"].map(
        fmt,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    val_text = dataset["validation"].map(
        fmt,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    train_tok = train_text.map(
        tok,
        batched=True,
        remove_columns=["text"],
    )
    val_tok = val_text.map(
        tok,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_acc,
        num_train_epochs=cfg.train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        fp16=True,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    print("[TRAINING] Starting training with Llama-3.2-1B-Instruct...")
    trainer.train()

    print("[SAVE] Saving LoRA adapter and tokenizer...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("[DONE] LoRA Llama DSP model saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
