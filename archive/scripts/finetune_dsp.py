#!/usr/bin/env python
import os
from dataclasses import dataclass

from datasets import load_dataset, DatasetDict, concatenate_datasets

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


@dataclass
class TrainConfig:
    base_model: str = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")

    fft_path: str = "data/verified/fft_qwen3_235b.dedup.jsonl"
    sampling_path: str = "data/verified/sampling_aliasing_qwen3_235b.dedup.jsonl"


    output_dir: str = "models/dsp_lora"

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
    eval_steps: int = 200
    save_steps: int = 200


cfg = TrainConfig()


def load_combined_dataset() -> DatasetDict:
    data_files = {
        "fft": cfg.fft_path,
        "sampling": cfg.sampling_path,
    }

    raw = load_dataset("json", data_files=data_files)
    combined = concatenate_datasets([raw["fft"], raw["sampling"]])

    combined = combined.shuffle(seed=cfg.seed)
    split = combined.train_test_split(test_size=0.1, seed=cfg.seed)

    ds = DatasetDict(
        train=split["train"],
        validation=split["test"],
    )
    print(f"[DATA] Train: {len(ds['train'])} | Validation: {len(ds['validation'])}")
    return ds

def get_model_and_tokenizer():
    print(f"[MODEL] Loading base model: {cfg.base_model}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
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


def build_formatting_func(tokenizer):

    def _format(batch):
        texts = []
        keys = list(batch.keys())

        if all(k in keys for k in ["prompt_en", "response_en", "prompt_sq", "response_sq"]):
            systems = batch["system"] if "system" in keys else [None] * len(batch["prompt_en"])
            prompt_en = batch["prompt_en"]
            response_en = batch["response_en"]
            prompt_sq = batch["prompt_sq"]
            response_sq = batch["response_sq"]

            for sys_msg, pe, re, ps, rs in zip(systems, prompt_en, response_en, prompt_sq, response_sq):
                msgs = []

                if sys_msg is not None and sys_msg != "":
                    msgs.append({"role": "system", "content": sys_msg})

                user_parts = []
                if pe is not None and pe != "":
                    user_parts.append("EN:\n" + pe)
                if ps is not None and ps != "":
                    user_parts.append("SQ:\n" + ps)
                user_content = "\n\n".join(user_parts) if user_parts else ""

                asst_parts = []
                if re is not None and re != "":
                    asst_parts.append("EN:\n" + re)
                if rs is not None and rs != "":
                    asst_parts.append("SQ:\n" + rs)
                asst_content = "\n\n".join(asst_parts) if asst_parts else ""

                if user_content == "" or asst_content == "":
                    continue

                msgs.append({"role": "user", "content": user_content})
                msgs.append({"role": "assistant", "content": asst_content})

                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)

            return {"text": texts}

        if "messages" in keys:
            for msgs in batch["messages"]:
                text = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            return {"text": texts}

        raise ValueError(f"Unknown dataset schema. Keys: {keys}")

    return _format


def build_tokenize_func(tokenizer):
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

    dataset = load_combined_dataset()
    tokenizer, model = get_model_and_tokenizer()

    fmt_fn = build_formatting_func(tokenizer)

    train_text = dataset["train"].map(
        fmt_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    val_text = dataset["validation"].map(
        fmt_fn,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    tok_fn = build_tokenize_func(tokenizer)

    train_tok = train_text.map(
        tok_fn,
        batched=True,
        remove_columns=["text"],
    )
    val_tok = val_text.map(
        tok_fn,
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

    print("[TRAINING] Starting training...")
    trainer.train()

    print("[SAVE] Saving adapter and tokenizer...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("[DONE] LoRA DSP model saved to:", cfg.output_dir)


if __name__ == "__main__":
    main()
