import yaml, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def train_from_config(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], load_in_4bit=True)
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
    model = get_peft_model(model, lora_cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds = load_dataset("json", data_files=cfg["dataset_path"])["train"]

    def format_text(batch):
        return {"text": f"Q: {batch['prompt_en']}\nA: {batch['response_en']}"}

    ds = ds.map(format_text)
    tokenized = ds.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=cfg["max_seq_length"]), batched=True)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_epochs"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_dir="./training/logs",
        fp16=True
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

if __name__ == "__main__":
    train_from_config("training/configs/train_fft_v5.yaml")
