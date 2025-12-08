#!/usr/bin/env python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
LORA = "models/dsp_lora"
OUT = "models/qwen2_5_1_5b_dsp_merged"

def main():
    os.makedirs(OUT, exist_ok=True)

    print("[INFO] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("[INFO] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA)

    print("[INFO] Merging LoRA into base weights...")
    model = model.merge_and_unload()

    print("[INFO] Saving merged model...")
    tokenizer.save_pretrained(OUT)
    model.save_pretrained(OUT)

    print(f"[DONE] Merged model saved to: {OUT}")

if __name__ == "__main__":
    main()
