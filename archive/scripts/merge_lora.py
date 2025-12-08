#!/usr/bin/env python
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = "models/dsp_lora"
OUT_DIR = "models/qwen2_5_1_5b_dsp_merged"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",          # keep it simple, merge on CPU
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("Merging LoRA weights into base...")
    model = model.merge_and_unload()   # turns it into a plain HF model

    print("Saving merged model to:", OUT_DIR)
    model.save_pretrained(OUT_DIR)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(OUT_DIR)

    print("Done. Merged HF model is in:", OUT_DIR)


if __name__ == "__main__":
    main()
