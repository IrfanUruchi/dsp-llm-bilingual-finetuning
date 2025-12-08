#!/usr/bin/env python
import os
import sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = "models/dsp_lora"


def load_model(which: str):

    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {which} model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    if which == "base":
        model = base
    elif which == "lora":
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    else:
        raise ValueError("which must be 'base' or 'lora'")

    model.eval()
    return tokenizer, model


def chat(model, tokenizer, user_en, user_sq=None, max_new_tokens=256):

    user_parts = [f"EN:\n{user_en}"]
    if user_sq:
        user_parts.append(f"SQ:\n{user_sq}")
    user_content = "\n\n".join(user_parts)

    messages = [
        {
            "role": "system",
            "content": "You are a DSP tutor specialized in sampling, aliasing and FFT. Answer in English and optionally Albanian.",
        },
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,     
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    which = "base"
    if len(sys.argv) > 1:
        which = sys.argv[1] 

    tokenizer, model = load_model(which)

    q_en = "For Fs = 10 kHz and N = 1024, what is the FFT bin spacing and what frequency does bin k correspond to?"
    q_sq = "Për Fs = 10 kHz dhe N = 1024, cili është rezolucioni në frekuencë dhe frekuenca e bin-it k?"

    print("\n=== MODEL:", which, "===")
    print("QUESTION (EN):", q_en)
    print("QUESTION (SQ):", q_sq)

    ans = chat(model, tokenizer, q_en, q_sq)
    print("\n--- Answer ---")
    print(ans)


if __name__ == "__main__":
    main()
