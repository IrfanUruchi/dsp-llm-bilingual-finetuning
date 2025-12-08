#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import sys
import os

# --- PATHS ---
BASE_MODEL = "llm-dsp/datasets/Stackoverflow/Llama-3.2-1B-Instruct"
LORA_DIR   = "llm-dsp/training/models/dsp_llama_1b_lora"

def load_model():
    print(f"[LOAD] Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[LOAD] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    if os.path.exists(LORA_DIR):
        print(f"[LOAD] Loading & Merging LoRA adapter from {LORA_DIR}...")
        model = PeftModel.from_pretrained(model, LORA_DIR)
        model = model.merge_and_unload() # Makes inference faster
    else:
        print(f"\nWARNING: LoRA adapter not found at {LORA_DIR}")
        print("Running with Base Model only (Expect English or generic answers).\n")

    model.eval()
    return tokenizer, model

def generate(model, tokenizer, text):
    
    messages = [
        {
            "role": "system", 
            "content": (
                "Je një inxhinier ekspert i DSP (Përpunimi i Sinjalit Dixhital). "
                "Përgjigju shkurt, saktë dhe vetëm në gjuhën Shqipe. "
                "Mos përsërit fjalë."
            )
        },
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,     
        top_p=0.9,
        repetition_penalty=1.2, 
        no_repeat_ngram_size=3, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=generation_config
        )

    input_length = inputs["input_ids"].shape[1]
    response_tokens = output[0][input_length:]
    
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response.strip()

def main():
    tokenizer, model = load_model()

    tests = [
        "Përshëndetje, si je?",
        "Çfarë është një sinjal dixhital?",
        "Çfarë është frekuenca e mostrimit?",
        "Shpjego me fjalë të thjeshta çfarë është FFT.",
        "Çfarë është aliasimi? Jep një shembull të thjeshtë.",
        "Shkruaj një paragraf të shkurtër për qytetin e Shkodrës.", # Non-DSP test
    ]

    print("\n==========================")
    print("   TESTING LLaMA 1B DSP")
    print("==========================\n")

    for q in tests:
        print(f"💬 PYETJE: {q}")
        ans = generate(model, tokenizer, q)
        print("=== PËRGJIGJA ===")
        print(ans)
        print("\n--------------------------\n")

if __name__ == "__main__":
    main()