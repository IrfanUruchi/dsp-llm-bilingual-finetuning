#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

BASE_MODEL = "llm-dsp/datasets/Stackoverflow/Llama-3.2-1B-Instruct"
LORA_ADAPTER = "llm-dsp/training/models/dsp_llama_1b_lora"

def main():
   
    print("------------------------------------------------")
    print("DIAGNOSTIC MODE")
    print("1. Load Base Model ONLY (Test if Llama is healthy)")
    print("2. Load Base + LoRA (Test your fine-tuning)")
    print("------------------------------------------------")
    choice = input("Select [1] or [2]: ").strip()

    use_lora = choice == "2"

    print(f"\n[LOAD] Loading Tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[LOAD] Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if use_lora:
        print(f"[LOAD] Loading LoRA Adapter: {LORA_ADAPTER}")
        try:
            model = PeftModel.from_pretrained(model, LORA_ADAPTER)
            print("[SUCCESS] Adapter attached.")
        except Exception as e:
            print(f"[ERROR] Could not load LoRA: {e}")
            return
    else:
        print("[INFO] Running in PURE BASE mode (No Fine-tuning).")

    model.eval()
   
    B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"

    system_prompt = "Je një asistent i dobishëm. Përgjigju vetëm në Shqip."

    print("\nExample Question: 'Kush është kryeqyteti i Shqipërisë?'")

    while True:
        msg = input("\n💬 Pyetje: ")
        if msg.lower() in ["exit", "quit"]: break

        full_prompt = f"{B_SYS}{system_prompt}{E_SYS}{B_INST}{msg}{E_INST}"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,         
                temperature=0.6,       
                top_p=0.9,
                repetition_penalty=1.2,s
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖: {text}")

if __name__ == "__main__":
    main()