#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# --- PATHS ---
BASE = "llm-dsp/datasets/Stackoverflow/Llama-3.2-1B-Instruct"
LORA = "llm-dsp/training/models/dsp_llama_1b_lora"

def main():
    print("[LOAD] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[LOAD] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("[LOAD] Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA)
    model.eval()

    system_instruction = (
        "Ti je një ekspert i teknologjisë. "
        "Përgjigju pyetjeve shkurt dhe saktë në gjuhën Shqipe. "
        "Mos përsërit fjalë. Mos shpik informacione."
    )

    print("\n[READY] Model loaded. Type 'exit' to quit.\n")

    while True:
        try:
            msg = input("💬 Pyetje: ")
            if msg.strip().lower() in ["exit", "quit", "stop"]:
                break

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": msg}
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
                temperature=0.1,        
                top_p=0.9,
                repetition_penalty=1.2,  
                no_repeat_ngram_size=3,  
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )

            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            print("\n=== RESPONSE ===")
            print(text)
            print("================\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()