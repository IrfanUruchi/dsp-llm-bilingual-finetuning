import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_model_path = "./Llama-3.2-1B-Instruct"
lora_path = "./llama3_dsp_lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_path)

print("Merging LoRA...")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

prompt = "Compute the frequency resolution for an FFT with Fs = 10 kHz and N = 1024."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\nGenerating model output...\n")
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.2,
    top_p=0.9
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
