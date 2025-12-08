import os
import json
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "llm-dsp/llm_dsp_english"
TEMP = "llm-dsp/llm_dsp_english_noquant"
DST = "llm-dsp/llm_dsp_english_fp16"

if os.path.exists(TEMP):
    print("[0] Removing old temp folder:", TEMP)
    shutil.rmtree(TEMP)

print("[0] Copying", SRC, "to temp folder", TEMP)
shutil.copytree(SRC, TEMP)

config_path = os.path.join(TEMP, "config.json")
print("[0] Cleaning config:", config_path)

with open(config_path, "r") as f:
    cfg = json.load(f)

if "quantization_config" in cfg:
    print("[0] Removing 'quantization_config' from config")
    cfg.pop("quantization_config")

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

os.makedirs(DST, exist_ok=True)

print("[1] Loading tokenizer from", TEMP)
tokenizer = AutoTokenizer.from_pretrained(TEMP)
tokenizer.save_pretrained(DST)
print("[1] Tokenizer saved to", DST)

print("[2] Loading model from (cleaned) temp folder", TEMP)
model = AutoModelForCausalLM.from_pretrained(
    TEMP,
    torch_dtype="float16",   
    device_map=None,      
    low_cpu_mem_usage=False,
)

print("[3] First parameter dtype:", next(model.parameters()).dtype)

print("[4] Saving fp16 model to", DST)
model.save_pretrained(DST, safe_serialization=True)

print("[DONE] Saved pure fp16 HF model to", DST)
