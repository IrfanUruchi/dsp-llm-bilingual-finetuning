#!/usr/bin/env python3

import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL  = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs/qwen2_5-1_5b_dsp_v5_lowmem")
MAX_NEW     = int(os.environ.get("MAX_NEW", "220"))
JSON_MODE   = bool(int(os.environ.get("JSON_MODE", "0")))  

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_SQ = (
    "Ti je një asistent për Përpunimin Digjital të Sinjaleve (DSP) dhe përgjigjesh "
    "vetëm në shqip. Ruaj saktë numrat, njësitë dhe simbolët (Δf, k, N, fs, f0). "
    "Ji i shkurtër dhe teknik."
)

def load_model_and_tokenizer():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, use_fast=True, trust_remote_code=True
    )

    if tok.pad_token is None:
     
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))

    return tok, model

tok, mdl = load_model_and_tokenizer()

def _collect_eos_ids(tokenizer):

    candidates = []
    for s in ("<|im_end|>", tokenizer.eos_token):
        if not s:
            continue
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0:
                candidates.append(tid)
        except Exception:
            pass

    seen, eos_ids = set(), []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            eos_ids.append(t)

    if not eos_ids:
        eos_ids = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else [tok.pad_token_id]
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
    return eos_ids

def build_inputs(messages):

    enc = tok.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return {k: v.to(mdl.device) for k, v in enc.items()}

def _extract_first_json_block(text: str) -> str:

    m = re.search(r"\{.*?\}", text, flags=re.S)
    return m.group(0) if m else "{}"

def _postprocess(text: str, json_mode: bool) -> str:
    text = text.strip()
    if not json_mode:
        return text

    block = _extract_first_json_block(text)

    try:
        _ = json.loads(block)
        return block
    except Exception:
        return "{}"

def gen_chat(user_text: str, json_mode: bool = False) -> str:
    messages = [{"role": "system", "content": SYSTEM_SQ}]
    if json_mode:
        messages.append({
            "role": "user",
            "content": (
                user_text + "\n\n"
                "Kthe VETËM një objekt JSON me kyçet: "
                "df_hz (numër), k (i plotë), alias_hz (numër). Pa tekst tjetër."
            )
        })
    else:
        messages.append({"role": "user", "content": user_text})

    inputs = build_inputs(messages)
    prompt_len = inputs["input_ids"].shape[-1]

    eos_ids = _collect_eos_ids(tok)

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,          
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
            use_cache=True,
        )

    gen_ids = out[0, prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return _postprocess(text, json_mode)

if __name__ == "__main__":
    q1 = (
        "Një sinusoidë me frekuencë 12000 Hz mostrohet me fs = 48000 Hz dhe analizohet "
        "me një FFT me N = 1024 pika. Llogarit rezolucionin në frekuencë Δf dhe indeksin "
        "e bin-it k që korrespondon me tonin. Trego nëse ka aliasim ndaj frekuencës Nyquist."
    )
    print("\n===== Shembull 1 (shpjegim) =====")
    print(gen_chat(q1, json_mode=False))

    q2 = (
        "Një sinusoidë me frekuencë 9675.02 Hz mostrohet me fs = 32889 Hz dhe analizohet "
        "me një FFT me N = 3308 pika. Kthe rezultatet në JSON."
    )
    print("\n===== Shembull 2 (vetëm JSON) =====")
    print(gen_chat(q2, json_mode=True))

    q3 = (
        "Një sinusoidë me frekuencë 14773.51 Hz mostrohet me fs = 63925 Hz dhe analizohet "
        "me një FFT me N = 1511 pika. Llogarit Δf dhe k; trego edhe nëse ka aliasim."
    )
    print("\n===== Shembull 3 (shpjegim) =====")
    print(gen_chat(q3, json_mode=False))
