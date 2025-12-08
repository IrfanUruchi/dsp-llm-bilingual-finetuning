#!/usr/bin/env python3

import os, re, json, math, torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL  = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "outputs/qwen2_5-1_5b_dsp_v5_final")
MAX_NEW     = int(os.environ.get("MAX_NEW", "220"))
JSON_MODE   = bool(int(os.environ.get("JSON_MODE", "0")))     
MATH_FIRST  = bool(int(os.environ.get("MATH_FIRST", "1")))       
LLM_ONLY    = bool(int(os.environ.get("LLM_ONLY", "0")))          

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM_SQ = (
    "Ti je një asistent për Përpunimin Digjital të Sinjaleve (DSP) dhe përgjigjesh "
    "vetëm në shqip. Ruaj saktë numrat, njësitë dhe simbolët (Δf, k, N, fs, f0). "
    "Ji i shkurtër dhe teknik."
)

def compute_fft_metrics(f0: float, fs: float, N: int) -> Tuple[float, int, float, bool]:
    """Return Δf (Hz), k (int), alias_hz (Hz), aliasing(bool)."""
    df = fs / N
    k  = int(round(f0 / df))
    fmod = f0 % fs
    alias = fs - fmod if fmod > fs/2 else fmod
    aliasing = f0 > fs/2
    return df, k, float(alias), aliasing


_RX_NUM   = r"([-+]?\d+(?:\.\d+)?)"
def _find_first(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.I)
    if not m: return None
    return float(m.group(1))

def parse_params(text: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:

    f0 = _find_first(r"(?:f0|f_0|f\s*0)\s*=\s*" + _RX_NUM, text)
    fs = _find_first(r"(?:fs|f_s|f\s*s)\s*=\s*" + _RX_NUM, text)
    N  = _find_first(r"(?:\bN\b)\s*=\s*" + _RX_NUM, text)
    
    if f0 is None:
        f0 = _find_first(r"(?:frekuenc[ëe]|frequency)\s*" + _RX_NUM + r"\s*Hz", text)
    if fs is None:
        fs = _find_first(r"(?:mostrohet me fs\s*=\s*|sampling(?:\s+rate)?\s*=?\s*)" + _RX_NUM, text)
    if N is None:
        N = _find_first(r"(?:FFT\s*me\s*N\s*=\s*|FFT with N\s*=\s*)" + _RX_NUM, text)
   
    if f0 is None or fs is None or N is None:
        nums = [float(x) for x in re.findall(_RX_NUM, text)]
       
        if len(nums) >= 3:
            cand = sorted(nums, reverse=True)  
            if fs is None: fs = cand[0]
            if f0 is None: f0 = cand[1]
            if N  is None:
              
                for z in cand[2:] + cand[:2]:
                    if abs(z - round(z)) < 1e-6 and 2 <= int(round(z)) <= 10_000_000:
                        N = z; break
                if N is None: N = cand[2]
 
    if N is not None: N = int(round(float(N)))
    return f0, fs, N

def format_explanation_sq(f0, fs, N, df, k, alias, aliasing) -> str:
    nyq = fs/2.0
    alias_txt = "po, ka aliasim (f0 > fs/2)" if aliasing else "jo, s’ka aliasim (f0 ≤ fs/2)"
    return (
        f"Δf = fs/N = {fs:.6g}/{N:d} = {df:.6g} Hz. "
        f"k = round(f0/Δf) = round({f0:.6g}/{df:.6g}) = {k:d}. "
        f"Nyquist = fs/2 = {nyq:.6g} Hz → {alias_txt}; "
        f"frekuenca e dukshme = {alias:.6g} Hz."
    )

def load_model_and_tokenizer():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|endoftext|>"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    return tok, model

tok, mdl = load_model_and_tokenizer()

def _collect_eos_ids(tokenizer):
    ids = []
    for s in ("<|im_end|>", tokenizer.eos_token):
        if not s: continue
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0: ids.append(tid)
        except Exception: pass
    if not ids:
        fallback = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        ids = [fallback]

    seen, out = set(), []
    for t in ids:
        if t not in seen: seen.add(t); out.append(t)
    return out

def build_inputs(messages):
    enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(enc, torch.Tensor):
        input_ids = enc.to(mdl.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=mdl.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        return {k: v.to(mdl.device) for k, v in enc.items()}

def _extract_first_json_block(text: str) -> str:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    return m.group(0) if m else "{}"

def _postprocess(text: str, json_mode: bool) -> str:
 
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    text = text.strip()
    if not json_mode:
        return text
    block = _extract_first_json_block(text)
    try:
        json.loads(block)
        return block
    except Exception:
        return "{}"

def llm_answer(user_text: str, json_mode: bool = False) -> str:
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

def solve_or_llm(prompt: str, json_mode: bool) -> str:
    if not LLM_ONLY and MATH_FIRST:
        f0, fs, N = parse_params(prompt)
        if f0 is not None and fs is not None and N is not None and N > 0 and fs > 0:
            df, k, alias_hz, aliasing = compute_fft_metrics(f0, fs, N)
            if json_mode:
                obj = {"df_hz": df, "k": k, "alias_hz": alias_hz}
                return json.dumps(obj, ensure_ascii=False)
            else:
                return format_explanation_sq(f0, fs, N, df, k, alias_hz, aliasing)
    
    return llm_answer(prompt, json_mode=json_mode)

if __name__ == "__main__":
    print(">>> BASE_MODEL:", BASE_MODEL)
    print(">>> ADAPTER_DIR:", ADAPTER_DIR)
    try:
        print(">>> <|im_end|> id:", tok.convert_tokens_to_ids("<|im_end|>"))
    except Exception:
        print(">>> <|im_end|> id: None")

    q1 = (
        "Një sinusoidë me frekuencë 12000 Hz mostrohet me fs = 48000 Hz dhe analizohet "
        "me një FFT me N = 1024 pika. Llogarit rezolucionin në frekuencë Δf dhe indeksin "
        "e bin-it k që korrespondon me tonin. Trego nëse ka aliasim ndaj frekuencës Nyquist."
    )
    print("\n===== Shembull 1 (shpjegim) =====")
    print(solve_or_llm(q1, json_mode=False))

    q2 = (
        "Një sinusoidë me frekuencë 9675.02 Hz mostrohet me fs = 32889 Hz dhe analizohet "
        "me një FFT me N = 3308 pika. Kthe rezultatet në JSON."
    )
    print("\n===== Shembull 2 (vetëm JSON) =====")
    print(solve_or_llm(q2, json_mode=True))

    q3 = (
        "Një sinusoidë me frekuencë 14773.51 Hz mostrohet me fs = 63925 Hz dhe analizohet "
        "me një FFT me N = 1511 pika. Llogarit Δf dhe k; trego edhe nëse ka aliasim."
    )
    print("\n===== Shembull 3 (shpjegim) =====")
    print(solve_or_llm(q3, json_mode=False))
