import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .base import LLM

class HFLocal(LLM):
    def __init__(self, model_path: str, load_in_4bit: bool = True):
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        qcfg = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=qcfg,
            low_cpu_mem_usage=True,
        )

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9) -> str:
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0], skip_special_tokens=True)
