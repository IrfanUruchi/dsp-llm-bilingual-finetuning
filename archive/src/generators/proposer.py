import json
from typing import Dict
import torch
from .json_utils import parse_json_strict, ensure_keys, REQUIRED_PROP_KEYS

class Proposer:
    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model

    def _prompt(self, topic: str, subtopic: str, params: Dict[str, float], retry_hint: str = "") -> str:
        sys = (
            "You are a Computer Engineering DSP TA.\n"
            "Return ONLY a single JSON object with keys: problem_en, solution_en, python_en.\n"
            "No commentary, no Markdown, no code fences."
        )
        usr = (
            f"Topic: {topic}/{subtopic}\n"
            f"Parameters (use exactly): {json.dumps(params)}\n"
            "Units: SI. Provide python_en that recomputes the numeric results and assigns a dict to variable `result`.\n"
            "Example python_en ending lines:\n"
            "df = fs/N\n"
            "k  = round(f0/df)\n"
            "result = {\"k\": int(k), \"df\": float(df)}\n"
            f"{retry_hint}"
        )
        return f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"

    def __call__(self, topic: str, subtopic: str, params: Dict[str, float]) -> Dict:
      
        hints = ["", "Your previous output was not valid JSON or missed keys. Output JSON only.", 
                 "You MUST include 'python_en' that sets a variable named result (a dict)."]
        for hint in hints:
            prompt = self._prompt(topic, subtopic, params, retry_hint=hint)
            ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **ids, max_new_tokens=480, do_sample=True, temperature=0.3, top_p=0.9,
                pad_token_id=self.tok.eos_token_id
            )
            text = self.tok.decode(out[0], skip_special_tokens=True)
            try:
                obj = parse_json_strict(text)
                ensure_keys(obj, REQUIRED_PROP_KEYS)
                
                if "result" not in obj.get("python_en",""):
                    raise KeyError("python_en must assign a variable named result (dict).")
                return obj
            except Exception:
                continue
      
        raise ValueError("Proposer could not produce strict JSON with required keys.")

