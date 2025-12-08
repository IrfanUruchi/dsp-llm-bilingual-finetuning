import json
from .json_utils import parse_json_strict, ensure_keys, REQUIRED_BIL_KEYS

class Bilingualizer:
    def __init__(self, tokenizer, model):
        self.tok = tokenizer
        self.model = model

    def __call__(self, proposer_obj: dict) -> dict:
        sys = ("Translate to Albanian. Preserve symbols/numbers. "
               "Return ONLY a single JSON object with keys: problem_sq, solution_sq.")
        usr = json.dumps({
            "problem_en": proposer_obj["problem_en"],
            "solution_en": proposer_obj["solution_en"]
        }, ensure_ascii=False)
        prompt = f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids, max_new_tokens=420, do_sample=True, temperature=0.2, top_p=0.9,
            pad_token_id=self.tok.eos_token_id
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        obj = parse_json_strict(text)
        ensure_keys(obj, REQUIRED_BIL_KEYS)
        return obj

