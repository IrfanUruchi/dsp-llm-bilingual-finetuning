import json
from .json_utils import parse_json_strict, ensure_keys, REQUIRED_BIL_KEYS
from src.llm_adapters.base import LLM

class BilingualizerTeacher:
    def __init__(self, teacher: LLM):
        self.teacher = teacher

    def __call__(self, proposer_obj: dict) -> dict:
        sys = ("Translate the following DSP problem+solution to Albanian.\n"
               "Preserve all numbers, symbols (N, fs, f0, Δf, k), and units.\n"
               "Return ONLY a single JSON object with keys: problem_sq, solution_sq.")
        usr = json.dumps({
            "problem_en": proposer_obj["problem_en"],
            "solution_en": proposer_obj["solution_en"]
        }, ensure_ascii=False)
        prompt = f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"
        text = self.teacher.generate(prompt, max_new_tokens=420, temperature=0.2, top_p=0.9)
        obj = parse_json_strict(text)
        ensure_keys(obj, REQUIRED_BIL_KEYS)
        return obj
