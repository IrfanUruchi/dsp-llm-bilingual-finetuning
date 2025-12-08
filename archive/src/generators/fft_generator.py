from dataclasses import dataclass
from typing import Dict, Any, List
import random, math, json, re, yaml
from .base_generator import BaseGenerator

class FFTGenerator(BaseGenerator):
    def generate_example(self, subtopic, difficulty):
    
        if subtopic == "bin_index_resolution":
            prompt_en = "A sinusoid of frequency 9675 Hz..."
            response_en = "Δf = fs/N ..."
            return {
                "id": f"FFT_{subtopic}_{difficulty}_{random.randint(1000,9999)}",
                "lang_pair": ["en", "sq"],
                "topic": "FFT",
                "subtopic": subtopic,
                "type": "numeric",
                "difficulty": difficulty,
                "prompt_en": prompt_en,
                "response_en": response_en,
                "prompt_sq": "Një sinusoid...",
                "response_sq": "Δf = fs/N ...",
                "verified": False
            }



def load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class Sample:
    spec_path: str
    seed: int
    difficulty: str
    params: Dict[str, Any]
    solver_result: Dict[str, Any]
    proposer_raw: Dict[str, Any]
    bilingual_raw: Dict[str, Any]
    packaged: Dict[str, Any]

def seeded_uniform(a,b):
    return random.uniform(a,b)

def eval_expr(expr: str, context: dict) -> Any:
    
    local = {**context, "uniform": seeded_uniform, "math": math}
    return eval(expr, {"__builtins__": {}}, local)
