from dataclasses import dataclass
from typing import Dict, Any, List
import random, math, json, re, yaml

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

SAFE_FUNCS = {
    "round": round,
    "int": int,
    "float": float,
    "abs": abs,
    "min": min,
    "max": max,
}

def eval_expr(expr: str, context: dict) -> Any:
   
    local = {**context, "uniform": seeded_uniform, "math": math, **SAFE_FUNCS}
    return eval(expr, {"__builtins__": {}}, local)