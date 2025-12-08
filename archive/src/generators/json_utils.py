import json, re

REQUIRED_PROP_KEYS = ("problem_en","solution_en","python_en")
REQUIRED_BIL_KEYS  = ("problem_sq","solution_sq")

def extract_last_json_block(text: str) -> str:
    
    start = text.rfind("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found.")
    return text[start:end+1]

def parse_json_strict(text: str) -> dict:
    js = extract_last_json_block(text)
    return json.loads(js)

def ensure_keys(d: dict, required: tuple):
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"Missing keys: {missing}")
