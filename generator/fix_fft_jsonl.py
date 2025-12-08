# scripts/fix_fft_jsonl.py

import json
import ast
from pathlib import Path

IN_PATH = Path("data/interim/fft_qwen3_235b.jsonl")
OUT_PATH = Path("data/interim/fft_qwen3_235b_fixed.jsonl")


def extract_objects_loose(text: str):
    objs = []
    n = len(text)
    in_string = False
    escape = False
    brace_depth = 0
    start_idx = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif ch == "}":
                if brace_depth > 0:
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        obj_str = text[start_idx:i+1]
                    
                        try:
                            obj = json.loads(obj_str)
                        except json.JSONDecodeError:
                            try:
                                obj = ast.literal_eval(obj_str)
                            except Exception:
                                obj = None
                        if isinstance(obj, dict):
                            objs.append(obj)
                        start_idx = None

    return objs


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    text = IN_PATH.read_text(encoding="utf-8")
    objs = extract_objects_loose(text)
    print(f"Found {len(objs)} JSON objects in the broken file.")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote fixed JSONL to {OUT_PATH}")


if __name__ == "__main__":
    main()
