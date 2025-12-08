#!/usr/bin/env python3
import json, sys, os
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python convert_for_training.py <input.jsonl> <output.jsonl>")
    sys.exit(1)

inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
outp.parent.mkdir(parents=True, exist_ok=True)

count = 0
with open(inp, "r", encoding="utf-8") as fin, open(outp, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
 
        fout.write(json.dumps({
            "instruction": obj["prompt_en"],
            "input": "",
            "output": obj["response_en"],
            "lang": "en"
        }, ensure_ascii=False) + "\n")

        fout.write(json.dumps({
            "instruction": obj["prompt_sq"],
            "input": "",
            "output": obj["response_sq"],
            "lang": "sq"
        }, ensure_ascii=False) + "\n")
        count += 2

print(f"Wrote {count} instruction examples → {outp}")
