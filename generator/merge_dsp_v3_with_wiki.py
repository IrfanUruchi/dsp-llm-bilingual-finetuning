#!/usr/bin/env python
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DSP = BASE / "generated" / "dsp_albanian_v3_math.jsonl"
WIKI = BASE / "generated" / "albanian_wiki_chat.jsonl"
OUT = BASE / "generated" / "dsp_albanian_v3_hybrid.jsonl"

random.seed(42)

def load_jsonl(path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    dsp = load_jsonl(DSP)
    wiki = load_jsonl(WIKI)

    print(f"[LOAD] DSP examples: {len(dsp)}")
    print(f"[LOAD] Wiki examples: {len(wiki)}")

    wiki_sample_size = min(len(wiki), len(dsp) * 2)
    wiki = random.sample(wiki, wiki_sample_size)

    print(f"[INFO] Using {len(wiki)} wiki examples.")

    combined = dsp + wiki
    random.shuffle(combined)

    with OUT.open("w", encoding="utf-8") as f:
        for obj in combined:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[DONE] Hybrid dataset written to: {OUT}")
    print(f"[INFO] Total examples: {len(combined)}")

if __name__ == "__main__":
    main()
