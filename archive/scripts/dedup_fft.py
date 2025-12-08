# scripts/dedup_fft.py

import json
import hashlib
from pathlib import Path

IN_PATH = Path("data/interim/fft_qwen3_235b.valid.jsonl")
OUT_PATH = Path("data/verified/fft_qwen3_235b.dedup.jsonl")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def make_key(item: dict) -> str:
    topic = item.get("topic", "")
    subtopic = item.get("subtopic", "")
    prompt_en = item.get("prompt_en", "")
    prompt_sq = item.get("prompt_sq", "")

    key_str = f"{topic}||{subtopic}||{prompt_en}||{prompt_sq}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    seen = set()
    n_in = 0
    n_out = 0
    n_dup = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = make_key(item)
            if key in seen:
                n_dup += 1
                continue

            seen.add(key)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Input items : {n_in}")
    print(f"Output items: {n_out}")
    print(f"Duplicates  : {n_dup}")
    print(f"Deduped dataset -> {OUT_PATH}")


if __name__ == "__main__":
    main()
