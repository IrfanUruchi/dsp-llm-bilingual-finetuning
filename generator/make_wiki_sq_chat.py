#!/usr/bin/env python
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
IN_DIR = BASE / "raw" / "wikipedia_sq"
OUT_PATH = BASE / "generated" / "albanian_wiki_chat.jsonl"

MIN_CHARS = 20 
MAX_CHARS = 5000


def iter_texts():
    file_count = 0
    line_count = 0
    text_count = 0

    for root, _, files in os.walk(IN_DIR):
        for fn in files:
            if not fn.startswith("wiki_"):
                continue
            path = Path(root) / fn
            file_count += 1
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    text = obj.get("text", "")
                    if not text:
                        continue
                    text_count += 1
                    yield text

    print(f"[DEBUG] Files: {file_count}, lines read: {line_count}, texts found: {text_count}")


def build_chat(paragraph: str):
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Je një shkrimtar dhe shpjegues në gjuhën shqipe. "
                    "Përdor shqip të pastër, gramatikisht të saktë."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Shkruaj një paragraf informues dhe të qartë në shqip "
                    "për një temë të përgjithshme."
                ),
            },
            {
                "role": "assistant",
                "content": paragraph,
            },
        ]
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for text in iter_texts():
            para = text.strip()
            if len(para) < MIN_CHARS or len(para) > MAX_CHARS:
                continue

            obj = build_chat(para)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[DONE] Wrote {n_written} wiki-based Albanian examples to: {OUT_PATH}")


if __name__ == "__main__":
    main()
