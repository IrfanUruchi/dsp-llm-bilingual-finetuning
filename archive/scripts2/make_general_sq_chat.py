#!/usr/bin/env python
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
IN_PATH = BASE / "raw" / "albanian_general.txt"
OUT_PATH = BASE / "generated" / "albanian_general_chat.jsonl"


def build_chat(paragraph: str):
    paragraph = paragraph.strip()
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
                "content": "Shkruaj një paragraf të qartë në shqip për një temë të përgjithshme.",
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
    with IN_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = build_chat(line)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[DONE] Wrote {n_written} general Albanian examples to: {OUT_PATH}")


if __name__ == "__main__":
    main()
