#!/usr/bin/env python3

import json
import sys
import re
from pathlib import Path
from statistics import mean

REQUIRED_KEYS = [
    "topic", "subtopic", "prompt_en", "response_en",
    "prompt_sq", "response_sq", "meta", "provenance"
]

NUM_PATTERN = re.compile(r"[-+]?[0-9]*\.?[0-9]+")

def extract_numbers(text):
    nums = [round(float(x), 6) for x in NUM_PATTERN.findall(text)]
    return nums


def verify_entry(obj):
    issues = []

    for k in REQUIRED_KEYS:
        if k not in obj:
            issues.append(f"Missing key: {k}")

    for field in ["prompt_en", "prompt_sq", "response_en", "response_sq"]:
        if not isinstance(obj.get(field, ""), str) or not obj[field].strip():
            issues.append(f"Empty or invalid field: {field}")

    nums_en = extract_numbers(obj.get("response_en", ""))
    nums_sq = extract_numbers(obj.get("response_sq", ""))
    if nums_en and nums_sq and len(nums_en) == len(nums_sq):
        diffs = [abs(a - b) for a, b in zip(nums_en, nums_sq)]
        if any(d > 1e-5 for d in diffs):
            issues.append("Numeric mismatch between EN and SQ responses.")
    elif nums_en and not nums_sq:
        issues.append("No numbers found in SQ response.")

    gloss_tags = obj.get("meta", {}).get("spec", {}).get("glossary_tags", [])
    if not gloss_tags:
        issues.append("Missing glossary tags in meta.spec.")

    ok = len(issues) == 0
    return ok, issues


def verify_dataset(path):
    total, ok_count = 0, 0
    issues_all = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                issues_all.append((line_no, [f"Invalid JSON: {e}"]))
                continue

            ok, issues = verify_entry(obj)
            total += 1
            if ok:
                ok_count += 1
            else:
                issues_all.append((line_no, issues))

    ratio = ok_count / total if total else 0
    print(f" Verified {ok_count}/{total} items ({ratio*100:.1f}% OK)\n")
    if issues_all:
        print("Issues found:")
        for line_no, probs in issues_all[:10]:  
            print(f"  • Line {line_no}: {probs}")
        if len(issues_all) > 10:
            print(f"... and {len(issues_all)-10} more")
    else:
        print("No issues found — consistent!")

    return {"total": total, "ok": ok_count, "failed": len(issues_all)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    stats = verify_dataset(file_path)
    print("\nSummary:", stats)
