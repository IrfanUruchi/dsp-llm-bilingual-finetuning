import json
from pathlib import Path

IN_PATH = Path("data/interim/sampling_aliasing_qwen3_235b.jsonl")
OUT_VALID = Path("data/interim/sampling_aliasing_qwen3_235b.valid.jsonl")
OUT_INVALID = Path("data/interim/sampling_aliasing_qwen3_235b.invalid.jsonl")

OUT_VALID.parent.mkdir(parents=True, exist_ok=True)


def check_numeric_meta(item: dict) -> bool:
    """
    Basic numeric sanity check for sampling_aliasing items.

    Rules:
    - If meta.fs or meta.has_aliasing missing -> treat as UNKNOWN -> accept.
    - If both fs and f0 present:
        * nyq = fs/2
        * If meta.has_aliasing is False, require f0 <= nyq + tol.
        * If meta.has_aliasing is True, require f0 > nyq - tol.
    """
    meta = item.get("meta") or {}

    fs = meta.get("fs")
    f0 = meta.get("f0")
    has_aliasing = meta.get("has_aliasing")

    if fs is None:
        return True

    if f0 is None:
        return True

    try:
        fs_val = float(fs)
        f0_val = float(f0)
    except (TypeError, ValueError):
        return False

    if fs_val <= 0 or f0_val < 0:
        return False

    nyq = fs_val / 2.0
    tol = 1e-6

    if not isinstance(has_aliasing, bool):
        return True

    if has_aliasing:
        return f0_val > nyq - tol
    else:
        return f0_val <= nyq + tol


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")

    n_total = 0
    n_valid = 0
    n_invalid = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, \
         OUT_VALID.open("w", encoding="utf-8") as fv, \
         OUT_INVALID.open("w", encoding="utf-8") as fi:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                fi.write(line + "\n")
                n_invalid += 1
                continue

            ok = check_numeric_meta(item)

            if ok:
                fv.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_valid += 1
            else:
                fi.write(json.dumps(item, ensure_ascii=False) + "\n")
                n_invalid += 1

    print(f"Total items   : {n_total}")
    print(f"Valid numeric : {n_valid}")
    print(f"Invalid numeric: {n_invalid}")
    print(f"Valid -> {OUT_VALID}")
    print(f"Invalid -> {OUT_INVALID}")


if __name__ == "__main__":
    main()
