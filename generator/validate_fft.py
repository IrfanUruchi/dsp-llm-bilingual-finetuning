# scripts/validate_fft.py

import json
from pathlib import Path

IN_PATH = Path("data/interim/fft_qwen3_235b.jsonl")
OUT_VALID = Path("data/interim/fft_qwen3_235b.valid.jsonl")
OUT_INVALID = Path("data/interim/fft_qwen3_235b.invalid.jsonl")

OUT_VALID.parent.mkdir(parents=True, exist_ok=True)


def almost_equal(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b), 1.0), abs_tol)


def check_fft_meta(item: dict) -> bool:

    meta = item.get("meta") or {}

    fs = meta.get("fs")
    N = meta.get("N")
    df = meta.get("df")
    f0 = meta.get("f0")
    k = meta.get("k")

    
    if fs is None or N is None:
        return True

    try:
        fs_val = float(fs)
        N_val = float(N)
    except (TypeError, ValueError):
        return False

    if fs_val <= 0 or N_val <= 0:
        return False

    df_calc = fs_val / N_val 

   
    if df is not None:
        try:
            df_val = float(df)
        except (TypeError, ValueError):
            return False
        if not almost_equal(df_val, df_calc):
            return False

   
    if f0 is not None and k is not None:
        try:
            f0_val = float(f0)
            k_val = int(round(float(k)))
        except (TypeError, ValueError):
            return False

        if f0_val < 0:
            return False

        k_calc = int(round(f0_val / df_calc))

        
        if abs(k_val - k_calc) > 2:
          
            if k_val < 0 or k_val > 4 * N_val:
                return False

    return True


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

            ok = check_fft_meta(item)

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
