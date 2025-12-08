#!/usr/bin/env python
import argparse, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.generators.orchestrator_teacher import generate_batch_teacher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="YAML spec path (src/specs/*.yaml)")
    ap.add_argument("--teacher_model", required=True, help="Cerebras model ID (e.g., llama-*-instruct)")
    ap.add_argument("--out", default="export/auto_v2/dsp_auto.jsonl")
    ap.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mode", default="programmatic", choices=["programmatic","hybrid"])  
    ap.add_argument("--programmatic_ratio", type=float, default=1.0)
    args = ap.parse_args()

    res = generate_batch_teacher(
        spec_path=args.spec,
        teacher_model_path=args.teacher_model,
        out_path=args.out,
        difficulty=args.difficulty,
        n=args.n,
        seed=args.seed,
        mode=args.mode,
        programmatic_ratio=args.programmatic_ratio,
    )
    print(res)

if __name__ == "__main__":
    main()
