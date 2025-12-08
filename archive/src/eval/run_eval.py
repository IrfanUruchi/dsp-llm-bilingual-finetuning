import argparse, json, yaml
from pathlib import Path
from src.verifiers.numeric import numeric_ok_rate
from src.verifiers.symbolic import symbolic_ok_rate
from src.linguistics.metrics import compute_corpus_bleu_chrf
from src.linguistics.terminology import terminology_hit_rate

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    suite_path = cfg["suites"][args.topic][args.suite]
    samples = list(load_jsonl(Path(suite_path)))

    numeric = numeric_ok_rate(samples)
    symbolic = symbolic_ok_rate(samples)
    bleu, chrf = compute_corpus_bleu_chrf(samples)
    term = terminology_hit_rate(samples, glossary_path="data/raw/glossary_en_sq.csv")

    out = {
        "topic": args.topic,
        "suite": args.suite,
        "metrics": {
            "numeric": numeric,
            "symbolic": symbolic,
            "BLEU": bleu,
            "ChrF": chrf,
            "terminology": term
        }
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eval] wrote {args.out}")

if __name__ == "__main__":
    main()
