import argparse, yaml, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    out = {"config": cfg, "scores": {}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[score] wrote {args.out} (stub)")

if __name__ == "__main__":
    main()
