import argparse, yaml
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    print("[QLoRA] Parsed config:")
    print(cfg)
    print("[QLoRA] TODO: implement transformers+peft training loop.")

if __name__ == "__main__":
    main()
