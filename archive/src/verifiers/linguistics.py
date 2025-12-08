import csv, re
from typing import List, Dict, Any, Tuple
import sacrebleu

def compute_corpus_bleu_chrf(samples: List[Dict[str, Any]]) -> Tuple[float, float]:
    hyps = [s.get("response_sq","") for s in samples]
    refs = [[s.get("response_en","") for s in samples]]
    if not hyps or not refs[0]:
        return 0.0, 0.0
    bleu = sacrebleu.corpus_bleu(hyps, refs).score
    chrf = sacrebleu.corpus_chrf(hyps, refs).score
    return float(bleu), float(chrf)

def load_glossary(path="GLOSSARY_extended.csv"):
    terms = set()
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                for cell in row:
                    if cell: terms.add(cell.strip().lower())
    except FileNotFoundError:
        pass
    return terms

def glossary_hits(text: str, glossary: set) -> int:
    words = re.findall(r"[A-Za-zëçÇË]+", text.lower())
    return sum(1 for w in words if w in glossary)

def bilingual_alignment(en: str, sq: str) -> bool:
    nums_en = re.findall(r"[-+]?\d*\.?\d+", en)
    nums_sq = re.findall(r"[-+]?\d*\.?\d+", sq)
    return set(nums_en) == set(nums_sq)
