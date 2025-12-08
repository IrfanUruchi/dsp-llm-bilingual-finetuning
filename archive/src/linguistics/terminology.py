from typing import List, Dict, Any, Set
import csv, os

def _load_glossary(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    terms = set()
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            terms.add(row[0].strip().lower())
            if len(row) > 1:
                terms.add(row[1].strip().lower())
    return terms

def terminology_hit_rate(samples: List[Dict[str, Any]], glossary_path: str) -> float:
    terms = _load_glossary(glossary_path)
    if not samples:
        return 0.0
    if not terms:
        return 1.0
    hits = 0
    for s in samples:
        txt = (s.get("response_en","") + " " + s.get("response_sq","")).lower()
        hits += int(any(t in txt for t in terms))
    return hits / len(samples)
