from typing import List, Dict, Any
from sympy import sympify, simplify

def _exprs(item: Dict[str, Any]):
    c = item.get("checks", {})
    return c.get("expr_en"), c.get("expr_sq")

def symbolic_ok(item: Dict[str, Any]) -> bool:
    en, sq = _exprs(item)
    if not en or not sq:
        return True
    try:
        return simplify(sympify(en) - sympify(sq)) == 0
    except Exception:
        return False

def symbolic_ok_rate(samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    ok = sum(1 for s in samples if symbolic_ok(s))
    return ok / len(samples)
