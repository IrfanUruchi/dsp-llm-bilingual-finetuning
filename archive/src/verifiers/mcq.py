import math, numpy as np

def build_distractors(spec: dict, params: dict, solver_result: dict):
    distractors = []
    for em in spec.get("error_models", []):
        loc = {"math": math, "np": np, **params, **solver_result}
        try:
            exec(em["code"], {"__builtins__": {}}, loc)
            for k,v in list(loc.items()):
                if k.endswith("_wrong"):
                    distractors.append(v)
        except Exception:
            continue
    
    uniq = []
    true_vals = set(map(lambda x: round(x,6) if isinstance(x,(float,int)) else x,
                        solver_result.values()))
    for d in distractors:
        key = round(d,6) if isinstance(d,(float,int)) else str(d)
        if key not in true_vals and key not in uniq:
            uniq.append(key)
    return uniq[:3]  
