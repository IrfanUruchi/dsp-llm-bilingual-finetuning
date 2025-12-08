
import random
from .common import load_spec, eval_expr

def sample_params(spec: dict, difficulty: str, seed: int):
    random.seed(seed)
    bucket = spec["difficulty_buckets"][difficulty]

    N = random.randint(*bucket["N"])
    fs = random.randint(*bucket["fs"])

 
    bucket_no_conflicts = {k: v for k, v in bucket.items() if k not in ("N", "fs")}
    ctx = {
        "difficulty_bucket": bucket,
        **bucket_no_conflicts,     
        "N": N,
        "fs": fs,
    }

    f0 = eval_expr(spec["parameters"]["f0"]["expr"], ctx)

    params = {"N": N, "fs": fs, "f0": f0}

    for test in spec.get("acceptance_tests", []):
        ok = bool(eval_expr(test["expr"], {**params, **bucket_no_conflicts}))
        if not ok:
            raise ValueError(f"Acceptance test failed: {test['name']}")
    return params
