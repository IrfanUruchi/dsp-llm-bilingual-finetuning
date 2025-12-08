# src/generators/orchestrator.py
import os
import json
import traceback

from .model_loader import load_tok_model
from .proposer import Proposer
from .bilingualizer import Bilingualizer
from .common import load_spec
from .sampler import sample_params

from ..verifiers.numeric import run_python_block, within_tol
from ..verifiers.linguistics import load_glossary, glossary_hits, bilingual_alignment
from ..verifiers.mcq import build_distractors  
from ..verifiers.format import check_schema
from ..packaging.packager import make_row, write_jsonl


def generate_batch(
    spec_path: str,
    model_path: str,
    out_path: str,
    difficulty: str = "medium",
    n: int = 20,
    seed: int = 42,
    glossary_path: str = "GLOSSARY_extended.csv",
):
   

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    spec = load_spec(spec_path)
    tok, shared_model = load_tok_model(model_path)
    prop = Proposer(tok, shared_model)
    bilin = Bilingualizer(tok, shared_model)

    glossary = load_glossary(glossary_path)

    rows = []
    failures = 0

    for i in range(n):
        try:
            
            params = sample_params(spec, difficulty, seed + i)

            prop_out = prop(spec["topic"], spec["subtopic"], params)

            solver_result = run_python_block(prop_out["python_en"], params)

            bilin_out = bilin(prop_out)

            
            assert bilingual_alignment(prop_out["solution_en"], bilin_out["solution_sq"]), \
                "EN/SQ numeric alignment failed"
            if glossary:
                assert glossary_hits(prop_out["problem_en"], glossary) >= 1, \
                    "Glossary coverage too low"

            row = make_row(
                spec["topic"],
                spec["subtopic"],
                params,
                prop_out,
                bilin_out,
                solver_result,
                spec
            )
            check_schema(row)
            rows.append(row)

        except Exception as e:
            failures += 1
            print("sample failed:", repr(e))
            traceback.print_exc()
            continue

   
    if rows:
        write_jsonl(rows, out_path)

    return {"ok": len(rows), "failed": failures, "out": out_path}
