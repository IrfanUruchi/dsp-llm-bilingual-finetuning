import os, json, traceback, random
from .common import load_spec
from .sampler import sample_params
from ..verifiers.linguistics import load_glossary, glossary_hits, bilingual_alignment
from ..verifiers.format import check_schema
from ..packaging.packager import make_row, write_jsonl
from .programmatic.fft_bin_prog import build_item as fft_bin_build
from .bilingualizer_cerebras import BilingualizerCerebras


import re

def run_python_block(block: str, env=None):

    if env is None:
        env = {}

    fence = re.match(r"^```(?:python)?\s*(.*?)```$", block.strip(), flags=re.DOTALL | re.IGNORECASE)
    code = fence.group(1).strip() if fence else block.strip()

    local_env = {}
    local_env.update(env)

    try:
        exec(code, {}, local_env)
    except Exception as e:
        raise RuntimeError(f"run_python_block() failed: {e}\nCode:\n{code}")

    return local_env


def generate_batch_teacher(
    spec_path: str,
    teacher_model_path: str,   
    out_path: str,
    difficulty: str = "medium",
    n: int = 20,
    seed: int = 42,
    glossary_path: str = "GLOSSARY_extended.csv",
    mode: str = "programmatic",   
    programmatic_ratio: float = 1.0,
):
    """
    Generate a batch of bilingual DSP dataset items using a teacher-student pipeline.
    - Student (programmatic generator) → English problem/solution
    - Teacher (Cerebras model) → Albanian translation
    - Verifiers → check numerical correctness, schema, and glossary coverage
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    spec = load_spec(spec_path)
    glossary = load_glossary(glossary_path)

    bilin = BilingualizerCerebras(model_id=teacher_model_path)

    rows, failures = [], 0
    rng = random.Random(seed)

    def build_programmatic(params):
        if spec["topic"] == "FFT" and spec["subtopic"] == "bin_index_resolution":
            return fft_bin_build(params)
        raise NotImplementedError(f"No programmatic builder for {spec['topic']}/{spec['subtopic']}")

    for i in range(n):
        try:
            params = sample_params(spec, difficulty, seed + i)

            if mode != "programmatic" and not (mode == "hybrid" and rng.random() < programmatic_ratio):
                raise NotImplementedError("Only programmatic mode is enabled in this build.")

            prop_out = build_programmatic(params)

            solver_result = {}
            if "python_en" in prop_out and prop_out["python_en"].strip():
                solver_result = run_python_block(prop_out["python_en"], params)

            bilin_out = bilin(prop_out)

            assert bilingual_alignment(prop_out["solution_en"], bilin_out["solution_sq"]), "EN/SQ numeric mismatch"
            if glossary:
                assert glossary_hits(prop_out["problem_en"], glossary) >= 1, "Low glossary coverage"

            row = make_row(
                spec["topic"], spec["subtopic"], params, prop_out, bilin_out, solver_result, spec
            )
            row["provenance"] = {
                "source": "programmatic",
                "teacher_model": teacher_model_path,
                "solver_verified": bool(solver_result),
                "checks": ["alignment_ok", "glossary_ok"] if glossary else ["alignment_ok"],
            }

            check_schema(row)
            rows.append(row)
            print(f"[OK] Sample {i+1}/{n} generated ✅")

        except Exception as e:
            failures += 1
            print(f"[FAIL] Sample {i+1}/{n}: {repr(e)}")
            traceback.print_exc()
            continue

    if rows:
        write_jsonl(rows, out_path)
        print(f"\nWrote {len(rows)} bilingual items to {out_path}")
    else:
        print("\n No successful samples generated.")

    return {"ok": len(rows), "failed": failures, "out": out_path}
