import json, uuid, time

def make_row(topic, subtopic, params, proposer, bilingual, solver_result, spec):
    pid = f"{topic}_{subtopic}_{uuid.uuid4().hex[:8]}"
    prompt_en = proposer["problem_en"].strip()
    response_en = proposer["solution_en"].strip()
    prompt_sq = bilingual["problem_sq"].strip()
    response_sq = bilingual["solution_sq"].strip()
    row = {
      "id": pid,
      "lang_pair": ["en","sq"],
      "topic": topic,
      "subtopic": subtopic,
      "type": "numeric",
      "prompt_en": prompt_en,
      "response_en": response_en,
      "prompt_sq": prompt_sq,
      "response_sq": response_sq,
      "meta": {
        "params": params,
        "solver_result": solver_result,
        "spec": {"glossary_tags": spec.get("glossary_tags",[])},
        "timestamp": int(time.time())
      }
    }
    return row

def write_jsonl(rows, out_path):
    with open(out_path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
