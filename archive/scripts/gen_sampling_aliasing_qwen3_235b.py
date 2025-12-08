# scripts/gen_sampling_aliasing_qwen3_235b.py
import os
import json
import time
import ast
from pathlib import Path

from cerebras.cloud.sdk import Cerebras 

MODEL_NAME = "qwen-3-235b-a22b-instruct-2507"

OUT_PATH = Path("data/interim/sampling_aliasing_qwen3_235b.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

GEN_PLAN = [
    ("basic_numeric", "numeric",        5, 40),   
    ("multi_tone",    "numeric",        5, 40),
    ("conceptual",    "conceptual",     6, 40), 
    ("explanation",   "explanatory",    6, 40),
    ("debug_student", "debug",          6, 40),  
]

SYSTEM_PROMPT = """
You are a senior Digital Signal Processing (DSP) professor.

Your task is to GENERATE HIGH-QUALITY BILINGUAL (ENGLISH + ALBANIAN) DSP
QUESTION–ANSWER PAIRS ABOUT THE TOPIC: "Sampling and aliasing".

Requirements:
- All math must be EXACT and NUMERICALLY CORRECT.
- Use realistic sampling frequencies fs (between about 8000 Hz and 192000 Hz).
- Use realistic signal frequencies f0 (positive, below 500000 Hz).
- Always respect Nyquist: f_Nyquist = fs / 2.
- When aliasing happens, explicitly state the ALIAS FREQUENCY / FREKUENCA E ALIASIMIT.

Language & style:
- English: concise, technical, exam-style language.
- Albanian: correct DSP terminology, concise and technical.
- Use the same symbols in both languages (fs, f0, f_Nyquist, Δf, etc.).

Albanian terminology (use consistently):
- sampling           → mostrim
- sampling frequency → frekuenca e mostrimit
- Nyquist frequency  → frekuenca Nyquist
- aliasing           → aliasim
- continuous-time signal → sinjal në kohë të vazhdueshme
- discrete-time signal   → sinjal diskret (ose sinjal i mostruar)

Output format:
- OUTPUT A JSON ARRAY.
- EACH ELEMENT must strictly follow THIS SCHEMA:

  {
    "id": "<string, unique>",
    "lang_pair": ["en", "sq"],
    "topic": "sampling_aliasing",
    "subtopic": "<one of: basic_numeric, multi_tone, conceptual, explanation, debug_student>",
    "type": "<numeric | conceptual | explanatory | debug>",
    "system": "<short system message in Albanian for the STUDENT model>",
    "prompt_en": "<question or conversation in English>",
    "response_en": "<correct answer in English>",
    "prompt_sq": "<question or conversation in Albanian>",
    "response_sq": "<correct answer in Albanian>",
    "meta": {
      "fs": <number or null>,
      "f0": <number or null>,
      "frequencies": "<optional description for multi-tone or null>",
      "difficulty": "<easy | medium | hard>",
      "has_aliasing": <true | false>
    }
  }

Rules:
- Do NOT include markdown.
- Do NOT include comments.
- Do NOT add backticks.
- All numbers and formulae must be consistent between English and Albanian versions.
- The Albanian text must be natural, not word-for-word from English.

CRITICAL FORMAT RULES (MUST OBEY):
- Your ENTIRE reply MUST be exactly one valid JSON ARRAY.
- DO NOT write any text before or after the JSON array.
- DO NOT include explanations, comments, or notes.
- Use only double quotes (") for keys and string values.
- Do NOT put unescaped line breaks inside string values.
- If you need multiple lines in a string, use "\\n".
- Do not use trailing commas.
"""


def build_user_prompt(subtype: str, qtype: str, n_items: int) -> str:
    """
    Create the user prompt that tells Qwen what kind of items to generate.
    """
    if subtype == "basic_numeric":
        body = f"""
Generate {n_items} independent JSON items of subtype "basic_numeric" and type "numeric".

Each item should:
- Involve ONE sinusoid sampled at fs.
- Given fs and f0, ask the student to:
  - compute or state the Nyquist frequency,
  - decide whether aliasing occurs,
  - and give the alias frequency if aliasing happens.
- Use diverse ranges for fs and f0 so questions are not repetitive.
- Vary difficulty between easy, medium, and hard.
- Sometimes choose f0 < fs/2 (no aliasing), sometimes f0 > fs/2 (aliasing).

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "multi_tone":
        body = f"""
Generate {n_items} independent JSON items of subtype "multi_tone" and type "numeric".

Each item should:
- Involve 2 or 3 sinusoids sampled at the same fs.
- Ask the student which tones alias and what their alias frequencies are.
- Vary situations:
  - some tones just below Nyquist,
  - some just above,
  - some far above.
- Mix cases where some tones alias and some do not.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "conceptual":
        body = f"""
Generate {n_items} independent JSON items of subtype "conceptual" and type "conceptual".

Each item should:
- Ask a short conceptual question about sampling and aliasing.
- Examples of themes:
  - Why aliasing occurs when sampling too slowly.
  - Relationship between continuous frequency and discrete-time frequency.
  - Effect of anti-aliasing filters.
  - Practical examples (audio, communications, sensors).
- Answers must be technically correct and concise.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "explanation":
        body = f"""
Generate {n_items} independent JSON items of subtype "explanation" and type "explanatory".

Each item should:
- Present a concrete numerical scenario (fs, one or more f0).
- Ask the student to:
  - check for aliasing,
  - find alias frequencies,
  - AND explain step-by-step why.
- The answer should include a clear, ordered explanation in both English and Albanian.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "debug_student":
        body = f"""
Generate {n_items} independent JSON items of subtype "debug_student" and type "debug".

Each item should:
- Describe a student's WRONG reasoning or answer about sampling/aliasing.
- The prompt fields (en/sq) must show the student attempt plus a teacher asking for correction.
- The assistant/teacher response must gently:
  - point out the error,
  - give the correct reasoning and answer.

Examples of student mistakes:
- Miscomputing Nyquist (e.g. fs instead of fs/2).
- Claiming no aliasing when f0 >> fs/2.
- Confusing continuous-time and discrete-time frequencies.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    else:
        raise ValueError(f"Unknown subtype: {subtype}")

    return body.strip()


def extract_json_region(text: str) -> str:

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


def extract_objects_loose(text: str):

    objs = []
    n = len(text)
    in_string = False
    escape = False
    brace_depth = 0
    start_idx = None

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif ch == "}":
                if brace_depth > 0:
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        obj_str = text[start_idx:i+1]
                        try:
                            obj = json.loads(obj_str)
                        except json.JSONDecodeError:
                            try:
                                obj = ast.literal_eval(obj_str)
                            except Exception:
                                obj = None
                        if isinstance(obj, dict):
                            objs.append(obj)
                        start_idx = None

    return objs

def get_client() -> Cerebras:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY env var is not set.")
    return Cerebras(api_key=api_key)


def call_teacher(client: Cerebras, user_prompt: str, max_tokens: int = 4000) -> list:

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )

    text = completion.choices[0].message.content
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in text
        )
    text = text.strip()

    raw_debug_path = OUT_PATH.with_suffix(".raw_last.txt")
    raw_debug_path.write_text(text, encoding="utf-8")

    region = extract_json_region(text)
    items = extract_objects_loose(region)

    if not items:
        bad_debug_path = OUT_PATH.with_suffix(".bad_json.txt")
        bad_debug_path.write_text(region, encoding="utf-8")
        raise ValueError("No valid JSON objects found in model output.")

    return items

def main():
    client = get_client()
    print(f"Writing to {OUT_PATH}")
    num_written = 0

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for subtype, qtype, runs, items_per_run in GEN_PLAN:
            for run_idx in range(runs):
                print(f"\n=== {subtype} | run {run_idx+1}/{runs} | target {items_per_run} items ===")
                user_prompt = build_user_prompt(subtype, qtype, items_per_run)

                try:
                    items = call_teacher(client, user_prompt)
                except Exception as e:
                    print(f"Error on {subtype} run {run_idx+1}: {e}")
                    time.sleep(3)
                    continue

                print(f"Parsed {len(items)} objects from model output.")

                for item in items:
                    item.setdefault("topic", "sampling_aliasing")
                    item.setdefault("subtopic", subtype)
                    item.setdefault("type", qtype)
                    item.setdefault("lang_pair", ["en", "sq"])

                    if "id" not in item or not item["id"]:
                        item["id"] = f"{subtype}_{int(time.time()*1000)}_{num_written}"

                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + "\n")
                    num_written += 1

                time.sleep(0.5)

    print(f"\nDone. Wrote ~{num_written} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
