# scripts/gen_fft_qwen3_235b.py

import os
import json
import time
import ast
from pathlib import Path

from cerebras.cloud.sdk import Cerebras  

MODEL_NAME = "qwen-3-235b-a22b-instruct-2507"

OUT_PATH = Path("data/interim/fft_qwen3_235b.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

GEN_PLAN = [
    ("bin_index_resolution", "numeric",        5, 40),
    ("multi_tone_spectrum",  "numeric",        5, 40),
    ("conceptual",           "conceptual",     6, 40),
    ("explanation",          "explanatory",    6, 40),
    ("debug_student",        "debug",          6, 40),
]

SYSTEM_PROMPT = """
You are a senior Digital Signal Processing (DSP) professor.

Your task is to GENERATE HIGH-QUALITY BILINGUAL (ENGLISH + ALBANIAN) DSP
QUESTION–ANSWER PAIRS ABOUT THE TOPIC: "DFT/FFT analysis of sinusoids".

We are working with N-point FFTs (DFTs) of sampled real sinusoids.

Core relationships:
- Sampling frequency: fs (Hz)
- FFT length: N (points)
- Frequency resolution: Δf = fs / N
- Continuous-time tone frequency: f0 (Hz)
- FFT bin index for a tone: k ≈ round(f0 / Δf)
- Nyquist frequency: f_Nyquist = fs / 2
- For real-valued signals, main lines appear at positive k (0 ≤ k ≤ N/2 for one-sided spectrum).

Requirements:
- All math must be EXACT and NUMERICALLY CORRECT.
- Use realistic sampling frequencies fs (8 kHz–192 kHz).
- Use reasonable FFT sizes N (e.g. 128, 256, 512, 1024, 2048, 4096).
- Use realistic tone frequencies f0 (positive, below 500 kHz).
- When you compute Δf, k, alias conditions, make sure the numbers are consistent.
- Avoid extremely awkward fractions; use decimals where needed.

Language & style:
- English: concise, technical, exam-style language.
- Albanian: correct DSP terminology, concise and technical.
- Use the same symbols in both languages (fs, f0, N, Δf, k, f_Nyquist).

Albanian terminology (use consistently):
- sampling frequency → frekuenca e mostrimit
- FFT / DFT → FFT / DFT
- frequency resolution → rezolucioni në frekuencë
- bin index → indeksi i bin-it
- magnitude spectrum → spektri i amplitudës
- Nyquist frequency → frekuenca Nyquist
- aliasing → aliasim
- leakage → rrjedhje spektrale
- window → dritare

Output format:
- OUTPUT A JSON ARRAY.
- EACH ELEMENT must strictly follow THIS SCHEMA:

  {
    "id": "<string, unique>",
    "lang_pair": ["en", "sq"],
    "topic": "FFT",
    "subtopic": "<one of: bin_index_resolution, multi_tone_spectrum, conceptual, explanation, debug_student>",
    "type": "<numeric | conceptual | explanatory | debug>",
    "system": "<short system message in Albanian for the STUDENT model>",
    "prompt_en": "<question or conversation in English>",
    "response_en": "<correct answer in English>",
    "prompt_sq": "<question or conversation in Albanian>",
    "response_sq": "<correct answer in Albanian>",
    "meta": {
      "fs": <number or null>,
      "N": <number or null>,
      "f0": <number or null>,
      "df": <number or null>,
      "k": <number or null>,
      "frequencies": "<optional description for multi-tone or null>",
      "difficulty": "<easy | medium | hard>",
      "has_aliasing": <true | false | null>
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
    if subtype == "bin_index_resolution":
        body = f"""
Generate {n_items} independent JSON items of subtype "bin_index_resolution" and type "numeric".

Each item should:
- Describe one real sinusoid with frequency f0 sampled at fs and analyzed with an N-point FFT.
- Ask the student to:
  - compute the frequency resolution Δf = fs / N,
  - find the FFT bin index k corresponding to f0 (usually k = round(f0 / Δf)),
  - state whether aliasing occurs relative to Nyquist (f0 vs fs/2).
- Provide clear numeric values for fs, N, and f0.
- Vary difficulty (easy, medium, hard) and ranges of fs, N, f0.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "multi_tone_spectrum":
        body = f"""
Generate {n_items} independent JSON items of subtype "multi_tone_spectrum" and type "numeric".

Each item should:
- Describe a signal containing 2 or 3 sinusoidal components with different frequencies.
- Give fs and N for the FFT.
- Ask the student to:
  - compute Δf = fs / N,
  - find the approximate FFT bin indices for each tone,
  - identify which bins (k values) will show spectral lines.
- You may also ask which tones are close together (spectral resolution issue), but keep it numeric.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "conceptual":
        body = f"""
Generate {n_items} independent JSON items of subtype "conceptual" and type "conceptual".

Each item should:
- Ask a short conceptual question about DFT/FFT, for example:
  - relationship between Δf, fs, and N,
  - why increasing N improves frequency resolution,
  - difference between time-domain sampling and frequency-domain resolution,
  - what happens if a tone is not exactly on a bin (leakage),
  - interpretation of bin index k in Hz.
- The answer must be technically correct and concise in both English and Albanian.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "explanation":
        body = f"""
Generate {n_items} independent JSON items of subtype "explanation" and type "explanatory".

Each item should:
- Present a concrete numerical scenario (fs, N, one or more f0 values).
- Ask the student to:
  - compute Δf,
  - find k for each tone,
  - discuss whether the FFT can resolve those tones (are they in different bins?),
  - optionally discuss leakage or bin mismatch.
- The answer should give a step-by-step explanation in both English and Albanian.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    elif subtype == "debug_student":
        body = f"""
Generate {n_items} independent JSON items of subtype "debug_student" and type "debug".

Each item should:
- Describe a student's WRONG reasoning about FFT-related quantities.
- The prompts (en/sq) should show:
  - the student's attempt (e.g., wrong Δf, wrong bin index k, confusion about fs and N),
  - and a teacher asking to check or correct the reasoning.
- The assistant/teacher response must:
  - point out the error,
  - show the correct calculations (Δf, k, etc.),
  - explain clearly why the student's interpretation is wrong.

Examples of typical mistakes:
- Using Δf = N / fs instead of fs / N.
- Confusing k with frequency directly.
- Assuming no leakage when f0 is not a multiple of Δf.
- Misinterpreting Nyquist vs FFT bin indices.

Remember:
- Return a JSON ARRAY of {n_items} items following the exact schema.
"""
    else:
        raise ValueError(f"Unknown subtype: {subtype}")

    return body.strip()


def extract_json_region(text: str) -> str:
    """
    Extract substring between first '[' and last ']'.
    If not found, just return the whole text.
    """
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
             
                    item.setdefault("topic", "FFT")
                    item.setdefault("subtopic", subtype)
                    item.setdefault("type", qtype)
                    item.setdefault("lang_pair", ["en", "sq"])

                    if "id" not in item or not item["id"]:
                        item["id"] = f"fft_{subtype}_{int(time.time()*1000)}_{num_written}"

                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line + "\n")
                    num_written += 1

                time.sleep(0.5)

    print(f"\nDone. Wrote ~{num_written} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
