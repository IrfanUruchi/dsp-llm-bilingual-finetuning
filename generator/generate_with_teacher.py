import os
import json
import time
from pathlib import Path

import yaml
from cerebras.cloud.sdk import Cerebras


def get_client() -> Cerebras:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY nuk është vendosur në environment.")
    return Cerebras(api_key=api_key)

client = get_client()


PROMPT_TEMPLATE = """
Gjenero NJË shembull pyetje–përgjigje në format JSON për një dataset trajnimi.

Kërkesat:
- Gjuha: vetëm shqip.
- Fusha: DSP, tema e përshkruar më poshtë.
- Pyetja duhet të jetë e qartë dhe realiste, si për studentë të vitit të parë ose të dytë.
- Përgjigjja duhet të ketë gjithmonë këtë strukturë:
  1) Një shpjegim i shkurtër me fjalë.
  2) Formulat matematikore përkatëse (p.sh. Δf = Fs / N, fₖ = k · Δf, Fs_min = 2 · f_max).
  3) Llogaritja numerike hap-pas-hapi kur ka numra.
  4) Rezultati përfundimtar i rrumbullakosur qartë.
- Formulat DUHET të jenë të sakta. Mos shpik formula të reja.
- Mos e përsërit pyetjen në përgjigje.
- Mos përdor gjuhë tjetër (pa anglisht, pa gjermanisht).

Tema:
{topic_description}

Kthe vetëm një objekt JSON me këtë strukturë:

{{
  "messages": [
    {{"role": "system", "content": "TEKST SISTEMI NË SHQIP"}},
    {{"role": "user", "content": "PYETJE NË SHQIP"}},
    {{"role": "assistant", "content": "PËRGJIGJE NË SHQIP"}}
  ]
}}
"""



def call_teacher(system_prompt: str, user_prompt: str) -> str:
    
    resp = client.chat.completions.create(
        model="llama3.3-70b",  
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=750,
    )

    msg = resp.choices[0].message
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")

def is_albanian_only(text: str) -> bool:
    text_low = text.lower()
    bad_tokens = [
        " the ", " and ", " or ",
        " und ", " ist ", " das ", " die ", " der ",
        "hello", "hi ", "yes ", "no ",
    ]
    return not any(b in text_low for b in bad_tokens)


def validate_example(ex: dict) -> bool:
    if "messages" not in ex:
        return False

    msgs = ex["messages"]
    if len(msgs) < 3:
        return False

    roles = [m.get("role") for m in msgs[:3]]
    if roles != ["system", "user", "assistant"]:
        return False

    for m in msgs:
        c = m.get("content", "")
        if not c.strip():
            return False
        if not is_albanian_only(c):
            return False
        if "<ë/>" in c or "()." in c:
            return False

    return True


def extract_json_block(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Nuk gjetëm bllok JSON nga modeli.")
    return raw[start:end+1]


def main():
    base = Path(__file__).resolve().parents[1]
    cfg_path = base / "configs" / "topics_dsp_v2.yaml"
    out_path = base / "generated" / "dsp_albanian_v1.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    global_sys = config["global_system_prompt"]

    total_good = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for topic in config["topics"]:
            name = topic["name"]
            n = topic["n_examples"]
            desc = topic["description"]

            print(f"\n== Tema: {name} (synimi: {n} shembuj) ==")

            for i in range(n):
                user_prompt = PROMPT_TEMPLATE.format(topic_description=desc)

                for attempt in range(3):
                    try:
                        raw = call_teacher(global_sys, user_prompt)
                        json_str = extract_json_block(raw)
                        ex = json.loads(json_str)

                        if validate_example(ex):
                            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                            print(f"  [{name}] {i+1}/{n} ✔️")
                            total_good += 1
                            break
                        else:
                            print(f"  [{name}] {i+1}/{n} ✖ invalid, tentimi {attempt+1}")
                    except Exception as e:
                        print(f"  [{name}] error: {e} (tentimi {attempt+1})")
                        time.sleep(1)

                else:
                    print(f"  [{name}] FAILED për shembull {i+1}/{n}")

    print("\n=====================================")
    print(f" Dataset i plotë: {total_good} shembuj OK")
    print(f" Ruajtur te: {out_path}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
