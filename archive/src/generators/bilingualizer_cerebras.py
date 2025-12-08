import os, json, re
from cerebras.cloud.sdk import Cerebras
from .json_utils import parse_json_strict, ensure_keys, REQUIRED_BIL_KEYS

SYSTEM = (
    "Translate the following DSP problem+solution to Albanian (sq). "
    "STYLE: clear, technical, exam-style. "
    "STRICT RULES:\n"
    "1) Preserve ALL numbers and units EXACTLY (e.g., 32889, 9.942261, Hz). Do not change decimals.\n"
    "2) Preserve math symbols and variable names EXACTLY (Δf, k, N, fs, f0). Do NOT translate them.\n"
    "3) Do NOT translate code or function names (e.g., round(f0/Δf) stays round(f0/Δf)).\n"
    "4) Use these term choices in Albanian:\n"
    "   - Compute → Llogarit\n"
    "   - State whether → Trego nëse\n"
    "   - aliasing → aliasim\n"
    "   - alias frequency → frekuenca e aliasimit\n"
    "   - Nyquist frequency → frekuenca e Nyquist\n"
    "5) Output ONLY one JSON object with keys: problem_sq, solution_sq."
)

TERM_FIXES = [
    (r"\bKompute\b", "Llogarit"),
    (r"\bKomputo\b", "Llogarit"),
    (r"\bKompute(ni)?\b", "Llogarit"),
    (r"\bThërras\b", "Trego"),
    (r"\bali(a)?sing\b", "aliasim"),
    (r"\bAlias(ing)?\b", "aliasim"),
    (r"\bfrekuenc(ë|e) alias\b", "frekuenca e aliasimit"),
    (r"\bNyquist\b", "Nyquist"),
    (r"\basaj aliasing\b", "nuk ka aliasim"),
    (r"\brr(e|ë)th\(", "round("),  
]

def _postprocess_sq(text: str) -> str:
    for pat, rep in TERM_FIXES:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
   
    text = re.sub(r"\s+=\s+", " = ", text)
    text = re.sub(r"\s+Hz\b", " Hz", text)
    return text

class BilingualizerCerebras:
    def __init__(self, model_id: str):
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY not set")
        self.client = Cerebras(api_key=api_key)
        self.model_id = model_id

    def __call__(self, proposer_obj: dict) -> dict:
        payload = {
            "role": "user",
            "content": json.dumps(
                {
                    "problem_en": proposer_obj["problem_en"],
                    "solution_en": proposer_obj["solution_en"],
                },
                ensure_ascii=False,
            ),
        }
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": SYSTEM}, payload],
            temperature=0.06,  
            max_tokens=420,
        )

        
        choice = resp.choices[0]
        text = None
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            text = choice.message.content
        if not text and hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            text = choice.delta.content
        if not text and isinstance(choice, dict):
            text = choice.get("message", {}).get("content") or choice.get("delta", {}).get("content")
        if not text:
            raise ValueError("No content returned from Cerebras chat completion")

        obj = parse_json_strict(text)
        ensure_keys(obj, REQUIRED_BIL_KEYS)

       
        obj["problem_sq"] = _postprocess_sq(obj["problem_sq"])
        obj["solution_sq"] = _postprocess_sq(obj["solution_sq"])
        return obj
